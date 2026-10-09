# offline_preprocess.py
"""
Offline preprocessing pipeline for debt-collection dialogue.

Produces two JSONL files:
  - session_chunks.jsonl : one row per call, containing session-level tags
  - pair_chunks.jsonl    : one row per customer/agent dialogue pair

Pipeline:
  Stage 1  ASR text   -> structured turns
  Stage 2  turns      -> pair-level chunks
  Stage 3  full call  -> session-level chunk   (1 LLM call per call)
  Stage 4  pair chunk -> pair-level metadata   (concurrent LLM calls)
  Stage 5  human label -> merged into pair chunk metadata
  Stage 6  linkage    -> prev/next pointers, topic/stage assignment

Config:
  .env             -> LLM + concurrency + paths
  prompts/prompts.yaml -> all prompt templates

Run:
  python offline_preprocess.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config import AppConfig, load_config
from utils.llm_client import LLMClient, build_llm_client
from utils.logger import setup_logger
from utils.text_utils import (
    AGENT_NAMES,
    CUSTOMER_NAMES,
    ROLE_PATTERN,
    extract_quoted_spans,
    fuzzy_contains,
)

logger = setup_logger(Path(__file__).resolve().name)


# ======================================================================
# Data models
# ======================================================================


@dataclass
class Turn:
    """One utterance in the conversation."""

    turn_id: int
    role: str  # "agent" | "customer"
    text: str
    char_len: int


@dataclass
class PairChunk:
    """A pair-level chunk: one contiguous agent run + one customer run."""

    chunk_id: str
    chunk_type: str  # "pair" | "single_agent" | "single_customer"
    call_id: str
    turn_start: int
    turn_end: int
    role_sequence: List[str]
    content: str
    char_len: int
    label: Dict[str, Any] = field(default_factory=dict)

    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    session_chunk_id: Optional[str] = None
    topic_label: Optional[str] = None
    topic_segment_id: Optional[str] = None
    topic_position_in_session: Optional[int] = None

    call_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionChunk:
    """Session-level chunk: one row per call."""

    chunk_id: str
    call_id: str
    content: str
    char_len: int
    chunk_type: str = "session"
    label: Dict[str, Any] = field(default_factory=dict)
    call_meta: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Stage 1: ASR text -> structured turns
# ======================================================================


def stage1_parse_turns(call_id: str, text: str) -> List[Turn]:
    raw_turns = re.split(r"[；;]", text or "")
    turns: List[Turn] = []
    turn_id = 0

    for raw in raw_turns:
        raw = raw.strip()
        if not raw:
            continue

        m = ROLE_PATTERN.match(raw)
        if m:
            role_raw, content = m.group(1), m.group(2).strip()
            if role_raw in AGENT_NAMES:
                role = "agent"
            elif role_raw in CUSTOMER_NAMES:
                role = "customer"
            else:
                role = "agent"
            if content:
                turn_id += 1
                turns.append(
                    Turn(
                        turn_id=turn_id,
                        role=role,
                        text=content,
                        char_len=len(content),
                    )
                )
        else:
            if turns and raw:
                turns[-1].text = f"{turns[-1].text}；{raw}"
                turns[-1].char_len = len(turns[-1].text)

    return turns


# ======================================================================
# Stage 2: turns -> pair chunks
# ======================================================================


def _collect_same_role_run(turns: List[Turn], start: int) -> Tuple[List[Turn], int]:
    role = turns[start].role
    group = [turns[start]]
    j = start + 1
    while j < len(turns) and turns[j].role == role:
        group.append(turns[j])
        j += 1
    return group, j


def stage2_build_pair_chunks(
    call_id: str,
    turns: List[Turn],
    call_meta: Dict[str, Any],
) -> List[PairChunk]:
    pairs: List[PairChunk] = []
    i = 0
    while i < len(turns):
        group, j = _collect_same_role_run(turns, i)

        if j < len(turns):
            next_group, k = _collect_same_role_run(turns, j)
            combined = group + next_group
            i = k
        else:
            combined = group
            i = j

        if len(combined) == 1:
            chunk_type = (
                "single_agent" if combined[0].role == "agent" else "single_customer"
            )
        else:
            chunk_type = "pair"

        content = "\n".join(
            f"{'客户' if t.role == 'customer' else '催收员'}：{t.text}"
            for t in combined
        )

        pairs.append(
            PairChunk(
                chunk_id=f"pair_{call_id}_t{combined[0].turn_id}_t{combined[-1].turn_id}",
                chunk_type=chunk_type,
                call_id=call_id,
                turn_start=combined[0].turn_id,
                turn_end=combined[-1].turn_id,
                role_sequence=[t.role for t in combined],
                content=content,
                char_len=len(content),
                call_meta=call_meta,
            )
        )

    return pairs


# ======================================================================
# Stage 3: session-level chunk (1 LLM call per call)
# ======================================================================


def _format_turns_with_ids(turns: List[Turn]) -> str:
    return f"[Turn No.]角色: 对话内容\n" + "\n".join(
        f"[{t.turn_id}]{'催收员: ' if t.role == 'agent' else '客户: '} {t.text}"
        for t in turns
    )


async def stage3_build_session_chunk(
    llm: LLMClient,
    session_text: str,
    prompt_template: str,
    call_id: str,
    turns: List[Turn],
    call_meta: Dict[str, Any],
) -> SessionChunk:

    prompt = prompt_template.format(session_text=session_text)

    try:
        data = await llm.call_json(prompt)
        if isinstance(data, dict):
            data.setdefault("label_source", "llm")
    except Exception as e:  # noqa: BLE001
        logger.error("Stage 3 session metadata failed for %s: %s", call_id, e)
        data = {"error": str(e)}

    return SessionChunk(
        chunk_id=f"session_{call_id}",
        call_id=call_id,
        content=session_text,
        char_len=sum(t.char_len for t in turns),
        label=data,
        call_meta=call_meta,
    )


# ======================================================================
# Stage 4: pair-level metadata (concurrent LLM calls)
# ======================================================================


async def stage4_extract_pair_meta(
    llm: LLMClient,
    prompt_template: str,
    session_text: str,
    chunk: PairChunk,
    pair_sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with pair_sem:
        prompt = prompt_template.format(
            session_content=session_text, chunk_content=chunk.content
        )
        try:
            data = await llm.call_json(prompt)
            if isinstance(data, dict):
                data.setdefault("label_source", "llm")
        except Exception as e:  # noqa: BLE001
            logger.error("Stage 4 failed for %s: %s", chunk.chunk_id, e)
            return {"error": str(e), "label_source": "error"}
        return data


# ======================================================================
# Stage 5: human label fusion
# ======================================================================


def stage5_parse_label(label_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not label_text:
        return rows

    for line in label_text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        if "类型" in line and "执行情况" in line:
            continue

        stripped = line.replace("|", "").strip()
        if stripped and set(stripped) <= set("-: "):
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 3:
            continue

        action_type, status, evidence_raw = parts[0], parts[1], parts[2]
        evidence = extract_quoted_spans(evidence_raw)
        if not evidence and evidence_raw.strip():
            evidence = [evidence_raw.strip()]

        rows.append(
            {
                "action_type": action_type,
                "execution_status": status,
                "evidence": evidence,
            }
        )

    return rows


def stage5_fuse_label_into_pairs(
    pair_chunks: List[PairChunk],
    label_structured: List[Dict[str, Any]],
) -> List[PairChunk]:
    for item in label_structured:
        action_type = item["action_type"]
        status = item["execution_status"]
        evidences = item["evidence"]

        for chunk in pair_chunks:
            if not any(fuzzy_contains(ev, chunk.content) for ev in evidences):
                continue

            meta = chunk.label

            at_list = list(meta.get("action_type") or [])
            if action_type not in at_list:
                at_list.append(action_type)
            meta["action_type"] = at_list

            # Human overrides LLM
            meta["execution_status"] = status

            ev_list = list(meta.get("evidence") or [])
            for ev in evidences:
                if ev not in ev_list:
                    ev_list.append(ev)
            meta["evidence"] = ev_list
            meta["label_source"] = "human"

    return pair_chunks


# ======================================================================
# Stage 6: linkage
# ======================================================================


def stage6_build_linked_list(pair_chunks: List[PairChunk]) -> List[PairChunk]:
    ordered = sorted(pair_chunks, key=lambda c: c.turn_start)
    for i, c in enumerate(ordered):
        c.prev_chunk_id = ordered[i - 1].chunk_id if i > 0 else None
        c.next_chunk_id = ordered[i + 1].chunk_id if i < len(ordered) - 1 else None
    return ordered


def stage6_assign_session_and_topic(
    pair_chunks: List[PairChunk],
    session: SessionChunk,
) -> List[PairChunk]:
    for c in pair_chunks:
        c.session_chunk_id = session.chunk_id

        matched_idx: Optional[int] = None
        matched_label: Optional[str] = None
        source: Optional[str] = None

        topics = session.label.get("topics") or []
        for idx, t in enumerate(topics):
            ts, te = t.get("turn_start", -1), t.get("turn_end", -1)
            if ts <= c.turn_start and c.turn_end <= te:
                matched_idx, matched_label, source = idx, t.get("label", ""), "topic"
                break

        if matched_label is None:
            stages = session.label.get("stages") or []
            for idx, s in enumerate(stages):
                ts, te = s.get("turn_start", -1), s.get("turn_end", -1)
                if ts <= c.turn_start and c.turn_end <= te:
                    matched_idx, matched_label, source = (
                        idx,
                        s.get("label", ""),
                        "stage",
                    )
                    break

        if matched_label is not None:
            c.topic_label = matched_label
            c.topic_segment_id = f"{source}_{session.call_id}_{matched_idx}"
            c.topic_position_in_session = matched_idx

    return pair_chunks


# ======================================================================
# Orchestrator
# ======================================================================


class OfflinePreprocessor:
    """Runs Stage 1 ~ Stage 6 for a batch of raw samples.

    Args:
        cfg: AppConfig loaded from .env + prompts.yaml.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.llm = build_llm_client(cfg.llm)
        self.session_prompt = cfg.prompts.get("session_prompt")
        self.pair_prompt = cfg.prompts.get("pair_prompt")
        if not self.session_prompt or not self.pair_prompt:
            raise RuntimeError(
                "prompts.yaml must define both 'session_prompt' and 'pair_prompt'"
            )

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.call_sem = asyncio.Semaphore(cfg.call_concurrency)
        self.pair_sem = asyncio.Semaphore(cfg.pair_concurrency)

        self.f_session = self.output_dir / "session_chunks.jsonl"
        self.f_pair = self.output_dir / "pair_chunks.jsonl"
        self.f_fail = self.output_dir / "failures.jsonl"

    # ------------------------------------------------------------------
    # Per-call pipeline
    # ------------------------------------------------------------------
    async def _process_one_call(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        call_id = raw["call_id"]
        text = raw.get("dialog", "")
        label_text = raw.get("label", "") or ""

        call_meta = {
            "call_id": call_id,
            "calldate": raw.get("calldate"),
            "custno": raw.get("custno"),
            "colluserid": raw.get("colluserid"),
            "mobtyp": raw.get("mobtyp"),
            "talktime": raw.get("talktime"),
        }

        # Stage 1
        turns = stage1_parse_turns(call_id, text)
        if not turns:
            raise ValueError(f"No turns parsed for {call_id}")

        # Stage 2
        pair_chunks = stage2_build_pair_chunks(call_id, turns, call_meta)

        session_text = _format_turns_with_ids(turns)

        # Stage 3
        session_chunk = await stage3_build_session_chunk(
            self.llm, session_text, self.session_prompt, call_id, turns, call_meta
        )

        # Stage 4 (concurrent)
        async def _one(chunk: PairChunk) -> PairChunk:
            meta = await stage4_extract_pair_meta(
                self.llm, self.pair_prompt, session_text, chunk, self.pair_sem
            )
            chunk.label = meta
            return chunk

        pair_chunks = list(await asyncio.gather(*[_one(c) for c in pair_chunks]))

        # Stage 5 : this is not concurrent implemented given that label_text is empty
        label_structured = stage5_parse_label(label_text)
        if label_structured:
            pair_chunks = stage5_fuse_label_into_pairs(pair_chunks, label_structured)
            logger.info(
                "Stage 5: fused %d label rows for %s", len(label_structured), call_id
            )
        else:
            logger.debug("Stage 5: no label provided for %s, skipping fusion", call_id)

        # Stage 6
        pair_chunks = stage6_build_linked_list(pair_chunks)
        pair_chunks = stage6_assign_session_and_topic(pair_chunks, session_chunk)

        return {"session": session_chunk, "pairs": pair_chunks}

    # ------------------------------------------------------------------
    # Batch orchestrator
    # ------------------------------------------------------------------
    async def run(self, samples: List[Dict[str, Any]], batch_size: int = 50) -> None:
        state = {
            "n_sessions": 0,
            "n_pairs": 0,
            "n_fail": 0,
            "first_session": True,
            "first_pair": True,
            "first_fail": True,
        }

        # 1. Initialize files as valid JSON arrays
        fs = open(self.f_session, "w", encoding="utf-8")
        fp = open(self.f_pair, "w", encoding="utf-8")
        ff = open(self.f_fail, "w", encoding="utf-8")

        fs.write("[\n")
        fp.write("[\n")
        ff.write("[\n")

        async def _process_task(raw: Dict[str, Any]) -> Dict[str, Any]:
            async with self.call_sem:
                try:
                    return await self._process_one_call(raw)
                except Exception as e:  # noqa: BLE001
                    logger.exception("Failed to process %s: %s", raw.get("call_id"), e)
                    return {"error": str(e), "call_id": raw.get("call_id")}

        try:
            # 2. Schedule ALL LLM requests immediately
            tasks = [_process_task(s) for s in samples]

            # Buffers to hold data in memory until batch_size is reached
            session_buf, pair_buf, fail_buf = [], [], []
            buffered_count = 0

            def _flush_buffers():
                """Helper to write all buffers to disk and clear them."""
                if fail_buf:
                    ff.write("".join(fail_buf))
                    fail_buf.clear()
                if session_buf:
                    fs.write("".join(session_buf))
                    session_buf.clear()
                if pair_buf:
                    fp.write("".join(pair_buf))
                    pair_buf.clear()

            # 3. Yield results AS THEY COMPLETE
            for coro in asyncio.as_completed(tasks):
                res = await coro

                # Format and append to our memory buffers
                if "error" in res:
                    prefix = "" if state["first_fail"] else ",\n"
                    fail_buf.append(prefix + json.dumps(res, ensure_ascii=False))
                    state["first_fail"] = False
                    state["n_fail"] += 1
                else:
                    prefix_s = "" if state["first_session"] else ",\n"
                    session_buf.append(
                        prefix_s
                        + json.dumps(asdict(res["session"]), ensure_ascii=False)
                    )
                    state["first_session"] = False
                    state["n_sessions"] += 1

                    for p in res["pairs"]:
                        prefix_p = "" if state["first_pair"] else ",\n"
                        pair_buf.append(
                            prefix_p + json.dumps(asdict(p), ensure_ascii=False)
                        )
                        state["first_pair"] = False
                        state["n_pairs"] += 1

                buffered_count += 1

                # 4. Write to disk only when we accumulate enough returns
                if buffered_count >= batch_size:
                    _flush_buffers()
                    buffered_count = 0

            # 5. Flush any leftover items after all tasks are done
            if buffered_count > 0:
                _flush_buffers()

        finally:
            # 6. Ensure valid JSON array closure even on KeyboardInterrupt
            fs.write("\n]\n")
            fp.write("\n]\n")
            ff.write("\n]\n")

            fs.close()
            fp.close()
            ff.close()

        logger.info(
            "Done. sessions=%d pairs=%d failures=%d",
            state["n_sessions"],
            state["n_pairs"],
            state["n_fail"],
        )
        logger.info("Output dir: %s", self.output_dir.resolve())


# ======================================================================
# File loader & CLI
# ======================================================================


def _load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    if not content.strip():
        return []

    # 1) Whole file as a single JSON value (object or array)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")

    # 2) Fallback: JSONL / NDJSON
    samples: List[Dict[str, Any]] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as line_err:
            raise ValueError(
                f"Not valid as JSONL on line {lineno}: {line_err}\n"
                f"Line preview: {line[:2000]!r}"
            ) from line_err
        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected JSON object on line {lineno}, got {type(obj).__name__}"
            )
        samples.append(obj)
    return samples


async def _amain(args: argparse.Namespace) -> None:
    cfg = load_config(
        env_path=args.env,
        prompts_path=args.prompts,
    )

    logger.info("LLM provider=%s model=%s", cfg.llm.provider, cfg.llm.model)
    logger.info(
        "call_concurrency=%d pair_concurrency=%d",
        cfg.call_concurrency,
        cfg.pair_concurrency,
    )

    input_path = args.input or cfg.input_path
    samples = _load_samples(input_path)
    logger.info("Loaded %d samples from %s", len(samples), input_path)

    pre = OfflinePreprocessor(cfg)
    await pre.run(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline preprocessing for debt-collection dialogue."
    )
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to prompts.yaml (overrides PROMPTS_PATH in .env)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to samples.json (overrides INPUT_PATH in .env)",
    )
    args = parser.parse_args()

    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
