"""
kt_gen.py — Knowledge tree construction with ontology-aware Pydantic extraction.

Pipeline phases (independently callable for debugging):
  1. construct(corpus)     — LLM extraction, expensive. Saves a checkpoint.
  2. postprocess()          — triple_deduplicate + Level-4 community detection.
  3. build_knowledge_graph(corpus, debug=False)  — orchestrates 1 → 2.

Debug workflow:
  # Run 1 (cold): construct + checkpoint + postprocess
  KTBuilder("debt_collection").build_knowledge_graph("corpus.json")

  # Run 2 (warm): reload checkpoint, skip construction
  KTBuilder("debt_collection").build_knowledge_graph("corpus.json", debug=True)
"""

import hashlib
import json
import os
import pickle
import threading
import time
from concurrent import futures
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type

import nanoid
import networkx as nx
import tiktoken
import json_repair
from pydantic import BaseModel, ValidationError

from config import get_config
from utils import call_llm_api, graph_processor, tree_comm
from utils.logger import logger

from models.m1_sop_models import DebtCollectionExtraction

# =============================================================================
# Ontology enum → node-type mapping
# =============================================================================
# Enums listed here become graph NODES when they appear as field values.
# Their member values (e.g., "Stage_1_Opening") are used to build node IDs,
# matching the ID convention that `_resolve_node_id` produces for the
# corresponding model instance.
#
# Enums NOT in this set (e.g., RFDEnum) remain scalar PROPERTIES on their
# parent node — they are classification labels, not entities.
_ENUM_NODE_TYPES = {
    "MasterSOPStage",
    "SubStage",
    "ComplianceArtifact",
    "ObjectionCategory",
    "CallOutcome",
    "RFD",
    # Extend as more ontology-backed enums are added, e.g.:
    # "DebtorPersona",
    # "PersuasionStrategy",
}

# =============================================================================
# KTBuilder
# =============================================================================


class KTBuilder:
    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        dataset_name: str,
        schema_path: Optional[str] = None,
        mode: Optional[str] = None,
        config=None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ):
        if config is None:
            config = get_config()

        self.config = config
        self.dataset_name = dataset_name
        self.schema = self._load_schema(
            schema_path or config.get_dataset_config(dataset_name).schema_path
        )

        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.lock = threading.RLock()

        self.datasets_no_chunk = config.construction.datasets_no_chunk
        self.token_len = 0
        self.token_len_per_chunk: Dict[str, int] = {}

        self.llm_client = call_llm_api.LLMCompletionCall()
        self.all_chunks: Dict[str, str] = {}
        self.llm_responses: Dict[str, str] = {}
        self._response_cache_lock = threading.Lock()
        self._load_llm_response_cache()

        self.mode = mode or config.construction.mode

        self.pydantic_model: Optional[Type[BaseModel]] = (
            pydantic_model if pydantic_model is not None else DebtCollectionExtraction
        )

        self.metrics = {
            "chunks_total": 0,
            "chunks_validated": 0,
            "chunks_failed_validation": 0,
            "chunks_failed_json": 0,
            "nodes_emitted": 0,
            "edges_emitted": 0,
        }

        if self.mode == "pydantic" and self.pydantic_model is None:
            raise ValueError(
                "mode='pydantic' requires a Pydantic model; pass "
                "pydantic_model=... or set KTBuilder.pydantic_model."
            )

        # Debug/observability flags
        self._constructed = False  # Set True once construct() completes
        self._postprocessed = False  # Set True once postprocess() completes

    # -------------------------------------------------------------------------
    # Schema / model loading
    # -------------------------------------------------------------------------

    def _load_schema(self, schema_path: Optional[str]) -> Dict[str, Any]:
        if not schema_path:
            return {}
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Schema file not found: {schema_path}")
            return {}

    @lru_cache(maxsize=8)
    def _serialized_pydantic_schema(self, model_repr: str) -> str:
        assert self.pydantic_model is not None
        return json.dumps(
            self.pydantic_model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

    def _get_pydantic_schema_str(self) -> str:
        if self.pydantic_model is None:
            raise ValueError("No Pydantic model bound.")
        key = f"{self.pydantic_model.__module__}.{self.pydantic_model.__qualname__}"
        return self._serialized_pydantic_schema(key)

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def _split_text_with_overlap(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_tail_tokens: int = 100,
    ) -> List[str]:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("gpt2")

        tokens = encoding.encode(text)
        if len(tokens) <= chunk_size:
            return [text]

        windows: List[List[int]] = []
        start = 0
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            windows.append([start, end])
            start += step

        if len(windows) >= 2:
            idx = 0
            while idx < len(windows):
                cur_len = windows[idx][1] - windows[idx][0]
                if cur_len < min_tail_tokens:
                    if idx > 0:
                        windows[idx - 1][1] = windows[idx][1]
                        windows.pop(idx)
                        continue
                    elif idx + 1 < len(windows):
                        windows[idx + 1][0] = windows[idx][0]
                        windows.pop(idx)
                        continue
                idx += 1

        chunks: List[str] = []
        for s, e in windows:
            decoded = encoding.decode(tokens[s:e])
            if (e - s < 5) or (len(decoded.strip()) < 5):
                continue
            chunks.append(decoded)
        return chunks

    def chunk_text(self, text) -> Tuple[List[str], Dict[str, str]]:
        if self.dataset_name in self.datasets_no_chunk:
            chunks = [
                (
                    f"Labels='{text.get('title', '')}' Content='{text.get('text', '')}'".strip()
                    if isinstance(text, dict)
                    else str(text)
                )
            ]
        else:
            raw_text = str(text)
            if isinstance(text, dict):
                raw_text = (
                    f"Labels='{text.get('title', '')}' {text.get('text', '')}".strip()
                )
            chunk_size = getattr(self.config.construction, "chunk_size", 1000)
            overlap = getattr(self.config.construction, "overlap", 200)
            min_tail_tokens = getattr(self.config.construction, "min_tail_tokens", 100)
            chunks = self._split_text_with_overlap(
                raw_text, chunk_size, overlap, min_tail_tokens
            )

        chunk2id: Dict[str, str] = {}
        for chunk in chunks:
            chunk_id = self._stable_chunk_id(chunk)
            chunk2id[chunk_id] = chunk

        with self.lock:
            self.all_chunks.update(chunk2id)

        return chunks, chunk2id

    @staticmethod
    def _stable_chunk_id(chunk: str) -> str:
        return hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]

    # -------------------------------------------------------------------------
    # Text cleaning
    # -------------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        if not text:
            return "[EMPTY_TEXT]"
        if self.dataset_name == "graphrag-bench":
            safe_chars = {*" .:,!?()-+=[]{}()\\/|_^~<>*&$#@!;\"'`"}
        else:
            safe_chars = {*" .:,!?()-+="}
        cleaned = "".join(
            ch for ch in text if ch.isalnum() or ch.isspace() or ch in safe_chars
        ).strip()
        return cleaned or "[EMPTY_AFTER_CLEANING]"

    # -------------------------------------------------------------------------
    # Chunk persistence
    # -------------------------------------------------------------------------

    def save_chunks_to_file(self):
        os.makedirs("output/chunks", exist_ok=True)
        chunk_file = f"output/chunks/{self.dataset_name}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in self.all_chunks.items():
                escaped = chunk_text.replace("\n", "\\n").replace("\t", "\\t")
                f.write(f"id: {chunk_id}\tChunk: {escaped}\n")
        logger.info(f"Chunk data saved to {chunk_file} ({len(self.all_chunks)} chunks)")

    # -------------------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------------------

    def _get_construction_prompt(self, chunk: str) -> str:
        construction_prompts = self.config.prompts["construction"]

        base_prompt_type = (
            self.dataset_name
            if self.dataset_name in construction_prompts
            else "general"
        )

        if self.mode == "pydantic":
            candidate = f"{base_prompt_type}_pydantic"
            prompt_type = (
                candidate
                if candidate in construction_prompts
                else "debt_collection_agent_pydantic"
            )
            if prompt_type not in construction_prompts:
                raise KeyError(
                    f"No pydantic construction prompt registered; expected "
                    f"'{candidate}' or fallback 'debt_collection_agent_pydantic'."
                )
            recommend_schema = self._get_pydantic_schema_str()
        else:
            recommend_schema = json.dumps(self.schema, ensure_ascii=False)
            prompt_type = (
                f"{base_prompt_type}_agent"
                if self.mode == "agent"
                else base_prompt_type
            )

        return self.config.get_prompt_formatted(
            "construction", prompt_type, schema=recommend_schema, chunk=chunk
        )

    # -------------------------------------------------------------------------
    # LLM interaction
    # -------------------------------------------------------------------------

    def extract_with_llm(self, prompt: str) -> str:
        response = self.llm_client.call_api(prompt)
        parsed_dict = json_repair.loads(response)
        return json.dumps(parsed_dict, ensure_ascii=False)

    def token_cal(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _track_tokens(self, chunk_id: str, prompt: str, response: Optional[str]):
        cost = self.token_cal(prompt + (response or ""))
        with self.lock:
            self.token_len += cost
            self.token_len_per_chunk[chunk_id] = (
                self.token_len_per_chunk.get(chunk_id, 0) + cost
            )

    def _validate_and_parse_llm_response(
        self, prompt: str, llm_response: str
    ) -> Optional[dict]:
        if llm_response is None:
            return None
        try:
            return json_repair.loads(llm_response)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Pydantic validation
    # -------------------------------------------------------------------------

    def _validate_pydantic_response(
        self, raw_response: Optional[str]
    ) -> Optional[BaseModel]:
        if raw_response is None or self.pydantic_model is None:
            return None
        try:
            data = json_repair.loads(raw_response)
        except Exception as e:
            self.metrics["chunks_failed_json"] += 1
            logger.warning(
                f"[{self.dataset_name}] LLM output was not valid JSON: "
                f"{type(e).__name__}: {e}"
            )
            return None

        try:
            return self.pydantic_model.model_validate(data)
        except ValidationError as e:
            self.metrics["chunks_failed_validation"] += 1
            first = e.errors()[0] if e.errors() else {}
            logger.warning(
                f"[{self.dataset_name}] Pydantic validation failed "
                f"({e.error_count()} error(s)); first={first}"
            )
            return None

    # -------------------------------------------------------------------------
    # Graph walk — Pydantic path
    # -------------------------------------------------------------------------

    def _resolve_node_id(self, obj: BaseModel, prefix: str) -> str:
        for field_name in type(obj).model_fields:
            if field_name.endswith("_id"):
                val = getattr(obj, field_name, None)
                if val is None:
                    continue
                if isinstance(val, Enum):
                    val = val.value
                return f"{prefix}__{val}"

        with self.lock:
            node_id = f"{prefix}__{self.node_counter}"
            self.node_counter += 1
        return node_id

    def _enum_target_node_id(self, enum_val: Enum) -> Optional[str]:
        cls_name = type(enum_val).__name__
        if not cls_name.endswith("Enum"):
            return None
        prefix = cls_name[:-4]
        if prefix not in _ENUM_NODE_TYPES:
            return None
        return f"{prefix}__{enum_val.value}"

    @staticmethod
    def _serialize_scalar(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return str(value)

    def _walk_pydantic(
        self,
        obj: Any,
        parent_id: Optional[str],
        edge_label: Optional[str],
        nodes_out: List[Tuple[str, Dict[str, Any]]],
        edges_out: List[Tuple[str, str, str]],
        hints_out: Dict[str, Dict[str, Any]],  # NEW
        visited: set,
    ) -> Optional[str]:
        # ── List — recurse on each item ──────────────────────────────────────
        if isinstance(obj, list):
            for item in obj:
                self._walk_pydantic(
                    item,
                    parent_id,
                    edge_label,
                    nodes_out,
                    edges_out,
                    hints_out,
                    visited,
                )
            return None

        # ── Non-model — nothing to emit ──────────────────────────────────────
        if not isinstance(obj, BaseModel):
            return None

        cls = type(obj)
        cls_name = cls.__name__
        node_id = self._resolve_node_id(obj, cls_name)

        if parent_id and edge_label:
            edges_out.append((parent_id, node_id, edge_label))

        if node_id in visited:
            return node_id
        visited.add(node_id)

        # ── Properties: scalars, non-node-type enums, flat scalar lists ──────
        props: Dict[str, Any] = {
            "name": node_id,
            "class": cls_name,
        }
        for field_name, field_info in cls.model_fields.items():
            val = getattr(obj, field_name, None)
            if val is None or isinstance(val, BaseModel):
                continue

            # Bare enum
            if isinstance(val, Enum):
                # Node-type enums are emitted as edges below — skip as property.
                if self._enum_target_node_id(val) is None:
                    s = self._serialize_scalar(val)
                    if s is not None:
                        props[field_name] = s
                continue

            # List
            if isinstance(val, list):
                if all(not isinstance(x, BaseModel) for x in val):
                    # Drop node-type enum members from the property string;
                    # they become edges. Keep plain scalars/enums.
                    filtered = [
                        x
                        for x in val
                        if not (
                            isinstance(x, Enum)
                            and self._enum_target_node_id(x) is not None
                        )
                    ]
                    if filtered:
                        serialized = [self._serialize_scalar(x) for x in filtered]
                        props[field_name] = ", ".join(s for s in serialized if s)
                continue

            # Plain scalar
            s = self._serialize_scalar(val)
            if s is not None:
                props[field_name] = s

        nodes_out.append(
            (node_id, {"label": "entity", "properties": props, "level": 2})
        )

        # ── Edges: nested models, and node-type enums become edges ───────────
        for field_name, field_info in cls.model_fields.items():
            val = getattr(obj, field_name, None)
            if val is None:
                continue

            inner_edge_label = field_name
            extra = field_info.json_schema_extra or {}
            if "edge_label" in extra:
                inner_edge_label = extra["edge_label"]

            # Nested model
            if isinstance(val, BaseModel):
                self._walk_pydantic(
                    val,
                    node_id,
                    inner_edge_label,
                    nodes_out,
                    edges_out,
                    hints_out,
                    visited,
                )

            # Bare enum that maps to a node type → edge + stub hint
            elif isinstance(val, Enum):
                target_id = self._enum_target_node_id(val)
                if target_id:
                    hints_out.setdefault(
                        target_id,
                        {
                            "label": "entity",
                            "properties": {
                                "name": target_id,
                                "class": type(val).__name__[:-4],
                                "value": val.value,
                            },
                            "level": 2,
                        },
                    )
                    edges_out.append((node_id, target_id, inner_edge_label))

            # List of models / enums
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, BaseModel):
                        self._walk_pydantic(
                            item,
                            node_id,
                            inner_edge_label,
                            nodes_out,
                            edges_out,
                            hints_out,
                            visited,
                        )
                    elif isinstance(item, Enum):
                        target_id = self._enum_target_node_id(item)
                        if target_id:
                            hints_out.setdefault(
                                target_id,
                                {
                                    "label": "entity",
                                    "properties": {
                                        "name": target_id,
                                        "class": type(item).__name__[:-4],
                                        "value": item.value,
                                    },
                                    "level": 2,
                                },
                            )
                            edges_out.append((node_id, target_id, inner_edge_label))

        return node_id

    # -------------------------------------------------------------------------
    # LLM response cache (debug)
    # -------------------------------------------------------------------------
    def _llm_response_cache_path(self) -> str:
        return f"output/chunks/{self.dataset_name}_responses.json"

    def _load_llm_response_cache(self):
        """
        Load cached LLM responses from a previous run.

        Detection rule: if `output/chunks/{dataset_name}.txt` exists,
        we assume a prior construction run happened, and any matching
        response cache file is restored.  Missing / corrupt cache is
        treated as a cold start (no error).
        """
        chunks_path = f"output/chunks/{self.dataset_name}.txt"
        if not os.path.exists(chunks_path):
            logger.info(
                f"[{self.dataset_name}] No prior chunks file — cold start, "
                f"LLM will be called for every chunk."
            )
            return

        cache_path = self._llm_response_cache_path()
        if not os.path.exists(cache_path):
            logger.info(
                f"[{self.dataset_name}] Chunks file present but no cached "
                f"responses at {cache_path}; LLM will be called."
            )
            return

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected cache format: {type(data).__name__}")
            self.llm_responses = {str(k): str(v) for k, v in data.items()}
            logger.info(
                f"[{self.dataset_name}] Loaded {len(self.llm_responses)} "
                f"cached LLM responses from {cache_path}."
            )
        except Exception as e:
            logger.warning(
                f"[{self.dataset_name}] Failed to load response cache "
                f"({type(e).__name__}: {e}); starting cold."
            )
            self.llm_responses = {}

    def _save_llm_response_cache(self):
        """Persist the response cache. Safe to call from multiple threads."""
        cache_path = self._llm_response_cache_path()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with self._response_cache_lock:
                snapshot = dict(self.llm_responses)

                tmp = cache_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                os.replace(tmp, cache_path)
        except Exception as e:
            logger.warning(
                f"[{self.dataset_name}] Failed to save response cache: "
                f"{type(e).__name__}: {e}"
            )

    def process_with_pydantic(self, chunk: str, id: str):
        prompt = self._get_construction_prompt(chunk)

        # ── Response cache: reuse LLM outputs across runs ────────────────
        if id in self.llm_responses:
            llm_response = self.llm_responses[id]
            logger.debug(f"[{id}] Using cached LLM response")
        else:
            llm_response = self.extract_with_llm(prompt)
            with self._response_cache_lock:
                self.llm_responses[id] = llm_response
            self._track_tokens(id, prompt, llm_response)
            # Persist eagerly so a crash mid-run preserves progress.
            self._save_llm_response_cache()

        self._track_tokens(id, prompt, llm_response)
        self.metrics["chunks_total"] += 1

        validated = self._validate_pydantic_response(llm_response)
        if validated is None:
            return
        self.metrics["chunks_validated"] += 1

        nodes_out: List[Tuple[str, Dict[str, Any]]] = []
        edges_out: List[Tuple[str, str, str]] = []
        hints_out: Dict[str, Dict[str, Any]] = {}
        visited: set = set()

        for field_name, _ in type(validated).model_fields.items():
            val = getattr(validated, field_name, None)
            if val is None:
                continue
            self._walk_pydantic(
                val,
                parent_id=None,
                edge_label=field_name,
                nodes_out=nodes_out,
                edges_out=edges_out,
                hints_out=hints_out,
                visited=visited,
            )

        with self.lock:
            self._merge_nodes_and_edges(nodes_out, edges_out, hints_out, chunk_id=id)

    # -------------------------------------------------------------------------
    # Graph merge
    # -------------------------------------------------------------------------

    def _merge_nodes_and_edges(
        self,
        nodes: List[Tuple[str, Dict[str, Any]]],
        edges: List[Tuple[str, str, str]],
        hints: Optional[Dict[str, Dict[str, Any]]] = None,
        chunk_id: Optional[str] = None,
    ):
        """
        Merge nodes, edges, and enum-target hints into the graph.

        Provenance rule (做法 B):
        * Node IDs stay canonical (`{Class}__{value}`) — no chunk namespacing.
        * Each node carries a `provenance` list accumulating every chunk_id
            that contributed to it.  Duplicates are suppressed.
        * Edges also accumulate provenance per (u, v, relation); dedup is
            still performed later by `triple_deduplicate`, but the list lets
            you trace which chunks produced a given relation.

        Hints are stub nodes for referenced-but-not-yet-declared canonical
        targets.  They are created ONLY when a node with the same ID is absent,
        so a real node with richer properties always wins.
        """
        # ── 1) Hints — only if the node doesn't already exist ─────────────────
        for node_id, hint_data in (hints or {}).items():
            if node_id not in self.graph:
                self.metrics["nodes_emitted"] += 1
                props = dict(hint_data["properties"])
                if chunk_id is not None:
                    props["provenance"] = [chunk_id]
                hint_data = {**hint_data, "properties": props}
                self.graph.add_node(node_id, **hint_data)
            elif chunk_id is not None:
                self._append_provenance(node_id, chunk_id)

        # ── 2) Real nodes ─────────────────────────────────────────────────────
        for node_id, node_data in nodes:
            self.metrics["nodes_emitted"] += 1

            if node_id in self.graph:
                existing = self.graph.nodes[node_id].get("properties", {})
                incoming = node_data["properties"]

                # Merge semantic props: incoming first, existing wins on conflicts
                merged = {**incoming, **existing}

                # Provenance: union of old and new
                prov = list(existing.get("provenance", []))
                if chunk_id is not None and chunk_id not in prov:
                    prov.append(chunk_id)
                if prov:
                    merged["provenance"] = prov

                self.graph.nodes[node_id]["properties"] = merged
                self.graph.nodes[node_id]["level"] = node_data.get("level", 2)
            else:
                props = dict(node_data["properties"])
                if chunk_id is not None:
                    props["provenance"] = [chunk_id]
                self.graph.add_node(node_id, **{**node_data, "properties": props})

        # ── 3) Edges ──────────────────────────────────────────────────────────
        # MultiDiGraph allows parallel edges; we tag each with the current
        # chunk_id so dedup can later merge identical (u, v, relation) triples
        # while keeping a full provenance list.
        for u, v, relation in edges:
            self.metrics["edges_emitted"] += 1
            self.graph.add_edge(
                u,
                v,
                relation=relation,
                provenance=[chunk_id] if chunk_id is not None else [],
            )

    def _append_provenance(self, node_id: str, chunk_id: str):
        """Append chunk_id to a node's provenance list, deduping."""
        props = self.graph.nodes[node_id].get("properties", {})
        prov = list(props.get("provenance", []))
        if chunk_id not in prov:
            prov.append(chunk_id)
            props["provenance"] = prov
            self.graph.nodes[node_id]["properties"] = props

    # -------------------------------------------------------------------------
    # Graph walk — legacy paths
    # -------------------------------------------------------------------------

    def _find_or_create_entity(
        self,
        entity_name: str,
        chunk_id: str,
        nodes_to_add: list,
        entity_type: Optional[str] = None,
    ) -> str:
        with self.lock:
            entity_node_id = next(
                (
                    n
                    for n, d in self.graph.nodes(data=True)
                    if d.get("label") == "entity"
                    and d["properties"]["name"] == entity_name
                ),
                None,
            )
            if not entity_node_id:
                entity_node_id = f"entity_{self.node_counter}"
                properties = {"name": entity_name, "chunk id": chunk_id}
                if entity_type:
                    properties["schema_type"] = entity_type
                nodes_to_add.append(
                    (
                        entity_node_id,
                        {"label": "entity", "properties": properties, "level": 2},
                    )
                )
                self.node_counter += 1
        return entity_node_id

    def _find_or_create_entity_direct(
        self,
        entity_name: str,
        chunk_id: str,
        entity_type: Optional[str] = None,
    ) -> str:
        entity_node_id = next(
            (
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("label") == "entity" and d["properties"]["name"] == entity_name
            ),
            None,
        )
        if not entity_node_id:
            entity_node_id = f"entity_{self.node_counter}"
            properties = {"name": entity_name, "chunk id": chunk_id}
            if entity_type:
                properties["schema_type"] = entity_type
            self.graph.add_node(
                entity_node_id, label="entity", properties=properties, level=2
            )
            self.node_counter += 1
        return entity_node_id

    @staticmethod
    def _validate_triple_format(triple: list) -> Optional[tuple]:
        try:
            if len(triple) > 3:
                triple = triple[:3]
            elif len(triple) < 3:
                return None
            return tuple(triple)
        except Exception:
            return None

    def _process_attributes(
        self, extracted_attr: dict, chunk_id: str, entity_types: Optional[dict] = None
    ) -> Tuple[list, list]:
        nodes_to_add, edges_to_add = [], []
        for entity, attributes in extracted_attr.items():
            for attr in attributes:
                attr_node_id = f"attr_{self.node_counter}"
                nodes_to_add.append(
                    (
                        attr_node_id,
                        {
                            "label": "attribute",
                            "properties": {"name": attr, "chunk id": chunk_id},
                            "level": 1,
                        },
                    )
                )
                self.node_counter += 1
                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self._find_or_create_entity(
                    entity, chunk_id, nodes_to_add, entity_type
                )
                edges_to_add.append((entity_node_id, attr_node_id, "has_attribute"))
        return nodes_to_add, edges_to_add

    def _process_triples(
        self,
        extracted_triples: list,
        chunk_id: str,
        entity_types: Optional[dict] = None,
    ) -> Tuple[list, list]:
        nodes_to_add, edges_to_add = [], []
        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue
            subj, pred, obj = validated_triple
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None
            subj_node_id = self._find_or_create_entity(
                subj, chunk_id, nodes_to_add, subj_type
            )
            obj_node_id = self._find_or_create_entity(
                obj, chunk_id, nodes_to_add, obj_type
            )
            edges_to_add.append((subj_node_id, obj_node_id, pred))
        return nodes_to_add, edges_to_add

    def process_level1_level2(self, chunk: str, id: str):
        prompt = self._get_construction_prompt(chunk)
        llm_response = self.extract_with_llm(prompt)
        self._track_tokens(id, prompt, llm_response)
        self.metrics["chunks_total"] += 1

        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)
        if not parsed_response:
            self.metrics["chunks_failed_json"] += 1
            return
        self.metrics["chunks_validated"] += 1

        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})

        attr_nodes, attr_edges = self._process_attributes(
            extracted_attr, id, entity_types
        )
        triple_nodes, triple_edges = self._process_triples(
            extracted_triples, id, entity_types
        )

        with self.lock:
            self._merge_nodes_and_edges(
                attr_nodes + triple_nodes, attr_edges + triple_edges, chunk_id=id
            )

    def _process_attributes_agent(
        self, extracted_attr: dict, chunk_id: str, entity_types: Optional[dict] = None
    ):
        for entity, attributes in extracted_attr.items():
            for attr in attributes:
                attr_node_id = f"attr_{self.node_counter}"
                self.graph.add_node(
                    attr_node_id,
                    label="attribute",
                    properties={"name": attr, "chunk id": chunk_id},
                    level=1,
                )
                self.node_counter += 1
                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self._find_or_create_entity_direct(
                    entity, chunk_id, entity_type
                )
                self.graph.add_edge(
                    entity_node_id, attr_node_id, relation="has_attribute"
                )

    def _process_triples_agent(
        self,
        extracted_triples: list,
        chunk_id: str,
        entity_types: Optional[dict] = None,
    ):
        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue
            subj, pred, obj = validated_triple
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None
            subj_node_id = self._find_or_create_entity_direct(subj, chunk_id, subj_type)
            obj_node_id = self._find_or_create_entity_direct(obj, chunk_id, obj_type)
            self.graph.add_edge(subj_node_id, obj_node_id, relation=pred)

    def process_level1_level2_agent(self, chunk: str, id: str):
        prompt = self._get_construction_prompt(chunk)
        llm_response = self.extract_with_llm(prompt)
        self._track_tokens(id, prompt, llm_response)
        self.metrics["chunks_total"] += 1

        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)
        if not parsed_response:
            self.metrics["chunks_failed_json"] += 1
            return
        self.metrics["chunks_validated"] += 1

        new_schema_types = parsed_response.get("new_schema_types", {})
        if new_schema_types:
            self._update_schema_with_new_types(new_schema_types)

        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})

        with self.lock:
            self._process_attributes_agent(extracted_attr, id, entity_types)
            self._process_triples_agent(extracted_triples, id, entity_types)

    def _update_schema_with_new_types(self, new_schema_types: Dict[str, List[str]]):
        try:
            schema_path = self.config.datasets[self.dataset_name].schema_path
            if not schema_path:
                return
            with open(schema_path, "r", encoding="utf-8") as f:
                current_schema = json.load(f)
            updated = False
            for key, json_key in (
                ("nodes", "Nodes"),
                ("relations", "Relations"),
                ("attributes", "Attributes"),
            ):
                if key in new_schema_types:
                    for new_item in new_schema_types[key]:
                        if new_item not in current_schema.get(json_key, []):
                            current_schema.setdefault(json_key, []).append(new_item)
                            updated = True
            if updated:
                with open(schema_path, "w", encoding="utf-8") as f:
                    json.dump(current_schema, f, ensure_ascii=False, indent=2)
                self.schema = current_schema
        except Exception as e:
            logger.error(
                f"Failed to update schema for '{self.dataset_name}': "
                f"{type(e).__name__}: {e}"
            )

    # -------------------------------------------------------------------------
    # Document orchestration (construction phase only)
    # -------------------------------------------------------------------------

    def process_document(self, doc: Dict[str, Any]) -> None:
        if not doc:
            raise ValueError("Document is empty or None")

        chunks, chunk2id = self.chunk_text(doc)
        if not chunks or not chunk2id:
            raise ValueError(
                f"No valid chunks generated. chunks={len(chunks)}, "
                f"chunk2id={len(chunk2id)}"
            )

        for chunk in chunks:
            chunk_id = self._stable_chunk_id(chunk)
            if self.mode == "pydantic":
                self.process_with_pydantic(chunk, chunk_id)
            elif self.mode == "agent":
                self.process_level1_level2_agent(chunk, chunk_id)
            else:
                self.process_level1_level2(chunk, chunk_id)

    def process_all_documents(self, documents: List[Dict[str, Any]]) -> None:
        max_workers = min(
            self.config.construction.max_workers, (os.cpu_count() or 1) + 4
        )
        start_construct = time.time()
        total_docs = len(documents)
        logger.info(
            f"Starting processing {total_docs} documents with {max_workers} workers..."
        )

        processed_count = 0
        failed_count = 0

        try:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                all_futures = [
                    executor.submit(self.process_document, doc) for doc in documents
                ]
                for future in futures.as_completed(all_futures):
                    try:
                        future.result()
                        processed_count += 1
                        if processed_count % 5 == 0 or processed_count == total_docs:
                            elapsed = time.time() - start_construct
                            avg = elapsed / processed_count if processed_count else 0
                            remaining = (total_docs - processed_count) * avg
                            logger.info(
                                f"Progress: {processed_count}/{total_docs} "
                                f"({processed_count / total_docs * 100:.1f}%) "
                                f"[{failed_count} failed] "
                                f"ETA: {remaining / 60:.1f} min"
                            )
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Document failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Executor error: {type(e).__name__}: {e}")
            return

        logger.info(f"Construction Time: {time.time() - start_construct:.1f}s")
        logger.info(f"Successfully processed: {processed_count}/{total_docs}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Metrics: {json.dumps(self.metrics)}")

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================

    def _checkpoint_path(self) -> str:
        return f"output/graphs/{self.dataset_name}_checkpoint.pkl"

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self._checkpoint_path())

    def save_checkpoint(self, path: Optional[str] = None):
        """
        Persist the constructed graph + metadata to disk.

        Called at the end of construct() — before Level 3/4 — so that a debug
        run can reload and skip the expensive LLM phase entirely.
        """
        path = path or self._checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "graph": self.graph,
            "all_chunks": self.all_chunks,
            "node_counter": self.node_counter,
            "metrics": self.metrics,
            "token_len": self.token_len,
            "token_len_per_chunk": self.token_len_per_chunk,
            "mode": self.mode,
            "dataset_name": self.dataset_name,
            "saved_at": time.time(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(
            f"Checkpoint saved: {path} "
            f"(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """
        Restore a previously saved checkpoint. Returns True on success.
        """
        path = path or self._checkpoint_path()
        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}")
            return False

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.graph = payload["graph"]
        self.all_chunks = payload["all_chunks"]
        self.node_counter = payload["node_counter"]
        self.metrics = payload["metrics"]
        self.token_len = payload["token_len"]
        self.token_len_per_chunk = payload.get("token_len_per_chunk", {})
        self._constructed = True

        logger.info(
            f"Checkpoint loaded: {path} "
            f"(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
        return True

    def _log_graph_stats(self, prefix: str = ""):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        class_counts: Dict[str, int] = {}
        level_counts: Dict[int, int] = {}
        provenance_lengths: List[int] = []

        for _, d in self.graph.nodes(data=True):
            cls = d.get("properties", {}).get("class") or d.get("label", "?")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            lvl = d.get("level", -1)
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

            prov = d.get("properties", {}).get("provenance")
            if prov:
                provenance_lengths.append(len(prov))

        logger.info(
            f"{prefix}Graph stats: nodes={n_nodes}, edges={n_edges}, "
            f"levels={level_counts}"
        )
        logger.info(
            f"{prefix}Top node classes: "
            f"{sorted(class_counts.items(), key=lambda x: -x[1])[:8]}"
        )

        degrees = [d for _, d in self.graph.degree()]
        if degrees:
            import statistics

            logger.info(
                f"{prefix}Degree: min={min(degrees)}, "
                f"median={statistics.median(degrees)}, "
                f"max={max(degrees)}, "
                f"isolated={sum(1 for d in degrees if d == 0)}"
            )

        if provenance_lengths:
            logger.info(
                f"{prefix}Provenance: nodes_with_prov={len(provenance_lengths)}, "
                f"max_chunks_per_node={max(provenance_lengths)}, "
                f"median={statistics.median(provenance_lengths):.1f}"
            )

    # =========================================================================
    # PUBLIC PHASES
    # =========================================================================

    def construct(self, corpus: str) -> None:
        """
        Phase 1 — LLM extraction. Expensive; safe to skip via load_checkpoint().
        Ends by saving a checkpoint to output/graphs/.
        """
        logger.info(f"========{'Start Construction':^20}========")
        logger.info("➖" * 30)
        start = time.time()

        with open(corpus, "r", encoding="utf-8") as f:
            documents = json_repair.load(f)

        self.process_all_documents(documents)
        self.save_chunks_to_file()
        self._constructed = True

        logger.info(
            f"Construction complete in {time.time() - start:.1f}s "
            f"(tokens={self.token_len})"
        )
        self._log_graph_stats(prefix="[post-construct] ")

        # Save checkpoint BEFORE Level 3/4 — this is the debug boundary.
        self.save_checkpoint()

    def postprocess(self) -> None:
        """
        Phase 2 — deduplicate triples and detect communities (Level 3/4).
        Reloadable, cheap-ish, and independently callable for debugging.
        """
        logger.info(f"========{'Start Postprocessing':^20}========")
        logger.info("➖" * 30)
        start = time.time()

        self.triple_deduplicate()
        logger.info(
            f"After dedup: nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}"
        )

        self.process_level4()
        self._postprocessed = True

        logger.info(f"Postprocessing complete in {time.time() - start:.1f}s")
        self._log_graph_stats(prefix="[post-level4] ")

    def build_knowledge_graph(
        self,
        corpus: str,
    ) -> List[Dict[str, Any]]:
        """
        Orchestrator.

        Args:
            corpus: Path to the corpus JSON.
        """
        logger.info(f"========{'Start Building':^20}========")
        logger.info("➖" * 30)

        """
            debug: If True and a checkpoint exists, load it and skip
                   the expensive construction phase.
            skip_postprocess: If True, stop after construction; useful for
                   inspecting the raw graph before Level 3/4.
        """
        debug: bool = getattr(self.config.system, "debug", False)
        skip_postprocess: bool = getattr(self.config.system, "skip_postprocess", False)

        # ── Phase 1: construct (or reload) ───────────────────────────────────
        if debug and self.checkpoint_exists():
            logger.info("Debug mode: checkpoint found — skipping construction.")
            if not self.load_checkpoint():
                logger.warning("Checkpoint load failed; falling back to construct().")
                self.construct(corpus)
        else:
            if debug:
                logger.info(
                    "Debug mode requested but no checkpoint found — "
                    "running full construction."
                )
            self.construct(corpus)

        # ── Optional early exit for inspection ───────────────────────────────
        if skip_postprocess:
            logger.info("skip_postprocess=True — returning pre-Level3/4 output.")
            return self.format_output()

        # ── Phase 2: postprocess ─────────────────────────────────────────────
        logger.info(f"🚀🚀🚀🚀 {'Processing Level 3 and 4':^20} 🚀🚀🚀🚀")
        logger.info("➖" * 20)
        self.postprocess()

        # ── Final output ─────────────────────────────────────────────────────
        output = self.format_output()
        json_output_path = f"output/graphs/{self.dataset_name}_new.json"
        os.makedirs("output/graphs", exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Final graph saved to {json_output_path}")
        return output

    # =========================================================================
    # Post-processing internals
    # =========================================================================

    def triple_deduplicate(self):
        """
        Collapse parallel (u, v, relation) edges into one, merging their
        `provenance` lists so no contribution history is lost.
        """
        new_graph = nx.MultiDiGraph()

        # Copy all nodes
        for node, node_data in self.graph.nodes(data=True):
            new_graph.add_node(node, **node_data)

        # Group edges by (u, v, relation); union their provenance
        grouped: Dict[Tuple[str, str, str], List[str]] = {}
        seen_attrs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            relation = data.get("relation")
            group_key = (u, v, relation)

            prov = list(data.get("provenance", []))
            if group_key in grouped:
                for cid in prov:
                    if cid and cid not in grouped[group_key]:
                        grouped[group_key].append(cid)
            else:
                grouped[group_key] = prov
                # Keep the first-seen edge's non-provenance, non-relation attributes
                seen_attrs[group_key] = {
                    k: val
                    for k, val in data.items()
                    if k not in ("provenance", "relation")
                }

        for (u, v, relation), prov in grouped.items():
            attrs = dict(seen_attrs.get((u, v, relation), {}))
            attrs["relation"] = relation
            attrs["provenance"] = prov
            new_graph.add_edge(u, v, **attrs)

        self.graph = new_graph

    def process_level4(self):
        level2_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("level") == 2
        ]
        if not level2_nodes:
            logger.warning("No level-2 nodes; skipping community detection.")
            return

        # Exclude isolated nodes — they cannot form meaningful communities.
        connected = [n for n in level2_nodes if self.graph.degree(n) > 0]
        isolated = [n for n in level2_nodes if self.graph.degree(n) == 0]
        if isolated:
            logger.warning(
                f"Excluded {len(isolated)} isolated level-2 nodes from community detection."
            )
        if not connected:
            logger.warning("No connected level-2 nodes; skipping community detection.")
            return

        start_comm = time.time()
        _tree_comm = tree_comm.FastTreeComm(
            self.graph,
            embedding_model=self.config.tree_comm.embedding_model,
            struct_weight=self.config.tree_comm.struct_weight,
        )
        comm_to_nodes = _tree_comm.detect_communities(level2_nodes)
        _tree_comm.create_super_nodes_with_keywords(comm_to_nodes, level=4)
        logger.info(f"Community Indexing Time: {time.time() - start_comm}s")

    # =========================================================================
    # Output helpers
    # =========================================================================

    def format_output(self) -> List[Dict[str, Any]]:
        output = []
        for u, v, data in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            output.append(
                {
                    "start_node": {
                        "label": u_data["label"],
                        "properties": u_data["properties"],
                    },
                    "relation": data["relation"],
                    "provenance": data.get("provenance", []),
                    "end_node": {
                        "label": v_data["label"],
                        "properties": v_data["properties"],
                    },
                }
            )
        return output

    def save_graphml(self, output_path: str):
        graph_processor.save_graph(self.graph, output_path)
