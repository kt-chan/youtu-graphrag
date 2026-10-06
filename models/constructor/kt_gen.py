"""kt_gen.py — Knowledge tree construction with ontology-aware Pydantic extraction.

Pipeline phases (independently callable for debugging):
  1. construct(corpus)              — LLM extraction, expensive. Saves a checkpoint.
  2. postprocess()                  — deduplicate() + detect_communities().
     ├── deduplicate()              — collapse parallel (u, v, relation) edges.
     └── detect_communities()       — Level-4 community detection (tree_comm).
  3. build_knowledge_graph(corpus)  — orchestrates 1 → 2 and writes JSON output.

Skipping Level 4 only:
  config.system.skip_postprocess = True
  → deduplicate() still runs, JSON output still written, only community
    detection is bypassed.

Debug workflow:
  # Run 1 (cold): construct + checkpoint + postprocess
  KTBuilder("debt_collection").build_knowledge_graph("corpus.json")

  # Run 2 (warm): reload checkpoint, skip construction (config.system.debug=True)
  KTBuilder("debt_collection").build_knowledge_graph("corpus.json")

Common infrastructure lives under utils/:
  * utils/checkpoint   — checkpoint save/load
  * utils/llm_cache    — disk-backed LLM response cache
  * utils/chunking     — TextChunker + stable_chunk_id

Only the `pydantic` extraction path is supported.  Legacy `agent` /
level-1/2 branches have been removed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent import futures
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import json_repair
import networkx as nx
import tiktoken
from pydantic import BaseModel, ValidationError

from config import ConfigManager, get_config
from schemas.debt_collection import DebtCollectionExtraction, GraphNodeEnum
from utils import call_llm_api, graph_processor, tree_comm
from utils.checkpoint import (
    checkpoint_exists as _ckpt_exists,
    checkpoint_path as _ckpt_path,
    load_checkpoint as _ckpt_load,
    save_checkpoint as _ckpt_save,
)
from utils.chunking import TextChunker, stable_chunk_id
from utils.llm_cache import LLMResponseCache
from utils.logger import logger


# =============================================================================
# Prompt building (pydantic-only)
# =============================================================================
@lru_cache(maxsize=8)
def _serialize_pydantic_schema(model: Type[BaseModel]) -> str:
    """序列化 Pydantic schema，并把 GraphNodeEnum 的中文说明注入
    `x-enum-descriptions`，使 DeepSeek / GLM 在 prompt 中直接看到每个
    枚举值的适用场景与决策建议。"""
    schema = model.model_json_schema()

    for defn in schema.get("$defs", {}).values():
        values = defn.get("enum")
        if not values:
            continue
        descs = {v: d for v in values if (d := GraphNodeEnum.description_for(v))}
        if descs:
            defn["x-enum-descriptions"] = descs

    return json.dumps(schema, ensure_ascii=False, indent=2)


def _build_construction_prompt(
    config: ConfigManager,
    dataset_name: str,
    pydantic_model: Type[BaseModel],
    chunk: str,
) -> str:
    construction_prompts = config.prompts["construction"]

    base_prompt_type = (
        dataset_name if dataset_name in construction_prompts else "general"
    )
    candidate = f"{base_prompt_type}_agent"
    prompt_type = (
        candidate if candidate in construction_prompts else "debt_collection_agent"
    )
    if prompt_type not in construction_prompts:
        raise KeyError(
            f"No pydantic construction prompt registered; expected "
            f"'{candidate}' or fallback 'debt_collection_agent_pydantic'."
        )

    schema_str = _serialize_pydantic_schema(pydantic_model)
    return config.get_prompt_formatted(
        "construction", prompt_type, schema=schema_str, chunk=chunk
    )


# =============================================================================
# GraphWalker — Pydantic model tree → nodes / edges / hints
# =============================================================================
class GraphWalker:
    """Walk a validated Pydantic tree and emit graph elements.

    Owns the anonymous node counter so any component that needs to allocate
    IDs shares a single source of truth.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.RLock()

    # ── ID factory ───────────────────────────────────────────────────────
    def next_counter(self) -> int:
        with self._lock:
            n = self._counter
            self._counter += 1
            return n

    @property
    def counter(self) -> int:
        return self._counter

    @counter.setter
    def counter(self, value: int) -> None:
        self._counter = value

    def resolve_node_id(self, obj: BaseModel, prefix: str) -> str:
        for field_name in type(obj).model_fields:
            if field_name.endswith("_id"):
                val = getattr(obj, field_name, None)
                if val is None:
                    continue
                if isinstance(val, Enum):
                    if val.value is None:
                        continue
                    val = val.value
                return f"{prefix}__{val}"
        return f"{prefix}__{self.next_counter()}"

    @staticmethod
    def enum_target_node_id(enum_val: Enum) -> Optional[str]:
        """Return the graph node ID for a node-producing enum member.

        Only enums subclassing `GraphNodeEnum` participate; all others stay
        scalar properties.  The prefix is declared by the enum itself via
        `node_prefix()`, so this function needs no registry.
        """
        if not isinstance(enum_val, GraphNodeEnum):
            return None
        return f"{enum_val.node_prefix()}__{enum_val.value}"

    @staticmethod
    def serialize_scalar(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Enum):
            return value.value
        return str(value)

    # ── Main walk ────────────────────────────────────────────────────────
    def walk(
        self,
        obj: Any,
        parent_id: Optional[str],
        edge_label: Optional[str],
        nodes_out: List[Tuple[str, Dict[str, Any]]],
        edges_out: List[Tuple[str, str, str]],
        hints_out: Dict[str, Dict[str, Any]],
        visited: Set[str],
    ) -> Optional[str]:
        # List — recurse on each item.
        if isinstance(obj, list):
            for item in obj:
                self.walk(
                    item,
                    parent_id,
                    edge_label,
                    nodes_out,
                    edges_out,
                    hints_out,
                    visited,
                )
            return None

        # Non-model — nothing to emit.
        if not isinstance(obj, BaseModel):
            return None

        cls = type(obj)
        cls_name = cls.__name__
        node_id = self.resolve_node_id(obj, cls_name)

        if parent_id and edge_label:
            edges_out.append((parent_id, node_id, edge_label))

        if node_id in visited:
            return node_id
        visited.add(node_id)

        props = self._collect_properties(obj, cls_name, node_id)
        nodes_out.append(
            (node_id, {"label": "entity", "properties": props, "level": 2})
        )

        self._collect_edges(obj, node_id, nodes_out, edges_out, hints_out, visited)
        return node_id

    # ── Property extraction ──────────────────────────────────────────────
    def _collect_properties(
        self, obj: BaseModel, cls_name: str, node_id: str
    ) -> Dict[str, Any]:
        props: Dict[str, Any] = {"name": node_id, "class": cls_name}
        for field_name in type(obj).model_fields:
            val = getattr(obj, field_name, None)
            if val is None or isinstance(val, BaseModel):
                continue

            # Bare enum — node-type enums become edges (handled in
            # _collect_edges); non-node enums remain scalar properties.
            if isinstance(val, Enum):
                if self.enum_target_node_id(val) is None:
                    s = self.serialize_scalar(val)
                    if s is not None:
                        props[field_name] = s
                continue

            # List of scalars / enums (lists containing models are handled
            # by _collect_edges).
            if isinstance(val, list):
                if all(not isinstance(x, BaseModel) for x in val):
                    filtered = [
                        x
                        for x in val
                        if not (
                            isinstance(x, Enum)
                            and self.enum_target_node_id(x) is not None
                        )
                    ]
                    if filtered:
                        serialized = [self.serialize_scalar(x) for x in filtered]
                        props[field_name] = ", ".join(s for s in serialized if s)
                continue

            # Plain scalar
            s = self.serialize_scalar(val)
            if s is not None:
                props[field_name] = s
        return props

    # ── Edge extraction ──────────────────────────────────────────────────
    def _collect_edges(
        self,
        obj: BaseModel,
        node_id: str,
        nodes_out: List[Tuple[str, Dict[str, Any]]],
        edges_out: List[Tuple[str, str, str]],
        hints_out: Dict[str, Dict[str, Any]],
        visited: Set[str],
    ) -> None:
        for field_name, field_info in type(obj).model_fields.items():
            val = getattr(obj, field_name, None)
            if val is None:
                continue

            inner_edge_label = field_name
            extra = field_info.json_schema_extra or {}
            if "edge_label" in extra:
                inner_edge_label = extra["edge_label"]

            if isinstance(val, BaseModel):
                self.walk(
                    val,
                    node_id,
                    inner_edge_label,
                    nodes_out,
                    edges_out,
                    hints_out,
                    visited,
                )
            elif isinstance(val, Enum):
                self._maybe_add_enum_edge(
                    val, node_id, inner_edge_label, edges_out, hints_out
                )
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, BaseModel):
                        self.walk(
                            item,
                            node_id,
                            inner_edge_label,
                            nodes_out,
                            edges_out,
                            hints_out,
                            visited,
                        )
                    elif isinstance(item, Enum):
                        self._maybe_add_enum_edge(
                            item, node_id, inner_edge_label, edges_out, hints_out
                        )

    def _maybe_add_enum_edge(
        self,
        enum_val: Enum,
        src_id: str,
        edge_label: str,
        edges_out: List[Tuple[str, str, str]],
        hints_out: Dict[str, Dict[str, Any]],
    ) -> None:
        if not isinstance(enum_val, GraphNodeEnum):
            return
        prefix = enum_val.node_prefix()
        target_id = f"{prefix}__{enum_val.value}"
        hints_out.setdefault(
            target_id,
            {
                "label": "entity",
                "properties": {
                    "name": target_id,
                    "class": prefix,
                    "value": enum_val.value,
                },
                "level": 2,
            },
        )
        edges_out.append((src_id, target_id, edge_label))


# =============================================================================
# Graph merge / dedup / output helpers
# =============================================================================
def _append_provenance(graph: nx.MultiDiGraph, node_id: str, chunk_id: str) -> None:
    """Append chunk_id to a node's provenance list, deduping."""
    props = graph.nodes[node_id].get("properties", {})
    prov = list(props.get("provenance", []))
    if chunk_id not in prov:
        prov.append(chunk_id)
        props["provenance"] = prov
        graph.nodes[node_id]["properties"] = props


def merge_nodes_and_edges(
    graph: nx.MultiDiGraph,
    nodes: List[Tuple[str, Dict[str, Any]]],
    edges: List[Tuple[str, str, str]],
    hints: Optional[Dict[str, Dict[str, Any]]] = None,
    chunk_id: Optional[str] = None,
    metrics: Optional[Dict[str, int]] = None,
) -> None:
    """Merge freshly-walked nodes, edges, and enum-target hints into `graph`.

    Provenance rule:
      * Node IDs stay canonical (`{Class}__{value}`) — no chunk namespacing.
      * Each node carries a `provenance` list accumulating every chunk_id
        that contributed to it.  Duplicates are suppressed.
      * Edges also accumulate provenance per (u, v, relation); dedup is
        performed later by `triple_deduplicate`.

    Hints are stub nodes for referenced-but-not-yet-declared canonical
    targets.  They are created ONLY when a node with the same ID is absent,
    so a real node with richer properties always wins.
    """
    if metrics is None:
        metrics = {}

    # ── 1) Hints — only if the node doesn't already exist ────────────────
    for node_id, hint_data in (hints or {}).items():
        if node_id not in graph:
            metrics["nodes_emitted"] = metrics.get("nodes_emitted", 0) + 1
            props = dict(hint_data["properties"])
            if chunk_id is not None:
                props["provenance"] = [chunk_id]
            graph.add_node(node_id, **{**hint_data, "properties": props})
        elif chunk_id is not None:
            _append_provenance(graph, node_id, chunk_id)

    # ── 2) Real nodes ────────────────────────────────────────────────────
    for node_id, node_data in nodes:
        metrics["nodes_emitted"] = metrics.get("nodes_emitted", 0) + 1

        if node_id in graph:
            existing = graph.nodes[node_id].get("properties", {})
            incoming = node_data["properties"]

            # Incoming (real, richer) wins on conflicts; existing supplies
            # any properties the incoming walk didn't carry.
            merged = {**existing, **incoming}

            prov = list(existing.get("provenance", []))
            if chunk_id is not None and chunk_id not in prov:
                prov.append(chunk_id)
            if prov:
                merged["provenance"] = prov

            graph.nodes[node_id]["properties"] = merged
            graph.nodes[node_id]["level"] = node_data.get("level", 2)
        else:
            props = dict(node_data["properties"])
            if chunk_id is not None:
                props["provenance"] = [chunk_id]
            graph.add_node(node_id, **{**node_data, "properties": props})

    # ── 3) Edges ─────────────────────────────────────────────────────────
    # MultiDiGraph allows parallel edges; each carries the current chunk_id
    # so dedup can later merge identical (u, v, relation) triples while
    # keeping a full provenance list.
    for u, v, relation in edges:
        metrics["edges_emitted"] = metrics.get("edges_emitted", 0) + 1
        graph.add_edge(
            u,
            v,
            relation=relation,
            provenance=[chunk_id] if chunk_id is not None else [],
        )


def triple_deduplicate(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Collapse parallel (u, v, relation) edges into one, merging provenance."""
    new_graph = nx.MultiDiGraph()

    for node, node_data in graph.nodes(data=True):
        new_graph.add_node(node, **node_data)

    grouped: Dict[Tuple[str, str, str], List[str]] = {}
    seen_attrs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for u, v, _key, data in graph.edges(keys=True, data=True):
        relation = data.get("relation")
        group_key = (u, v, relation)

        prov = list(data.get("provenance", []))
        if group_key in grouped:
            for cid in prov:
                if cid and cid not in grouped[group_key]:
                    grouped[group_key].append(cid)
        else:
            grouped[group_key] = prov
            # Keep the first-seen edge's non-provenance, non-relation attrs.
            seen_attrs[group_key] = {
                k: val for k, val in data.items() if k not in ("provenance", "relation")
            }

    for (u, v, relation), prov in grouped.items():
        attrs = dict(seen_attrs.get((u, v, relation), {}))
        attrs["relation"] = relation
        attrs["provenance"] = prov
        new_graph.add_edge(u, v, **attrs)

    return new_graph


def format_output(graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
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


def log_graph_stats(graph: nx.MultiDiGraph, prefix: str = "") -> None:
    import statistics

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    class_counts: Dict[str, int] = {}
    level_counts: Dict[int, int] = {}
    provenance_lengths: List[int] = []

    for _, d in graph.nodes(data=True):
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

    degrees = [d for _, d in graph.degree()]
    if degrees:
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


def _detect_communities(graph: nx.MultiDiGraph, config) -> None:
    """Level-4 community detection via tree_comm.FastTreeComm."""
    level2_nodes = [n for n, d in graph.nodes(data=True) if d.get("level") == 2]
    if not level2_nodes:
        logger.warning("No level-2 nodes; skipping community detection.")
        return

    # Exclude isolated nodes — they cannot form meaningful communities.
    connected = [n for n in level2_nodes if graph.degree(n) > 0]
    isolated = [n for n in level2_nodes if graph.degree(n) == 0]
    if isolated:
        logger.warning(
            f"Excluded {len(isolated)} isolated level-2 nodes from community detection."
        )
    if not connected:
        logger.warning("No connected level-2 nodes; skipping community detection.")
        return

    start = time.time()
    comm = tree_comm.FastTreeComm(
        graph,
        embedding_model=config.tree_comm.embedding_model,
        struct_weight=config.tree_comm.struct_weight,
    )
    comm_to_nodes = comm.detect_communities(level2_nodes)
    comm.create_super_nodes_with_keywords(comm_to_nodes, level=4)
    logger.info(f"Community Indexing Time: {time.time() - start}s")


# =============================================================================
# KTBuilder
# =============================================================================
class KTBuilder:
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
        self.lock = threading.RLock()

        self.datasets_no_chunk = config.construction.datasets_no_chunk
        self.token_len = 0
        self.token_len_per_chunk: Dict[str, int] = {}

        self.llm_client = call_llm_api.LLMCompletionCall()
        self.all_chunks: Dict[str, str] = {}

        self.mode = mode or config.construction.mode
        self.pydantic_model: Type[BaseModel] = (
            pydantic_model if pydantic_model is not None else DebtCollectionExtraction
        )

        # ── Composed helpers ─────────────────────────────────────────────
        self.walker = GraphWalker()
        self.chunker = TextChunker(
            dataset_name=dataset_name,
            datasets_no_chunk=self.datasets_no_chunk,
            chunk_size=getattr(config.construction, "chunk_size", 1000),
            overlap=getattr(config.construction, "overlap", 200),
            min_tail_tokens=getattr(config.construction, "min_tail_tokens", 100),
        )
        self.llm_cache = LLMResponseCache(dataset_name)
        self.llm_cache.load()

        self.metrics: Dict[str, int] = {
            "chunks_total": 0,
            "chunks_validated": 0,
            "chunks_failed_validation": 0,
            "chunks_failed_json": 0,
            "nodes_emitted": 0,
            "edges_emitted": 0,
        }

        # Debug/observability flags
        self._constructed = False
        self._postprocessed = False

    # -------------------------------------------------------------------------
    # Schema loading
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

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------
    def chunk_text(self, text) -> Tuple[List[str], Dict[str, str]]:
        chunks, chunk2id = self.chunker.chunk(text)
        with self.lock:
            self.all_chunks.update(chunk2id)
        return chunks, chunk2id

    # -------------------------------------------------------------------------
    # Token accounting
    # -------------------------------------------------------------------------
    def token_cal(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _track_tokens(
        self, chunk_id: str, prompt: str, response: Optional[str]
    ) -> None:
        cost = self.token_cal(prompt + (response or ""))
        with self.lock:
            self.token_len += cost
            self.token_len_per_chunk[chunk_id] = (
                self.token_len_per_chunk.get(chunk_id, 0) + cost
            )

    # -------------------------------------------------------------------------
    # LLM interaction
    # -------------------------------------------------------------------------
    def extract_with_llm(self, prompt: str) -> str:
        response = self.llm_client.call_api(prompt)
        parsed_dict = json_repair.loads(response)
        return json.dumps(parsed_dict, ensure_ascii=False)

    def _get_construction_prompt(self, chunk: str) -> str:
        return _build_construction_prompt(
            config=self.config,
            dataset_name=self.dataset_name,
            pydantic_model=self.pydantic_model,
            chunk=chunk,
        )

    # -------------------------------------------------------------------------
    # Pydantic validation
    # -------------------------------------------------------------------------
    def _validate_pydantic_response(
        self, raw_response: Optional[str]
    ) -> Optional[BaseModel]:
        if raw_response is None:
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
    # Chunk persistence
    # -------------------------------------------------------------------------
    def save_chunks_to_file(self) -> None:
        os.makedirs("output/chunks", exist_ok=True)
        chunk_file = f"output/chunks/{self.dataset_name}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in self.all_chunks.items():
                escaped = chunk_text.replace("\n", "\\n").replace("\t", "\\t")
                f.write(f"id: {chunk_id}\tChunk: {escaped}\n")
        logger.info(f"Chunk data saved to {chunk_file} ({len(self.all_chunks)} chunks)")

    # -------------------------------------------------------------------------
    # Extraction — pydantic-only
    # -------------------------------------------------------------------------
    def process_with_pydantic(self, chunk: str, id: str) -> None:
        prompt = self._get_construction_prompt(chunk)

        # Response cache: reuse LLM outputs across runs.
        if id in self.llm_cache:
            llm_response = self.llm_cache.get(id)
            logger.debug(f"[{id}] Using cached LLM response")
        else:
            llm_response = self.extract_with_llm(prompt)
            self.llm_cache.set(id, llm_response)
            # Persist eagerly so a crash mid-run preserves progress.
            self.llm_cache.save()

        self._track_tokens(id, prompt, llm_response)
        self.metrics["chunks_total"] += 1

        validated = self._validate_pydantic_response(llm_response)
        if validated is None:
            return
        self.metrics["chunks_validated"] += 1

        nodes_out: List[Tuple[str, Dict[str, Any]]] = []
        edges_out: List[Tuple[str, str, str]] = []
        hints_out: Dict[str, Dict[str, Any]] = {}
        visited: Set[str] = set()

        for field_name in type(validated).model_fields:
            val = getattr(validated, field_name, None)
            if val is None:
                continue
            self.walker.walk(
                val,
                parent_id=None,
                edge_label=field_name,
                nodes_out=nodes_out,
                edges_out=edges_out,
                hints_out=hints_out,
                visited=visited,
            )

        with self.lock:
            merge_nodes_and_edges(
                self.graph,
                nodes_out,
                edges_out,
                hints_out,
                chunk_id=id,
                metrics=self.metrics,
            )

    # -------------------------------------------------------------------------
    # Document orchestration
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
            chunk_id = stable_chunk_id(chunk)
            self.process_with_pydantic(chunk, chunk_id)

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
                        logger.warning(f"artifact_id: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Executor error: {type(e).__name__}: {e}")
            return

        logger.info(f"Construction Time: {time.time() - start_construct:.1f}s")
        logger.info(f"Successfully processed: {processed_count}/{total_docs}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Metrics: {json.dumps(self.metrics)}")

    # =========================================================================
    # Checkpointing
    # =========================================================================
    def checkpoint_exists(self) -> bool:
        return _ckpt_exists(self.dataset_name)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        _ckpt_save(
            dataset_name=self.dataset_name,
            graph=self.graph,
            all_chunks=self.all_chunks,
            node_counter=self.walker.counter,
            metrics=self.metrics,
            token_len=self.token_len,
            token_len_per_chunk=self.token_len_per_chunk,
            mode=self.mode,
            path=path,
        )

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        payload = _ckpt_load(self.dataset_name, path)
        if payload is None:
            return False

        self.graph = payload["graph"]
        self.all_chunks = payload["all_chunks"]
        self.walker.counter = payload["node_counter"]
        self.metrics = payload["metrics"]
        self.token_len = payload["token_len"]
        self.token_len_per_chunk = payload.get("token_len_per_chunk", {})
        self._constructed = True

        logger.info(
            f"Checkpoint loaded: {_ckpt_path(self.dataset_name)} "
            f"(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
        return True

    # =========================================================================
    # Post-processing — Level 3 (dedup) and Level 4 (communities)
    # =========================================================================
    def deduplicate(self) -> None:
        """Level-3: collapse parallel (u, v, relation) edges into one.

        Merges provenance lists so no contribution history is lost.  Cheap,
        idempotent, and safe to run even on an already-deduplicated graph.
        """
        before_nodes = self.graph.number_of_nodes()
        before_edges = self.graph.number_of_edges()

        self.graph = triple_deduplicate(self.graph)

        logger.info(
            f"After dedup: nodes={before_nodes}→{self.graph.number_of_nodes()}, "
            f"edges={before_edges}→{self.graph.number_of_edges()}"
        )

    def detect_communities(self) -> None:
        """Level-4: cluster level-2 entity nodes and add community super-nodes."""
        _detect_communities(self.graph, self.config)

    def postprocess(self) -> None:
        """Full post-processing: dedup + community detection.

        Use `deduplicate()` / `detect_communities()` individually if you
        need finer control (e.g. skip only Level 4).
        """
        logger.info(f"========{'Start Postprocessing':^20}========")
        logger.info("➖" * 30)
        start = time.time()

        self.deduplicate()
        self.detect_communities()

        self._postprocessed = True
        logger.info(f"Postprocessing complete in {time.time() - start:.1f}s")
        log_graph_stats(self.graph, prefix="[post-level4] ")

    # =========================================================================
    # Public phases
    # =========================================================================
    def construct(self, corpus: str) -> None:
        """Phase 1 — LLM extraction. Expensive; skip via load_checkpoint()."""
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
        log_graph_stats(self.graph, prefix="[post-construct] ")

        # Save checkpoint BEFORE Level 3/4 — this is the debug boundary.
        self.save_checkpoint()

    def build_knowledge_graph(self, corpus: str) -> List[Dict[str, Any]]:
        """Orchestrator: construct (or reload) → dedup → [Level-4] → write JSON.

        `config.system.skip_postprocess` skips ONLY community detection
        (Level 4).  Deduplication and JSON output always run, so the
        returned graph is always clean and persisted.
        """
        logger.info(f"========{'Start Building':^20}========")
        logger.info("➖" * 30)

        debug: bool = getattr(self.config.system, "debug", False)
        skip_communities: bool = getattr(self.config.system, "skip_postprocess", False)

        # ── Phase 1: construct (or reload) ───────────────────────────────
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

        # ── Phase 2: dedup (always) ──────────────────────────────────────
        logger.info(f"========{'Start Level 3 (dedup)':^20}========")
        logger.info("➖" * 30)
        start_pp = time.time()
        self.deduplicate()

        # ── Phase 3: community detection (skippable) ─────────────────────
        if skip_communities:
            logger.info("skip_postprocess=True — skipping Level 4 community detection.")
        else:
            logger.info(f"🚀🚀🚀🚀 {'Processing Level 4':^20} 🚀🚀🚀🚀")
            logger.info("➖" * 20)
            self.detect_communities()
            self._postprocessed = True

        logger.info(f"Postprocess stage complete in {time.time() - start_pp:.1f}s")
        log_graph_stats(
            self.graph,
            prefix="[post-level4] " if not skip_communities else "[post-dedup] ",
        )

        # ── Final output (always written) ────────────────────────────────
        output = format_output(self.graph)
        json_output_path = f"output/graphs/{self.dataset_name}_new.json"
        os.makedirs("output/graphs", exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Final graph saved to {json_output_path}")
        return output

    # =========================================================================
    # Output helpers
    # =========================================================================
    def format_output(self) -> List[Dict[str, Any]]:
        return format_output(self.graph)

    def save_graphml(self, output_path: str) -> None:
        graph_processor.save_graph(self.graph, output_path)
