# backend.py
import ast
import asyncio
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger
from utils.encoding import decode_bytes_with_detection
from utils.schema_utils import (
    ensure_demo_schema_exists,
    get_schema_path_for_dataset,
)
from utils.cache_utils import clear_cache_files

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from utils.document_parser import get_parser

    DOCUMENT_PARSER_AVAILABLE = True
except ImportError as e:
    DOCUMENT_PARSER_AVAILABLE = False
    logger.warning(f"Document parser not available: {e}")

try:
    from models.constructor import kt_gen as constructor
    from models.retriever import (
        agentic_decomposer as decomposer,
        enhanced_kt_retriever as retriever,
    )
    from config import get_config, ConfigManager

    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG components loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    logger.error(f"⚠️  GraphRAG components not available: {e}")


# ---------------------------------------------------------------------------
# App + middleware + static mounts
# ---------------------------------------------------------------------------
app = FastAPI(title="Youtu-GraphRAG Unified Interface", version="1.0.0")

if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
if os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared runtime state
# ---------------------------------------------------------------------------
config = None  # lazily initialized base config


def get_shared_config():
    """Lazily load and cache the base config (shared across requests)."""
    global config
    if config is None:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="GraphRAG components not available. Please install or configure them.",
            )
        config = get_config("config/base_config.yaml")
    return config


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_message(self, message: dict, client_id: str):
        ws = self.active_connections.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            self.disconnect(client_id)


manager = ConnectionManager()


async def send_event(client_id: str, message: dict) -> None:
    """Send a raw message dict over WS. Swallows errors."""
    await manager.send_message(message, client_id)


async def send_progress_update(
    client_id: str, stage: str, progress: int, message: str
) -> None:
    await send_event(
        client_id,
        {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None
    files_count: Optional[int] = None


class GraphConstructionRequest(BaseModel):
    dataset_name: str


class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None


class QuestionRequest(BaseModel):
    question: str
    dataset_name: str


class QuestionResponse(BaseModel):
    answer: str
    sub_questions: List[Dict]
    retrieved_triples: List[str]
    retrieved_chunks: List[str]
    reasoning_steps: List[Dict]
    visualization_data: Dict


# ===========================================================================
# VISUALIZATION HELPERS
# ===========================================================================
async def prepare_graph_visualization(graph_path: str) -> Dict:
    """Load a graph JSON file and convert it for the frontend."""
    try:
        if not os.path.exists(graph_path):
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        if isinstance(graph_data, list):
            return convert_graphrag_format(graph_data)
        if isinstance(graph_data, dict) and "nodes" in graph_data:
            return convert_standard_format(graph_data)
        return {"nodes": [], "links": [], "categories": [], "stats": {}}
    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}


def convert_graphrag_format(graph_data: List) -> Dict:
    """Convert GraphRAG relationship list to ECharts format."""
    nodes_dict: Dict[str, Dict] = {}
    links: List[Dict] = []

    for item in graph_data:
        if not isinstance(item, dict):
            continue

        start_node = item.get("start_node", {}) or {}
        end_node = item.get("end_node", {}) or {}
        relation = item.get("relation", "related_to")

        start_id = end_id = ""
        if start_node:
            props = start_node.get("properties", {}) or {}
            start_id = props.get("name", "")
            if start_id and start_id not in nodes_dict:
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": start_id[:200],
                    "category": props.get(
                        "schema_type", start_node.get("label", "entity")
                    ),
                    "symbolSize": 25,
                    "properties": props,
                }
        if end_node:
            props = end_node.get("properties", {}) or {}
            end_id = props.get("name", "")
            if end_id and end_id not in nodes_dict:
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": end_id[:200],
                    "category": props.get(
                        "schema_type", end_node.get("label", "entity")
                    ),
                    "symbolSize": 25,
                    "properties": props,
                }
        if start_id and end_id:
            links.append(
                {"source": start_id, "target": end_id, "name": relation, "value": 1}
            )

    categories_set = {n["category"] for n in nodes_dict.values()}
    categories = [
        {
            "name": cat,
            "itemStyle": {
                "color": f"hsl({i * 360 / max(len(categories_set), 1)}, 70%, 60%)"
            },
        }
        for i, cat in enumerate(categories_set)
    ]
    nodes = list(nodes_dict.values())
    return {
        "nodes": nodes[:500],
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000]),
        },
    }


def convert_standard_format(graph_data: Dict) -> Dict:
    """Convert standard {nodes: [], edges: []} to ECharts format."""
    node_types = {n.get("type", "entity") for n in graph_data.get("nodes", [])}
    categories = [
        {
            "name": t,
            "itemStyle": {
                "color": f"hsl({i * 360 / max(len(node_types), 1)}, 70%, 60%)"
            },
        }
        for i, t in enumerate(node_types)
    ]

    nodes = []
    for node in graph_data.get("nodes", []):
        attrs = node.get("attributes", [])
        nodes.append(
            {
                "id": node.get("id", ""),
                "name": node.get("name", node.get("id", ""))[:200],
                "category": node.get("type", "entity"),
                "value": len(attrs),
                "symbolSize": min(max(len(attrs) * 3 + 15, 15), 40),
                "attributes": attrs,
            }
        )

    links = [
        {
            "source": e.get("source", ""),
            "target": e.get("target", ""),
            "name": e.get("relation", "related_to"),
            "value": e.get("weight", 1),
        }
        for e in graph_data.get("edges", [])
    ]

    return {
        "nodes": nodes[:500],
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000]),
        },
    }


def prepare_subquery_visualization(
    sub_questions: List[Dict], reasoning_steps: List[Dict]
) -> Dict:
    nodes = [
        {
            "id": "original",
            "name": "Original Question",
            "category": "question",
            "symbolSize": 40,
        }
    ]
    links: List[Dict] = []
    for i, sub_q in enumerate(sub_questions):
        sub_id = f"sub_{i}"
        nodes.append(
            {
                "id": sub_id,
                "name": sub_q.get("sub-question", "")[:200] + "...",
                "category": "sub_question",
                "symbolSize": 30,
            }
        )
        links.append({"source": "original", "target": sub_id, "name": "decomposed to"})
    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}},
        ],
    }


def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    nodes: List[Dict] = []
    links: List[Dict] = []
    seen: set = set()

    for triple in triples[:10]:
        if not (
            isinstance(triple, str) and triple.startswith("[") and triple.endswith("]")
        ):
            continue
        try:
            parts = ast.literal_eval(triple)
        except Exception:
            continue
        if len(parts) != 3:
            continue
        source, relation, target = parts
        for entity in (source, target):
            if entity not in seen:
                seen.add(entity)
                nodes.append(
                    {
                        "id": str(entity),
                        "name": str(entity)[:200],
                        "category": "entity",
                        "symbolSize": 20,
                    }
                )
        links.append(
            {"source": str(source), "target": str(target), "name": str(relation)}
        )

    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": "entity", "itemStyle": {"color": "#95de64"}}],
    }


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    steps_data = [
        {
            "step": i + 1,
            "type": s.get("type", "unknown"),
            "question": s.get("question", "")[:50],
            "triples_count": s.get("triples_count", 0),
            "chunks_count": s.get("chunks_count", 0),
            "processing_time": s.get("processing_time", 0),
        }
        for i, s in enumerate(reasoning_steps)
    ]
    return {"steps": steps_data, "timeline": [s["processing_time"] for s in steps_data]}


# ===========================================================================
# GRAPH CONSTRUCTION — shared pipeline used by construct + reconstruct
# ===========================================================================
async def run_graph_construction(
    dataset_name: str,
    client_id: str,
    *,
    stage: str = "construction",
    demo_fallback: bool = True,
    include_visualization: bool = False,
) -> Dict:
    """Shared pipeline for building a knowledge graph.

    Parameters
    ----------
    stage                 : label used in progress/complete events.
    clear_caches          : whether to clear cache. When True and
                            `force_clear_caches` is False, the base-config
                            `system.debug` flag is honoured.
    force_clear_caches    : always clear caches even in debug mode.
    delete_existing_graph : remove `output/graphs/{dataset}_new.json` first.
    demo_fallback         : fall back to the demo corpus if the dataset's own
                            corpus is not found.
    include_visualization : return the prepared ECharts payload.
    """
    if not GRAPHRAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG components not available. Please install or configure them.",
        )

    cfg = get_shared_config()

    if not cfg.system.debug:
        await send_progress_update(client_id, stage, 5, "Cleaning old cache files...")
        await clear_cache_files(dataset_name)

    # Resolve corpus path
    corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
    if not os.path.exists(corpus_path) and demo_fallback:
        demo_corpus = "data/demo/demo_corpus.json"
        if os.path.exists(demo_corpus):
            corpus_path = demo_corpus
    if not os.path.exists(corpus_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    schema_path = get_schema_path_for_dataset(dataset_name)

    await send_progress_update(
        client_id, stage, 10, "Loading configuration and corpus..."
    )

    builder = constructor.KTBuilder(
        dataset_name, schema_path, mode=cfg.construction.mode, config=cfg
    )

    await send_progress_update(
        client_id, stage, 20, "Starting entity-relation extraction..."
    )

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: builder.build_knowledge_graph(corpus_path))

    graph_vis_data: Dict = {}
    if include_visualization:
        await send_progress_update(
            client_id, stage, 95, "Preparing visualization data..."
        )
        graph_vis_data = await prepare_graph_visualization(
            f"output/graphs/{dataset_name}_new.json"
        )

    await send_progress_update(client_id, stage, 100, "Graph construction completed!")
    await send_event(
        client_id,
        {
            "type": "complete",
            "stage": stage,
            "message": "Graph construction completed!",
            "timestamp": datetime.now().isoformat(),
        },
    )

    return {"graph_vis_data": graph_vis_data}


async def emit_graph_error(client_id: str, stage: str, message: str) -> None:
    await send_event(
        client_id,
        {
            "type": "error",
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        },
    )


# ===========================================================================
# UPLOAD
# ===========================================================================
ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".pdf", ".docx", ".doc"}
PLAIN_TEXT_EXT = {".txt", ".md"}
DOC_PARSER_EXT = {".pdf", ".docx", ".doc"}


def _derive_dataset_name(files: List[UploadFile]) -> str:
    if len(files) == 1:
        original = os.path.splitext(files[0].filename or "dataset")[0]
        cleaned = "".join(
            c for c in original if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        return cleaned.replace(" ", "_") or "dataset"
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{len(files)}files_{date_str}"


def _unique_dataset_name(base: str) -> str:
    name = base
    counter = 1
    while os.path.exists(f"data/uploaded/{name}"):
        name = f"{base}_{counter}"
        counter += 1
    return name


async def process_uploaded_files(files: List[UploadFile], client_id: str) -> tuple:
    """Persist uploads, extract text, write corpus.json.

    Returns (dataset_name, processed_count, skipped_files).
    """
    dataset_name = _unique_dataset_name(_derive_dataset_name(files))
    upload_dir = f"data/uploaded/{dataset_name}"
    os.makedirs(upload_dir, exist_ok=True)

    await send_progress_update(client_id, "upload", 10, "Starting file upload...")

    corpus_data: List[dict] = []
    skipped_files: List[str] = []
    processed_count = 0
    doc_parser = get_parser() if DOCUMENT_PARSER_AVAILABLE else None

    for i, file in enumerate(files):
        file_path = os.path.join(upload_dir, file.filename)
        content_bytes = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content_bytes)

        filename_lower = (file.filename or "").lower()
        ext = os.path.splitext(filename_lower)[1]
        progress = 10 + (i + 1) * 80 // len(files)

        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file type: {file.filename}")
            skipped_files.append(file.filename)
            await send_progress_update(
                client_id,
                "upload",
                progress,
                f"Skipped unsupported file: {file.filename}",
            )
            continue

        try:
            if ext in DOC_PARSER_EXT:
                if not doc_parser:
                    logger.warning(
                        f"Document parser not available, skipping {file.filename}"
                    )
                    skipped_files.append(file.filename)
                    await send_progress_update(
                        client_id,
                        "upload",
                        progress,
                        f"Skipped {file.filename} (parser unavailable)",
                    )
                    continue
                text = doc_parser.parse_file(file_path, ext)
                if text and text.strip():
                    corpus_data.append({"title": file.filename, "text": text})
                    processed_count += 1
                    await send_progress_update(
                        client_id, "upload", progress, f"Parsed {file.filename}"
                    )
                else:
                    logger.warning(f"No text extracted from {file.filename}")
                    skipped_files.append(file.filename)
                    await send_progress_update(
                        client_id, "upload", progress, f"No text in {file.filename}"
                    )
                continue

            if ext in PLAIN_TEXT_EXT:
                corpus_data.append(
                    {
                        "title": file.filename,
                        "text": decode_bytes_with_detection(content_bytes),
                    }
                )
                processed_count += 1
            elif ext == ".json":
                try:
                    data_obj = json.loads(decode_bytes_with_detection(content_bytes))
                    if isinstance(data_obj, list):
                        corpus_data.extend(data_obj)
                    else:
                        corpus_data.append(data_obj)
                    processed_count += 1
                except Exception:
                    corpus_data.append(
                        {
                            "title": file.filename,
                            "text": decode_bytes_with_detection(content_bytes),
                        }
                    )
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            skipped_files.append(file.filename)
            await send_progress_update(
                client_id, "upload", progress, f"Failed to process {file.filename}"
            )
            continue

        await send_progress_update(
            client_id, "upload", progress, f"Processed {file.filename}"
        )

    if processed_count == 0:
        msg = (
            "No supported files were uploaded. "
            "Allowed: .txt, .md, .json, .pdf, .docx, .doc"
        )
        if skipped_files:
            msg += f"; skipped: {', '.join(skipped_files)}"
        await send_progress_update(client_id, "upload", 0, msg)
        raise HTTPException(status_code=400, detail=msg)

    with open(f"{upload_dir}/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    # Ensure default demo schema exists
    ensure_demo_schema_exists()

    await send_progress_update(
        client_id, "upload", 100, "Upload completed successfully!"
    )
    return dataset_name, processed_count, skipped_files


# ===========================================================================
# QUESTION ANSWERING
# ===========================================================================
FINAL_MARKER_RE = re.compile(r"(?m)^[ \t]*(?:[#*_]+[ \t]*)*FINAL_ANSWER\b[^\n]*$")
NEW_QUERY_MARKER_RE = re.compile(r"(?m)^[ \t]*(?:[#*_]+[ \t]*)*NEW_QUERIES\b[^\n]*$")


def _dedup(items):
    return list({x: None for x in items}.keys())


def _merge_chunk_contents(ids, mapping) -> List[str]:
    return [
        f"[Chunk {idx}] {mapping.get(i, f'[Missing content for chunk {i}]')}"
        for idx, i in enumerate(ids, 1)
    ]


def _extract_final_answer(text: str) -> Optional[str]:
    matches = list(FINAL_MARKER_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    start = m.end()
    later_new = NEW_QUERY_MARKER_RE.search(text, pos=start)
    end = later_new.start() if later_new is not None else len(text)
    return text[start:end].strip() or text


def _parse_new_queries(text: str, exclude: str) -> List[str]:
    matches = list(NEW_QUERY_MARKER_RE.finditer(text))
    if not matches:
        return []
    m = matches[-1]
    after = text[m.end() :]
    for pat in (NEW_QUERY_MARKER_RE, FINAL_MARKER_RE):
        nxt = pat.search(after)
        if nxt is not None:
            after = after[: nxt.start()]

    out: List[str] = []
    for line in after.splitlines():
        c = line.strip()
        if not c or c == ":":
            continue
        if re.fullmatch(r"[-\*_=~]{3,}", c):
            continue
        if NEW_QUERY_MARKER_RE.match(c) or FINAL_MARKER_RE.match(c):
            continue
        c = re.sub(r"^[\-\*\u2022]\s*", "", c)
        c = re.sub(r"^\d+[\.\)]\s*", "", c)
        c = c.strip().strip('"').strip("'").strip()
        if not c or c == exclude or c in out:
            continue
        out.append(c)
    return out


async def answer_question(question: str, dataset_name: str, client_id: str) -> Dict:
    """Full agent-mode QA pipeline. Returns a dict matching QuestionResponse."""
    if not GRAPHRAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG components not available. Please install or configure them.",
        )

    await send_progress_update(
        client_id, "retrieval", 10, "Initializing retrieval system (agent mode)..."
    )

    graph_path = f"output/graphs/{dataset_name}_new.json"
    schema_path = get_schema_path_for_dataset(dataset_name)
    if not os.path.exists(graph_path):
        graph_path = "output/graphs/demo_new.json"
    if not os.path.exists(graph_path):
        raise HTTPException(
            status_code=404, detail="Graph not found. Please construct graph first."
        )

    cfg = get_shared_config()

    graphq = decomposer.GraphQ(dataset_name, config=cfg)
    kt_retriever = retriever.KTRetriever(
        dataset_name,
        graph_path,
        recall_paths=cfg.retrieval.recall_paths,
        schema_path=schema_path,
        top_k=cfg.retrieval.top_k_filter,
        mode="agent",
        config=cfg,
    )

    await send_progress_update(client_id, "retrieval", 40, "Building indices...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kt_retriever.build_indices)

    await send_event(
        client_id,
        {
            "type": "qa_update",
            "stage": "start",
            "message": "Question processing started",
            "dataset": dataset_name,
            "question": question,
            "timestamp": datetime.now().isoformat(),
        },
    )
    await asyncio.sleep(0)

    # ---- Step 1: decompose -----------------------------------------------
    await send_progress_update(client_id, "retrieval", 50, "Decomposing question...")
    try:
        decomposition = await loop.run_in_executor(
            None, lambda: graphq.decompose(question, schema_path)
        )
        sub_questions = decomposition.get("sub_questions", [])
        involved_types = decomposition.get("involved_types", {})
        await send_event(
            client_id,
            {
                "type": "qa_update",
                "stage": "decompose",
                "sub_questions_count": len(sub_questions),
                "sub_questions": [sq.get("sub-question", "") for sq in sub_questions][
                    :5
                ],
                "timestamp": datetime.now().isoformat(),
            },
        )
        await asyncio.sleep(0.05)
    except Exception as e:
        logger.error(f"Decompose failed: {e}")
        sub_questions = [{"sub-question": question}]
        involved_types = {"nodes": [], "relations": [], "attributes": []}

    reasoning_steps: List[Dict] = []
    all_triples: set = set()
    all_chunk_ids: set = set()
    all_chunk_contents: Dict[str, str] = {}

    # ---- Step 2: initial retrieval per sub-question -----------------------
    await send_progress_update(client_id, "retrieval", 65, "Initial retrieval...")
    for idx, sq in enumerate(sub_questions):
        sq_text = sq.get("sub-question", question)

        retrieval_results, elapsed = await loop.run_in_executor(
            None,
            lambda s=sq_text: kt_retriever.process_retrieval_results(
                s,
                top_k=cfg.retrieval.top_k_filter,
                involved_types=involved_types,
            ),
        )
        triples = retrieval_results.get("triples", []) or []
        chunk_ids = retrieval_results.get("chunk_ids", []) or []
        chunk_contents = retrieval_results.get("chunk_contents", []) or []
        if isinstance(chunk_contents, dict):
            all_chunk_contents.update(chunk_contents)
        else:
            for i_c, cid in enumerate(chunk_ids):
                if i_c < len(chunk_contents):
                    all_chunk_contents[cid] = chunk_contents[i_c]

        all_triples.update(triples)
        all_chunk_ids.update(chunk_ids)
        reasoning_steps.append(
            {
                "type": "sub_question",
                "question": sq_text,
                "triples": triples[:10],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": elapsed,
                "chunk_contents": list(all_chunk_contents.values())[:3],
            }
        )

        await send_event(
            client_id,
            {
                "type": "qa_update",
                "stage": "sub_question",
                "index": idx + 1,
                "total": len(sub_questions),
                "question": sq_text,
                "triples_preview": list(dict.fromkeys(triples))[:5],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": elapsed,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await asyncio.sleep(0)

    # ---- Step 3: single reasoning call -----------------------------------
    await send_progress_update(client_id, "retrieval", 75, "Reasoning...")
    await send_event(
        client_id,
        {
            "type": "qa_update",
            "stage": "ircot_start",
            "message": "Starting reasoning",
            "timestamp": datetime.now().isoformat(),
        },
    )
    await asyncio.sleep(0.05)

    initial_query = question
    current_query = question

    def _build_context():
        triples = _dedup(list(all_triples))
        chunk_ids = list(set(all_chunk_ids))
        chunk_contents = _merge_chunk_contents(chunk_ids, all_chunk_contents)
        ctx = (
            "=== Triples ===\n"
            + "\n".join(triples[:20])
            + "\n=== Chunks ===\n"
            + "\n---\n".join(chunk_contents[:10])
        )
        return ctx, triples, chunk_ids, chunk_contents

    async def _llm_call(query_text: str, context: str, step: int) -> str:
        prompt = kt_retriever.generate_ircot_prompt(
            initial_query=initial_query,
            current_query=query_text,
            context=context,
            previous_thoughts="",
            step=step,
        )
        try:
            return await loop.run_in_executor(
                None, lambda p=prompt: kt_retriever.generate_answer(p)
            )
        except Exception as e:
            logger.error(f"LLM call (step {step}) failed: {e}")
            return f"Reasoning error: {e}"

    def _record_step(query_text, triples, chunk_ids, chunk_contents, thought):
        reasoning_steps.append(
            {
                "type": "ircot_step",
                "question": query_text,
                "triples": triples[:10],
                "triples_count": len(triples),
                "chunks_count": len(chunk_ids),
                "processing_time": 0,
                "chunk_contents": chunk_contents[:3],
                "thought": (thought or "")[:300],
            }
        )

    ctx1, t1, ids1, cc1 = _build_context()
    reasoning = await _llm_call(current_query, ctx1, step=1)
    _record_step(current_query, t1, ids1, cc1, reasoning)

    await send_event(
        client_id,
        {
            "type": "qa_update",
            "stage": "ircot",
            "step": 1,
            "current_query": current_query,
            "thought_preview": (reasoning or "")[:200],
            "timestamp": datetime.now().isoformat(),
        },
    )
    await asyncio.sleep(0)

    final_answer = _extract_final_answer(reasoning)
    new_queries = _parse_new_queries(reasoning, exclude=current_query)

    if final_answer is None and new_queries:
        final_answer = "; ".join(q.strip() for q in new_queries if q and q.strip())
    if final_answer is None:
        final_answer = reasoning or "Unable to generate an answer."

    # ---- aggregation -----------------------------------------------------
    final_triples = _dedup(list(all_triples))[:20]
    final_chunk_ids = list(set(all_chunk_ids))
    final_chunk_contents = _merge_chunk_contents(final_chunk_ids, all_chunk_contents)[
        :10
    ]

    await send_progress_update(
        client_id, "retrieval", 100, "Answer generation completed!"
    )
    await send_event(
        client_id,
        {
            "type": "qa_complete",
            "answer_preview": (final_answer or "")[:300],
            "sub_questions_count": len(sub_questions),
            "triples_final_count": len(final_triples),
            "chunks_final_count": len(final_chunk_contents),
            "timestamp": datetime.now().isoformat(),
        },
    )

    visualization_data = {
        "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
        "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
        "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
        "retrieval_details": {
            "total_triples": len(final_triples),
            "total_chunks": len(final_chunk_contents),
            "sub_questions_count": len(sub_questions),
            "triples_by_subquery": [
                s.get("triples_count", 0)
                for s in reasoning_steps
                if s.get("type") == "sub_question"
            ],
        },
    }

    return {
        "answer": final_answer,
        "sub_questions": sub_questions,
        "retrieved_triples": final_triples,
        "retrieved_chunks": final_chunk_contents,
        "reasoning_steps": reasoning_steps,
        "visualization_data": visualization_data,
    }


# ===========================================================================
# DATASET HELPERS
# ===========================================================================
def list_datasets() -> Dict:
    datasets: List[Dict] = []

    upload_dir = "data/uploaded"
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if not os.path.isdir(item_path):
                continue
            if not os.path.exists(os.path.join(item_path, "corpus.json")):
                continue
            graph_path = f"output/graphs/{item}_new.json"
            datasets.append(
                {
                    "name": item,
                    "type": "uploaded",
                    "status": (
                        "ready" if os.path.exists(graph_path) else "needs_construction"
                    ),
                    "has_custom_schema": os.path.exists(f"schemas/{item}.json"),
                }
            )

    if os.path.exists("data/demo/demo_corpus.json"):
        datasets.append(
            {
                "name": "demo",
                "type": "demo",
                "status": (
                    "ready"
                    if os.path.exists("output/graphs/demo_new.json")
                    else "needs_construction"
                ),
                "has_custom_schema": False,
            }
        )

    return {"datasets": datasets}


def delete_dataset_files(dataset_name: str) -> Dict:
    if dataset_name == "demo":
        raise HTTPException(status_code=400, detail="Cannot delete demo dataset")

    deleted_files: List[str] = []
    candidates = [
        f"data/uploaded/{dataset_name}",
        f"output/graphs/{dataset_name}_new.json",
        f"schemas/{dataset_name}.json",
        f"retriever/faiss_cache_new/{dataset_name}",
        f"output/chunks/{dataset_name}.txt",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        deleted_files.append(path)

    return {
        "success": True,
        "message": f"Dataset '{dataset_name}' deleted successfully",
        "deleted_files": deleted_files,
    }


# ===========================================================================
# ROUTES
# ===========================================================================
@app.get("/")
async def read_root():
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Youtu-GraphRAG Unified Interface is running!", "status": "ok"}


@app.get("/api/status")
async def get_status():
    return {
        "message": "Youtu-GraphRAG Unified Interface is running!",
        "status": "ok",
        "graphrag_available": GRAPHRAG_AVAILABLE,
    }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...), client_id: str = "default"):
    """Upload files and prepare corpus.json for graph construction."""
    try:
        dataset_name, processed_count, skipped = await process_uploaded_files(
            files, client_id
        )
    except HTTPException:
        raise
    except Exception as e:
        await send_progress_update(client_id, "upload", 0, f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    msg = "Files uploaded successfully"
    if skipped:
        msg += f"; skipped unsupported: {', '.join(skipped)}"
    return FileUploadResponse(
        success=True,
        message=msg,
        dataset_name=dataset_name,
        files_count=processed_count,
    )


@app.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(
    request: GraphConstructionRequest, client_id: str = "default"
):
    """Build a knowledge graph from an uploaded corpus."""
    try:
        result = await run_graph_construction(
            request.dataset_name,
            client_id,
            stage="construction",
            demo_fallback=True,
            include_visualization=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        await send_progress_update(
            client_id, "construction", 0, f"Construction failed: {e}"
        )
        await emit_graph_error(client_id, "construction", f"Construction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return GraphConstructionResponse(
        success=True,
        message="Knowledge graph constructed successfully",
        graph_data=result["graph_vis_data"],
    )


@app.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(dataset_name: str, client_id: str = "default"):
    """Rebuild the graph for an existing dataset (removes old graph + cache)."""
    try:
        await run_graph_construction(
            dataset_name,
            client_id,
            stage="reconstruction",
            demo_fallback=(dataset_name == "demo"),
            include_visualization=False,
        )
    except HTTPException:
        raise
    except Exception as e:
        await send_progress_update(
            client_id, "reconstruction", 0, f"Reconstruction failed: {e}"
        )
        await emit_graph_error(
            client_id, "reconstruction", f"Reconstruction failed: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "message": "Dataset reconstructed successfully",
        "dataset_name": dataset_name,
    }


@app.get("/api/graph/{dataset_name}")
async def get_graph_data(dataset_name: str):
    """Get graph visualization data (falls back to demo sample)."""
    graph_path = f"output/graphs/{dataset_name}_new.json"
    if not os.path.exists(graph_path):
        return {
            "nodes": [
                {
                    "id": "node1",
                    "name": "Example Entity 1",
                    "category": "person",
                    "value": 5,
                    "symbolSize": 25,
                },
                {
                    "id": "node2",
                    "name": "Example Entity 2",
                    "category": "location",
                    "value": 3,
                    "symbolSize": 20,
                },
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {
                "total_nodes": 2,
                "total_edges": 1,
                "displayed_nodes": 2,
                "displayed_edges": 1,
            },
        }
    return await prepare_graph_visualization(graph_path)


@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, client_id: str = "default"):
    try:
        result = await answer_question(
            request.question, request.dataset_name, client_id
        )
    except HTTPException:
        raise
    except Exception as e:
        await send_progress_update(
            client_id, "retrieval", 0, f"Question answering failed: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    return QuestionResponse(**result)


@app.get("/api/datasets")
async def get_datasets():
    """List available datasets."""
    return list_datasets()


@app.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(dataset_name: str, schema_file: UploadFile = File(...)):
    """Upload a custom schema JSON for a dataset."""
    if dataset_name == "demo":
        raise HTTPException(
            status_code=400, detail="Cannot upload schema for demo dataset"
        )
    if not (schema_file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Schema file must be a .json file")

    content = await schema_file.read()
    try:
        data = json.loads(decode_bytes_with_detection(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Schema JSON must be an object")

    os.makedirs("schemas", exist_ok=True)
    with open(f"schemas/{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "message": "Schema uploaded successfully",
        "dataset_name": dataset_name,
    }


@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its associated files."""
    try:
        return delete_dataset_files(dataset_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")


# ===========================================================================
# STARTUP
# ===========================================================================
@app.on_event("startup")
async def startup_event():
    for d in ("data/uploaded", "output/graphs", "output/logs", "schemas"):
        os.makedirs(d, exist_ok=True)
    logger.info("🚀 Youtu-GraphRAG Unified Interface initialized")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
