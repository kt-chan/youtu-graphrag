# backend.py — FastAPI entry point, orchestration pipelines, routes.
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional

from fastapi import (
    FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.backend.retriever import kt_retriever as retriever
from app.utils.logger import logger
from app.utils.encoding import decode_bytes_with_detection
from app.utils.schema_utils import ensure_demo_schema_exists, get_schema_path_for_dataset
from app.utils.cache_utils import clear_cache_files
from app.utils.ws_manager import ConnectionManager
from app.utils import visualization as viz
from app.utils import dataset_manager as ds


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from app.utils.document_parser import get_parser
    DOCUMENT_PARSER_AVAILABLE = True
except ImportError as e:
    DOCUMENT_PARSER_AVAILABLE = False
    logger.warning(f"Document parser not available: {e}")

try:
    from backend.constructor import kt_gen as constructor
    from backend.retriever import (
        agentic_decomposer as decomposer,
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

# Directories based on location of main.py
BASE_DIR = Path(__file__).resolve().parent          # .../app

FRONTEND_DIR = BASE_DIR / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"                # Use BASE_DIR / "assets" if assets was also moved into app/

if FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

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
config = None                # lazily initialised base config
manager = ConnectionManager()


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
# Pydantic request / response models
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
# Shared helpers
# ===========================================================================
ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".pdf", ".docx", ".doc"}
PLAIN_TEXT_EXT = {".txt", ".md"}
DOC_PARSER_EXT = {".pdf", ".docx", ".doc"}


def _dedup(items):
    """Order-preserving unique."""
    return list({x: None for x in items}.keys())


def _merge_chunk_contents(ids, mapping) -> List[str]:
    return [
        f"[Chunk {idx}] {mapping.get(i, f'[Missing content for chunk {i}]')}"
        for idx, i in enumerate(ids, 1)
    ]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(raw: str) -> Dict:
    """Recover a decision dict from the LLM response.

    Tolerates raw JSON, ```json fenced``` blocks, and prose-wrapped JSON.
    Falls back to ``{"decision_type": "FINAL_ANSWER", "final_answer": raw}``
    so the pipeline never crashes on an unparseable response.
    """
    if not raw:
        return {"decision_type": "FINAL_ANSWER", "final_answer": ""}

    text = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _OBJECT_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("LLM response was not valid JSON; treating as final answer.")
    return {"decision_type": "FINAL_ANSWER", "final_answer": raw}


# ===========================================================================
# Graph construction pipeline
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

    Used by both /api/construct-graph and /api/datasets/{name}/reconstruct.
    """
    if not GRAPHRAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG components not available. Please install or configure them.",
        )

    cfg = get_shared_config()

    if not cfg.system.debug:
        await manager.progress(client_id, stage, 5, "Cleaning old cache files...")
        await clear_cache_files(dataset_name)

    corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
    if not os.path.exists(corpus_path) and demo_fallback:
        demo_corpus = "data/demo/demo_corpus.json"
        if os.path.exists(demo_corpus):
            corpus_path = demo_corpus
    if not os.path.exists(corpus_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    schema_path = get_schema_path_for_dataset(dataset_name)

    await manager.progress(client_id, stage, 10, "Loading configuration and corpus...")

    builder = constructor.KTBuilder(
        dataset_name, schema_path, mode=cfg.construction.mode, config=cfg
    )

    await manager.progress(client_id, stage, 20, "Starting entity-relation extraction...")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: builder.build_knowledge_graph(corpus_path))

    graph_vis_data: Dict = {}
    if include_visualization:
        await manager.progress(client_id, stage, 95, "Preparing visualization data...")
        graph_vis_data = await loop.run_in_executor(
            None,
            viz.prepare_graph_visualization,
            f"output/graphs/{dataset_name}_new.json",
        )

    await manager.progress(client_id, stage, 100, "Graph construction completed!")
    await manager.event(client_id, {
        "type": "complete",
        "stage": stage,
        "message": "Graph construction completed!",
    })

    return {"graph_vis_data": graph_vis_data}


# ===========================================================================
# Upload pipeline
# ===========================================================================
async def process_uploaded_files(files: List[UploadFile], client_id: str) -> tuple:
    """Persist uploads, extract text, write corpus.json.

    Returns (dataset_name, processed_count, skipped_files).
    """
    dataset_name = ds.unique_dataset_name(ds.derive_dataset_name(files))
    upload_dir = f"data/uploaded/{dataset_name}"
    os.makedirs(upload_dir, exist_ok=True)

    await manager.progress(client_id, "upload", 10, "Starting file upload...")

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
            await manager.progress(
                client_id, "upload", progress,
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
                    await manager.progress(
                        client_id, "upload", progress,
                        f"Skipped {file.filename} (parser unavailable)",
                    )
                    continue
                text = doc_parser.parse_file(file_path, ext)
                if text and text.strip():
                    corpus_data.append({"title": file.filename, "text": text})
                    processed_count += 1
                    await manager.progress(
                        client_id, "upload", progress, f"Parsed {file.filename}"
                    )
                else:
                    logger.warning(f"No text extracted from {file.filename}")
                    skipped_files.append(file.filename)
                    await manager.progress(
                        client_id, "upload", progress, f"No text in {file.filename}"
                    )
                continue

            if ext in PLAIN_TEXT_EXT:
                corpus_data.append({
                    "title": file.filename,
                    "text": decode_bytes_with_detection(content_bytes),
                })
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
                    corpus_data.append({
                        "title": file.filename,
                        "text": decode_bytes_with_detection(content_bytes),
                    })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            skipped_files.append(file.filename)
            await manager.progress(
                client_id, "upload", progress, f"Failed to process {file.filename}"
            )
            continue

        await manager.progress(
            client_id, "upload", progress, f"Processed {file.filename}"
        )

    if processed_count == 0:
        msg = (
            "No supported files were uploaded. "
            "Allowed: .txt, .md, .json, .pdf, .docx, .doc"
        )
        if skipped_files:
            msg += f"; skipped: {', '.join(skipped_files)}"
        await manager.progress(client_id, "upload", 0, msg)
        raise HTTPException(status_code=400, detail=msg)

    with open(f"{upload_dir}/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    ensure_demo_schema_exists()

    await manager.progress(client_id, "upload", 100, "Upload completed successfully!")
    return dataset_name, processed_count, skipped_files


# ===========================================================================
# Question-answering pipeline
# ===========================================================================
async def answer_question(question: str, dataset_name: str, client_id: str) -> Dict:
    """Full agent-mode QA pipeline. Returns a dict matching QuestionResponse."""
    if not GRAPHRAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG components not available. Please install or configure them.",
        )

    await manager.progress(
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

    await manager.progress(client_id, "retrieval", 40, "Building indices...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kt_retriever.build_indices)

    await manager.qa_update(client_id, {
        "stage": "start",
        "message": "Question processing started",
        "dataset": dataset_name,
        "question": question,
    })

    # ---- Step 1: decompose -----------------------------------------------
    await manager.progress(client_id, "retrieval", 50, "Decomposing question...")
    try:
        decomposition = await loop.run_in_executor(
            None, lambda: graphq.decompose(question, schema_path)
        )
        sub_questions = decomposition.get("sub_questions", [])
        involved_types = decomposition.get("involved_types", {})
        await manager.qa_update(client_id, {
            "stage": "decompose",
            "sub_questions_count": len(sub_questions),
            "sub_questions": [
                sq.get("sub-question", "") for sq in sub_questions
            ][:5],
        })
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
    await manager.progress(client_id, "retrieval", 65, "Initial retrieval...")
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

        # `chunk_contents` may be a dict {id: text} (post-refactor) or a list
        # aligned with `chunk_ids` (legacy).  Handle both.
        if isinstance(chunk_contents, dict):
            all_chunk_contents.update(chunk_contents)
        else:
            for i_c, cid in enumerate(chunk_ids):
                if i_c < len(chunk_contents):
                    all_chunk_contents[cid] = chunk_contents[i_c]

        all_triples.update(triples)
        all_chunk_ids.update(chunk_ids)

        reasoning_steps.append({
            "type": "sub_question",
            "question": sq_text,
            "triples": triples[:10],
            "triples_count": len(triples),
            "chunks_count": len(chunk_ids),
            "processing_time": elapsed,
            "chunk_contents": list(all_chunk_contents.values())[:3],
        })

        await manager.qa_update(client_id, {
            "stage": "sub_question",
            "index": idx + 1,
            "total": len(sub_questions),
            "question": sq_text,
            "triples_preview": list(dict.fromkeys(triples))[:5],
            "triples_count": len(triples),
            "chunks_count": len(chunk_ids),
            "processing_time": elapsed,
        })

    # ---- Step 3: single reasoning call -----------------------------------
    await manager.progress(client_id, "retrieval", 75, "Reasoning...")
    await manager.qa_update(client_id, {
        "stage": "ircot_start",
        "message": "Starting reasoning",
    })
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

    async def _llm_call(query_text: str, context: str) -> str:
        prompt = kt_retriever.generate_ircot_prompt(
            initial_query=initial_query,
            current_query=query_text,
            context=context,
        )
        try:
            return await loop.run_in_executor(
                None, lambda p=prompt: kt_retriever.generate_answer(p)
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Reasoning error: {e}"

    def _record_step(query_text, triples, chunk_ids, chunk_contents, thought):
        reasoning_steps.append({
            "type": "ircot_step",
            "question": query_text,
            "triples": triples[:10],
            "triples_count": len(triples),
            "chunks_count": len(chunk_ids),
            "processing_time": 0,
            "chunk_contents": chunk_contents[:3],
            "thought": (thought or "")[:300],
        })

    ctx1, t1, ids1, cc1 = _build_context()
    reasoning = await _llm_call(current_query, ctx1)
    _record_step(current_query, t1, ids1, cc1, reasoning)

    await manager.qa_update(client_id, {
        "stage": "ircot",
        "current_query": current_query,
        "thought_preview": (reasoning or "")[:200],
    })

    # ---- Parse LLM decision ---------------------------------------------
    #  Expected JSON: {reasoning, decision_type, final_answer, new_queries}
    parsed = _parse_llm_json(reasoning)
    decision = str(parsed.get("decision_type", "")).lower()

    if decision == "final_answer":
        final_answer = parsed.get("final_answer") or reasoning
    elif decision == "new_queries":
        queries = parsed.get("new_queries") or []
        final_answer = "; ".join(str(q) for q in queries) if isinstance(
            queries, list
        ) else str(queries)
    else:
        final_answer = parsed.get("final_answer") or reasoning

    if not isinstance(final_answer, str):
        final_answer = str(final_answer)
    if not final_answer:
        final_answer = "Unable to generate an answer."

    # ---- Aggregation -----------------------------------------------------
    final_triples = _dedup(list(all_triples))[:20]
    final_chunk_ids = list(set(all_chunk_ids))
    final_chunk_contents = _merge_chunk_contents(
        final_chunk_ids, all_chunk_contents
    )[:10]

    await manager.progress(
        client_id, "retrieval", 100, "Answer generation completed!"
    )
    await manager.qa_update(client_id, {
        "stage": "ircot_complete",
        "answer_preview": (final_answer or "")[:300],
        "sub_questions_count": len(sub_questions),
        "triples_final_count": len(final_triples),
        "chunks_final_count": len(final_chunk_contents),
    })

    visualization_data = {
        "subqueries": viz.prepare_subquery_visualization(sub_questions, reasoning_steps),
        "knowledge_graph": viz.prepare_retrieved_graph_visualization(final_triples),
        "reasoning_flow": viz.prepare_reasoning_flow_visualization(reasoning_steps),
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
# Routes
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
async def upload_files(
    files: List[UploadFile] = File(...), client_id: str = "default"
):
    """Upload files and prepare corpus.json for graph construction."""
    try:
        dataset_name, processed_count, skipped = await process_uploaded_files(
            files, client_id
        )
    except HTTPException:
        raise
    except Exception as e:
        await manager.progress(client_id, "upload", 0, f"Upload failed: {e}")
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
        await manager.progress(
            client_id, "construction", 0, f"Construction failed: {e}"
        )
        await manager.error(client_id, "construction", f"Construction failed: {e}")
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
        await manager.progress(
            client_id, "reconstruction", 0, f"Reconstruction failed: {e}"
        )
        await manager.error(
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
                {"id": "node1", "name": "Example Entity 1",
                 "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "Example Entity 2",
                 "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2",
                 "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {
                "total_nodes": 2, "total_edges": 1,
                "displayed_nodes": 2, "displayed_edges": 1,
            },
        }
    return viz.prepare_graph_visualization(graph_path)


@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, client_id: str = "default"):
    try:
        result = await answer_question(
            request.question, request.dataset_name, client_id
        )
    except HTTPException:
        raise
    except Exception as e:
        await manager.progress(
            client_id, "retrieval", 0, f"Question answering failed: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    return QuestionResponse(**result)


@app.get("/api/datasets")
async def get_datasets():
    """List available datasets."""
    return ds.list_datasets()


@app.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(
    dataset_name: str, schema_file: UploadFile = File(...)
):
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
    os.makedirs(f"schemas/{dataset_name}", exist_ok=True)
    with open(f"schemas/{dataset_name}/schema.json", "w", encoding="utf-8") as f:
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
        return ds.delete_dataset_files(dataset_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")


# ===========================================================================
# Startup
# ===========================================================================
@app.on_event("startup")
async def startup_event():
    for d in ("data/uploaded", "output/graphs", "output/logs", "schemas"):
        os.makedirs(d, exist_ok=True)
    logger.info("🚀 Youtu-GraphRAG Unified Interface initialized")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)