"""utils/dataset_manager.py — Dataset-level filesystem operations.

Knows where uploaded corpora and derived graphs live.  No FastAPI
coupling beyond HTTPException for consistent error surfaces.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Dict, List

from fastapi import HTTPException, UploadFile


def derive_dataset_name(files: List[UploadFile]) -> str:
    """Single upload → its filename; multi upload → ``Nfiles_YYYYMMDD``."""
    if len(files) == 1:
        original = os.path.splitext(files[0].filename or "dataset")[0]
        cleaned = "".join(
            c for c in original if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        return cleaned.replace(" ", "_") or "dataset"
    return f"{len(files)}files_{datetime.now().strftime('%Y%m%d')}"


def unique_dataset_name(base: str) -> str:
    name = base
    counter = 1
    while os.path.exists(f"data/uploaded/{name}"):
        name = f"{base}_{counter}"
        counter += 1
    return name


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
            datasets.append({
                "name": item,
                "type": "uploaded",
                "status": "ready" if os.path.exists(graph_path) else "needs_construction",
                "has_custom_schema": os.path.exists(f"schemas/{item}.json"),
            })

    if os.path.exists("data/demo/demo_corpus.json"):
        datasets.append({
            "name": "demo",
            "type": "demo",
            "status": (
                "ready" if os.path.exists("output/graphs/demo_new.json")
                else "needs_construction"
            ),
            "has_custom_schema": False,
        })

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
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        deleted_files.append(path)

    return {
        "success": True,
        "message": f"Dataset '{dataset_name}' deleted successfully",
        "deleted_files": deleted_files,
    }