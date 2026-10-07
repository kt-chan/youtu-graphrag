"""utils/dataset_manager.py — Dataset-level filesystem operations.

Knows where uploaded corpora and derived graphs live.  No FastAPI
coupling beyond HTTPException for consistent error surfaces.
"""
from __future__ import annotations

import os
import glob
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
                "has_custom_schema": os.path.exists(f"schemas/{item}/schema.json"),
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
    
    # Use glob patterns to catch directories and all file variations (e.g., .json, _new.json, .txt)
    patterns = [
        f"data/uploaded/{dataset_name}",
        f"output/graphs/{dataset_name}*",
        f"schemas/{dataset_name}*",
        f"retriever/faiss_cache_new/{dataset_name}",
        f"output/chunks/{dataset_name}*",
    ]

    for pattern in patterns:
        for path in glob.glob(pattern):
            if path in deleted_files:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
            deleted_files.append(path)

    return {
        "success": True,
        "message": f"Dataset '{dataset_name}' deleted successfully",
        "deleted_files": deleted_files,
    }