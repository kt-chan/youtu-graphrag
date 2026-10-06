"""Checkpoint persistence for knowledge-tree construction.

A checkpoint captures the fully-constructed (pre-Level-3/4) graph plus
metadata so a debug run can skip the expensive LLM phase entirely.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Optional

from utils.logger import logger


def checkpoint_path(dataset_name: str) -> str:
    return f"output/graphs/{dataset_name}_checkpoint.pkl"


def checkpoint_exists(dataset_name: str) -> bool:
    return os.path.exists(checkpoint_path(dataset_name))


def save_checkpoint(
    dataset_name: str,
    graph,
    all_chunks: Dict[str, str],
    node_counter: int,
    metrics: Dict[str, int],
    token_len: int,
    token_len_per_chunk: Dict[str, int],
    mode: str,
    path: Optional[str] = None,
) -> None:
    path = path or checkpoint_path(dataset_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "graph": graph,
        "all_chunks": all_chunks,
        "node_counter": node_counter,
        "metrics": metrics,
        "token_len": token_len,
        "token_len_per_chunk": token_len_per_chunk,
        "mode": mode,
        "dataset_name": dataset_name,
        "saved_at": time.time(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        f"Checkpoint saved: {path} "
        f"(nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()})"
    )


def load_checkpoint(
    dataset_name: str, path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the checkpoint payload, or ``None`` if no checkpoint exists."""
    path = path or checkpoint_path(dataset_name)
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)