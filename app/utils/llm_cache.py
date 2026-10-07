"""Thread-safe, disk-backed cache for LLM responses keyed by chunk_id."""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional

from app.utils.logger import logger


class LLMResponseCache:
    """Persist LLM outputs so runs can be resumed across crashes / restarts.

    Load rule: if ``output/chunks/{dataset}.txt`` exists, we assume a prior
    construction run happened; any matching cache file is restored.  A
    missing / corrupt cache is treated as a cold start (no error).
    """

    def __init__(self, dataset_name: str, chunks_path: Optional[str] = None):
        self.dataset_name = dataset_name
        self.chunks_path = chunks_path or f"output/chunks/{dataset_name}.txt"
        self._lock = threading.Lock()
        self.responses: Dict[str, str] = {}

    @property
    def path(self) -> str:
        return f"output/chunks/{self.dataset_name}_responses.json"

    # ── Persistence ──────────────────────────────────────────────────────
    def load(self) -> None:
        if not os.path.exists(self.chunks_path):
            logger.info(
                f"[{self.dataset_name}] No prior chunks file — cold start, "
                f"LLM will be called for every chunk."
            )
            return

        if not os.path.exists(self.path):
            logger.info(
                f"[{self.dataset_name}] Chunks file present but no cached "
                f"responses at {self.path}; LLM will be called."
            )
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected cache format: {type(data).__name__}")
            self.responses = {str(k): str(v) for k, v in data.items()}
            logger.info(
                f"[{self.dataset_name}] Loaded {len(self.responses)} "
                f"cached LLM responses from {self.path}."
            )
        except Exception as e:
            logger.warning(
                f"[{self.dataset_name}] Failed to load response cache "
                f"({type(e).__name__}: {e}); starting cold."
            )
            self.responses = {}

    def save(self) -> None:
        """Atomic write — safe to call concurrently from multiple threads."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with self._lock:
                snapshot = dict(self.responses)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
        except Exception as e:
            logger.warning(
                f"[{self.dataset_name}] Failed to save response cache: "
                f"{type(e).__name__}: {e}"
            )

    # ── Accessors ────────────────────────────────────────────────────────
    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self.responses

    def get(self, chunk_id: str) -> Optional[str]:
        return self.responses.get(chunk_id)

    def set(self, chunk_id: str, response: str) -> None:
        with self._lock:
            self.responses[chunk_id] = response