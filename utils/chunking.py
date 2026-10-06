"""Dataset-aware text chunking for knowledge-tree construction."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken


def stable_chunk_id(chunk: str) -> str:
    """Deterministic SHA1-based chunk identifier (12 hex chars)."""
    return hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]


def _get_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("gpt2")


def split_text_with_overlap(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_tail_tokens: int = 100,
) -> List[str]:
    encoding = _get_encoding()
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

    # Merge short tail windows into their neighbours.
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


class TextChunker:
    """Dataset-aware wrapper around `split_text_with_overlap`."""

    def __init__(
        self,
        dataset_name: str,
        datasets_no_chunk: Optional[Set[str]] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_tail_tokens: int = 100,
    ):
        self.dataset_name = dataset_name
        self.datasets_no_chunk = datasets_no_chunk or set()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tail_tokens = min_tail_tokens

    def chunk(self, text: Any) -> Tuple[List[str], Dict[str, str]]:
        """Return ``(chunks_list, {chunk_id: chunk_text})``."""
        if self.dataset_name in self.datasets_no_chunk:
            if isinstance(text, dict):
                chunk = (
                    f"Labels='{text.get('title', '')}' "
                    f"Content='{text.get('text', '')}'"
                ).strip()
            else:
                chunk = str(text)
            chunks = [chunk]
        else:
            if isinstance(text, dict):
                raw = (
                    f"Labels='{text.get('title', '')}' {text.get('text', '')}"
                ).strip()
            else:
                raw = str(text)
            chunks = split_text_with_overlap(
                raw, self.chunk_size, self.overlap, self.min_tail_tokens
            )

        chunk2id = {stable_chunk_id(c): c for c in chunks}
        return chunks, chunk2id