"""utils/logger.py — Unified colored logger + dedicated LLM exchange logger.

Exports:
    logger              — main application logger (colored console)
    llm_logger          — dedicated LLM exchange logger (file only, UTF-8, JSONL)
    setup_logger        — factory for additional named loggers
    progress            — one-liner progress helper
    log_llm_exchange    — record a single (prompt, response) pair as JSONL
"""

import json
import logging
import os
import sys
from typing import Optional

__all__ = [
    "logger",
    "llm_logger",
    "setup_logger",
    "progress",
    "log_llm_exchange",
]

# ── ANSI color codes ────────────────────────────────────────────────────────
COLORS = {
    "DEBUG":    "\033[0;36m",  # Cyan
    "INFO":     "\033[0;32m",  # Green
    "WARNING":  "\033[0;33m",  # Yellow
    "ERROR":    "\033[0;31m",  # Red
    "CRITICAL": "\033[0;35m",  # Magenta
    "RESET":    "\033[0m",
}

# Override with $LLM_LOG_FILE if you want a different location.
_LLM_LOG_DEFAULT = os.environ.get("LLM_LOG_FILE", "output/logs/llm.log")

_FMT = "[%(asctime)s] %(levelname)-8s %(module)s:%(lineno)d - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Color the entire log line by level (console only)."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        color = COLORS.get(record.levelname)
        return f"{color}{formatted}{COLORS['RESET']}" if color else formatted


def setup_logger(
    name: str = "Auto-GraphRAG",
    level: int = logging.INFO,
    log_file: Optional[str] =  "output/logs/llm.log",
    use_colors: bool = True,
) -> logging.Logger:
    """Build (or rebuild) a named logger.

    Args:
        name:        Logger name; used as the registry key.
        level:       Minimum level for handlers.
        log_file:    Optional file path.  Parent dir is created automatically.
        use_colors:  Emit colored output to stdout.  Set False for file-only
                     loggers so they never pollute the console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if use_colors:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            ColoredFormatter(fmt=_FMT, datefmt=_DATEFMT)
        )
        logger.addHandler(console_handler)


    try:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)
    except OSError as e:
        # Never let a broken log file crash the pipeline.
        logger.warning(f"Could not open log file {log_file!r}: {e}")

    return logger


# ── Main application logger ────────────────────────────────────────────────
logger = setup_logger("Auto-GraphRAG", level=logging.INFO, use_colors=True)

# ── Dedicated LLM exchange logger (file only, no console noise) ────────────
llm_logger = setup_logger(
    "Auto-GraphRAG.llm",
    level=logging.DEBUG,
    log_file=_LLM_LOG_DEFAULT,
    use_colors=False,
)


def progress(stage: str, message: str, *, done: Optional[bool] = None) -> None:
    """Unified progress logging helper.

    Args:
        stage:   Short stage/category name.
        message: Detail message.
        done:    Optional completion flag (✅ / ❌).
    """
    suffix = ""
    if done is True:
        suffix = " ✅"
    elif done is False:
        suffix = " ❌"
    logger.info(f"[{stage}] {message}{suffix}")


def log_llm_exchange(
    chunk_id: str,
    prompt: str,
    response: Optional[str],
    *,
    stage: str = "llm",
    max_chars: Optional[int] = None,
) -> None:
    """Record one LLM (prompt, response) exchange as JSONL.

    Every `llm_logger.info` call here emits exactly ONE physical line of the
    log file: a compact JSON object.  Any newlines inside the prompt or
    response are escaped by `json.dumps`, so a multi-line payload never
    splits a log record.

    Args:
        chunk_id:  Identifier of the chunk the exchange belongs to.
        prompt:    Full prompt sent to the model.
        response:  Raw model response (may be None on transport failure).
        stage:     Sub-label, e.g. "debt_collection:fresh" or ":cache".
        max_chars: Truncate either payload past this length. None = log in full.
    """

    def _clip(text: Optional[str]) -> str:
        """Return `text` on a single logical line (newlines escaped by JSON)."""
        if not text:
            return ""
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + f"... [truncated: {len(text)} chars total]"
        return text

    llm_logger.info(
        json.dumps(
            {"event": "start", "stage": stage, "chunk_id": chunk_id},
            ensure_ascii=False,
        )
    )
    llm_logger.info(
        json.dumps(
            {
                "event": "prompt",
                "stage": stage,
                "chunk_id": chunk_id,
                "chars": len(prompt or ""),
                "text": _clip(prompt),
            },
            ensure_ascii=False,
        )
    )
    llm_logger.info(
        json.dumps(
            {
                "event": "response",
                "stage": stage,
                "chunk_id": chunk_id,
                "chars": len(response or ""),
                "text": _clip(response),
            },
            ensure_ascii=False,
        )
    )
    llm_logger.info(
        json.dumps(
            {"event": "end", "stage": stage, "chunk_id": chunk_id},
            ensure_ascii=False,
        )
    )


# ── Sanity check ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    progress("demo", "halfway", done=False)
    progress("demo", "finished", done=True)
    log_llm_exchange(
        chunk_id="demo__abc",
        prompt="Line 1\nLine 2\nLine 3",
        response='{"msg": "hi\nthere"}',
        stage="demo",
    )