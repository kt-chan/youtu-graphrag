# utils/logger.py
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "offline_preprocess",
    level: int = logging.INFO,
    log_file: str = "./test/proprocessor/logs/",
) -> logging.Logger:
    """Configure and return a logger that writes to console and file. Idempotent."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        # File handler
        log_path = Path(log_file) / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        logger.propagate = False
    return logger