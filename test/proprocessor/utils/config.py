# utils/config.py
"""Load .env + prompts.yaml into a single Config object.

Usage:
    from utils.config import load_config
    cfg = load_config()
    cfg.llm.api_key
    cfg.prompts["session_prompt"]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    max_concurrency: int = 50
    max_retries: int = 3
    base_backoff: float = 1.0
    timeout: float = 60.0


@dataclass
class AppConfig:
    llm: LLMConfig
    prompts: Dict[str, str]
    call_concurrency: int = 20
    pair_concurrency: int = 200
    output_dir: str = "./offline_output"
    input_path: str = "./samples.json"


def _require(name: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _load_prompts(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompts file must be a YAML mapping: {path}")
    return data


def load_config(env_path: str = ".env", prompts_path: str = None) -> AppConfig:
    """Load .env (once) and prompts.yaml into AppConfig."""
    load_dotenv(env_path, override=False)

    prompts_path = prompts_path or os.getenv("PROMPTS_PATH", "./prompts/prompts.yaml")

    llm = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=_require("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_concurrency=_get_int("LLM_MAX_CONCURRENCY", 50),
        max_retries=_get_int("LLM_MAX_RETRIES", 3),
        base_backoff=_get_float("LLM_BASE_BACKOFF", 1.0),
        timeout=_get_float("LLM_TIMEOUT", 60.0),
    )

    return AppConfig(
        llm=llm,
        prompts=_load_prompts(prompts_path),
        call_concurrency=_get_int("CALL_CONCURRENCY", 20),
        pair_concurrency=_get_int("PAIR_CONCURRENCY", 200),
        output_dir=os.getenv("OUTPUT_DIR", "./offline_output"),
        input_path=os.getenv("INPUT_PATH", "./samples.json"),
    )