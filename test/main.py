#!/usr/bin/env python3
"""
Compile schema.json into a Pydantic v2 graph-ontology module via an LLM.

Reads:
  - .env           (API key, base URL, model)
  - schema.json    (the declarative ontology)
  - prompt below   (fixed template; only {{SCHEMA_JSON}} is substituted)

Writes:
  - app/backend/model/<domain_snake>.py

Exit codes:
  0  success
  1  configuration / IO error
  2  LLM returned no python code block
  3  generated module failed to parse
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import yaml

# --------------------------------------------------------------------------- #
# Paths: test/schema.json
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
SCHEMA_PATH = ROOT / "schema.json"
OUTPUT_DIR = ROOT / "model"
PROMPT_PATH = ROOT / "prompt.yaml"

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def load_config() -> dict:
    if not ENV_PATH.exists():
        sys.exit(f"[config] missing {ENV_PATH}")
    load_dotenv(ENV_PATH)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key:
        sys.exit("[config] LLM_API_KEY is not set in .env")

    return {
        "api_key": api_key,
        "base_url": base_url or None,
        "model": model or None,
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "8000")),
    }


def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        sys.exit(f"[schema] missing {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_prompt() -> dict[str, str]:
    """Return {'system': ..., 'user': ...} from prompt.yaml."""
    if not PROMPT_PATH.exists():
        sys.exit(f"[prompt] missing {PROMPT_PATH}")
    with PROMPT_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "user" not in data:
        sys.exit("[prompt] prompt.yaml must define a top-level `user:` key")

    return {
        "system": data.get("system", "You are a helpful assistant."),
        "user": data["user"],
    }
    
def build_messages(prompt: dict[str, str], schema: dict) -> list[dict]:
    user = prompt["user"].replace(
        "{{SCHEMA_JSON}}",
        json.dumps(schema, indent=2, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": user},
    ]


def call_llm(cfg: dict, messages: list[dict]) -> str:
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def extract_code(raw: str) -> str:
    m = CODE_BLOCK_RE.search(raw)
    if not m:
        print("[llm] raw response:\n" + raw, file=sys.stderr)
        sys.exit(2)
    return m.group(1).rstrip() + "\n"


def syntax_check(code: str) -> None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        print(f"[check] generated module has a syntax error: {exc}", file=sys.stderr)
        sys.exit(3)


def write_output(domain: str, code: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{domain}.py"
    out.write_text(code, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    domain = os.getenv("DOMAIN_SNAKE", "debt_collection")
    # `DOMAIN_SNAKE` can also come from .env; load_config() already did that.
    load_dotenv(ENV_PATH, override=False)
    domain = os.getenv("DOMAIN_SNAKE", domain)

    cfg = load_config()
    schema = load_schema()
    prompt = load_prompt()
    messages = build_messages(prompt, schema)
    
    print(f"[llm] model={cfg['model']} base_url={cfg['base_url'] or 'default'}")
    print(f"[llm] system chars = {len(prompt['system'])}")
    print(f"[llm] user   chars = {len(messages[1]['content'])}")

    raw = call_llm(cfg, messages)
    print(f"[llm] response chars = {len(raw)}")

    code = extract_code(raw)
    syntax_check(code)

    out = write_output(domain, code)
    print(f"[ok] wrote {out}  ({len(code)} bytes)")

if __name__ == "__main__":
    main()
