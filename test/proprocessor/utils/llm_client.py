# utils/llm_client.py
"""Async LLM client with concurrency control, retries, and JSON extraction."""

import asyncio
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(Path(__file__).resolve().name)


class LLMClient:
    """Abstract LLM client. Subclass and implement `_call`."""

    def __init__(
        self,
        max_concurrency: int = 50,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        timeout: float = 60.0,
    ):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.timeout = timeout

    async def call(self, prompt: str, system: Optional[str] = None) -> str:
        """Call LLM with retry + exponential backoff."""
        async with self.semaphore:
            last_err: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    return await asyncio.wait_for(
                        self._call(prompt, system), timeout=self.timeout
                    )
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    wait = self.base_backoff * (2**attempt) + random.random()
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retry in %.2fs",
                        attempt + 1,
                        self.max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
            raise RuntimeError(
                f"LLM call failed after {self.max_retries} retries: {last_err}"
            )

    async def call_json(
        self, prompt: str, system: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info("prompt: %s", prompt)
        raw = await self.call(prompt, system)
        output = extract_json(raw)
        logger.info("response: %s", output)
        return output

    async def _call(self, prompt: str, system: Optional[str]) -> str:
        raise NotImplementedError


def extract_json(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM response."""
    text = (text or "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON from LLM response: {text[:300]}")


# ----------------------------------------------------------------------
# Built-in provider: OpenAI-compatible
# ----------------------------------------------------------------------


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible async client.

    Works with OpenAI, Azure OpenAI (with proper base_url), vLLM, TGI,
    DashScope compatible mode, Zhipu, Moonshot, DeepSeek, etc.

    Configure via .env:
        LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def _call(self, prompt: str, system: Optional[str] = None) -> str:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"LLM API error {resp.status}: {data}")
                return data["choices"][0]["message"]["content"]


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_llm_client(llm_cfg) -> LLMClient:
    """Build an LLMClient from LLMConfig.

    Add branches here for other providers (anthropic, dashscope native, etc.).
    """
    provider = (llm_cfg.provider or "openai").lower()

    if provider in (
        "openai",
        "openai-compatible",
        "vllm",
        "deepseek",
        "moonshot",
        "zhipu",
    ):
        return OpenAICompatibleClient(
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            max_concurrency=llm_cfg.max_concurrency,
            max_retries=llm_cfg.max_retries,
            base_backoff=llm_cfg.base_backoff,
            timeout=llm_cfg.timeout,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
