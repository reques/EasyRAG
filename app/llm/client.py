"""OpenAI-compatible LLM client wrapper.

Supports any provider with an OpenAI-compatible /v1 endpoint:
  DeepSeek / OpenAI / Qwen / GLM / Moonshot / LM-Studio / Ollama

Usage::

    client = LLMClient()
    # async
    response = await client.chat([{"role": "user", "content": "Hello"}])
    # sync
    response = client.chat_sync([{"role": "user", "content": "Hello"}])
    # JSON extraction
    data = client.chat_json_sync([{"role": "user", "content": "Return JSON: {\"key\": \"value\"}"}])
"""
import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI, APITimeoutError, RateLimitError, APIConnectionError

from app.core.config import get_settings
from app.core.exceptions import LLMClientError, LLMOutputParseError, LLMTimeoutError
from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Thin wrapper around the OpenAI SDK for synchronous and async calls."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        cfg = get_settings()
        self.model = model or cfg.LLM_MODEL
        self.temperature = temperature if temperature is not None else cfg.LLM_TEMPERATURE
        self.max_tokens = max_tokens or cfg.LLM_MAX_TOKENS

        client_kwargs = dict(
            base_url=base_url or cfg.LLM_BASE_URL,
            api_key=api_key or cfg.LLM_API_KEY,
            timeout=cfg.LLM_TIMEOUT,
            max_retries=cfg.LLM_MAX_RETRIES,
        )
        self._sync_client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    # ── internal helpers ──────────────────────────────────────────────────

    def _call_kwargs(self, **extra) -> Dict[str, Any]:
        return dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra,
        )

    @staticmethod
    def _extract_text(response) -> str:
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract JSON from LLM output that may contain markdown fences."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Strip markdown code fences
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text, re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        raise LLMOutputParseError(
            f"Failed to parse JSON from LLM output: {text[:200]}"
        )

    # ── sync interface ────────────────────────────────────────────────────

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> str:
        """Synchronous chat completion. Returns the assistant text."""
        try:
            logger.debug("LLM sync call | model=%s | msgs=%d", self.model, len(messages))
            resp = self._sync_client.chat.completions.create(
                messages=messages, **self._call_kwargs(**extra)
            )
            text = self._extract_text(resp)
            logger.debug("LLM response length=%d", len(text))
            return text
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LLMClientError(f"LLM API error: {exc}") from exc

    def chat_json_sync(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> Any:
        """Synchronous chat + JSON parse."""
        text = self.chat_sync(messages, **extra)
        return self._parse_json(text)

    # ── async interface ───────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> str:
        """Async chat completion. Returns the assistant text."""
        try:
            logger.debug("LLM async call | model=%s | msgs=%d", self.model, len(messages))
            resp = await self._async_client.chat.completions.create(
                messages=messages, **self._call_kwargs(**extra)
            )
            return self._extract_text(resp)
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LLMClientError(f"LLM API error: {exc}") from exc

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> Any:
        """Async chat + JSON parse."""
        text = await self.chat(messages, **extra)
        return self._parse_json(text)


# Module-level singleton (lazy)
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the process-level default LLM client (created once)."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
