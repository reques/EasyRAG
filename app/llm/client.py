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

        import httpx
        client_kwargs = dict(
            base_url=base_url or cfg.LLM_BASE_URL,
            api_key=api_key or cfg.LLM_API_KEY,
            timeout=cfg.LLM_TIMEOUT,
            max_retries=cfg.LLM_MAX_RETRIES,
        )
        self._sync_client = OpenAI(
            **client_kwargs,
            http_client=httpx.Client(trust_env=False),
        )
        self._async_client = AsyncOpenAI(
            **client_kwargs,
            http_client=httpx.AsyncClient(trust_env=False),
        )

    # ── internal helpers ──────────────────────────────────────────────────

    def _call_kwargs(self, **extra) -> Dict[str, Any]:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
        )
        # temperature 只在调用方未显式传入时才用默认值
        if "temperature" not in extra:
            kwargs["temperature"] = self.temperature
        kwargs.update(extra)
        return kwargs

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
        """Synchronous chat completion. Returns the assistant text.

        Auto-retries once with backoff on empty response (API 偶发空返回).
        """
        import time as _time
        for attempt in range(2):
            try:
                logger.debug("LLM sync call | model=%s | msgs=%d | attempt=%d",
                             self.model, len(messages), attempt + 1)
                resp = self._sync_client.chat.completions.create(
                    messages=messages, **self._call_kwargs(**extra)
                )
                text = self._extract_text(resp)
                logger.debug("LLM response length=%d", len(text))
                if text.strip():
                    return text
                # 空响应：重试
                if attempt == 0:
                    logger.warning("LLM returned empty, retrying after 1.5s backoff")
                    _time.sleep(1.5)
                    continue
                return text  # 两次都空，返回空字符串
            except APITimeoutError as exc:
                raise LLMTimeoutError("LLM request timed out") from exc
            except (RateLimitError, APIConnectionError) as exc:
                raise LLMClientError(f"LLM API error: {exc}") from exc
        return ""  # unreachable, placate type checkers

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

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ):
        """Async streaming chat — 逐 token yield 增量文本。

        用于 SSE 端点把 LLM 输出实时推给前端, 避免一次性等待完整回复。
        用法::

            async for delta in client.chat_stream(messages):
                ...  # delta 为本次新增的文本片段
        """
        try:
            logger.debug("LLM stream call | model=%s | msgs=%d", self.model, len(messages))
            stream = await self._async_client.chat.completions.create(
                messages=messages, stream=True, **self._call_kwargs(**extra)
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LLMClientError(f"LLM API error: {exc}") from exc


# Module-level singletons (lazy, per-tier)
_tier_clients: Dict[str, LLMClient] = {}


def get_llm_client(tier: str = "main") -> LLMClient:
    """Return the process-level LLM client for the given tier (created once per tier).

    tier="main" — 主模型（默认），用于 ReAct 推理、答案生成等核心任务。
    tier="fast" — 快速模型，用于标题生成/意图识别/记忆提取等辅助任务。
                  未配置 LLM_FAST_* 时回退到主模型（不影响现有行为）。

    分级接口为阶段 1 引入, 本期所有调用点仍用 main; 后续成本优化时按需切 fast。
    """
    cfg = get_settings()
    if tier == "fast" and not cfg.LLM_FAST_MODEL:
        tier = "main"  # fast 未配置 → 回退主模型
    if tier not in _tier_clients:
        if tier == "fast":
            _tier_clients[tier] = LLMClient(
                base_url=cfg.LLM_FAST_BASE_URL or cfg.LLM_BASE_URL,
                api_key=cfg.LLM_FAST_API_KEY or cfg.LLM_API_KEY,
                model=cfg.LLM_FAST_MODEL,
            )
            logger.info("[llm] fast tier client created: model=%s", cfg.LLM_FAST_MODEL)
        else:
            _tier_clients[tier] = LLMClient()
    return _tier_clients[tier]
