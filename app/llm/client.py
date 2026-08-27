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
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Union

from openai import (
    AsyncOpenAI,
    OpenAI,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    BadRequestError,
    UnprocessableEntityError,
)

from app.core.config import get_settings
from app.core.exceptions import LLMClientError, LLMOutputParseError, LLMTimeoutError
from app.core.logger import get_logger
from app.llm.models import ChatModelProfile

logger = get_logger(__name__)


# Empty-response retry hints. deepseek-v4-flash and other reasoning models can
# spend the whole token budget on reasoning_content and return an empty content;
# appending a direct-output instruction makes the retry far more likely to work.
_EMPTY_RESPONSE_NUDGE = (
    "Note: your previous reply was empty. Please output your final answer"
    " directly, without any thinking/reasoning text or extra questions."
)
_EMPTY_JSON_NUDGE = (
    "Note: your previous reply was empty. Output ONLY a plain JSON object:"
    " no thinking text, no markdown fences, no extra words."
)


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
            # The SDK requires a string during construction. Chat endpoints
            # validate provider configuration before reaching this point.
            api_key=api_key or cfg.LLM_API_KEY or "not-configured",
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
    def _nudge_messages(
        messages: List[Dict[str, str]], hint: str
    ) -> List[Dict[str, str]]:
        """Return a copy of ``messages`` with a direct-output instruction appended."""
        nudged = list(messages)
        nudged.append({"role": "user", "content": hint})
        return nudged

    @staticmethod
    def _extract_text(response) -> str:
        message = response.choices[0].message
        text = message.content or ""
        if not text.strip():
            reasoning = getattr(message, "reasoning_content", None)
            if not reasoning:
                reasoning = (getattr(message, "model_extra", None) or {}).get(
                    "reasoning_content"
                )
            if reasoning:
                logger.warning(
                    "LLM returned empty content with %d chars of reasoning_content; "
                    "all tokens were likely spent on thinking. Consider raising "
                    "max_tokens or disabling reasoning for this call.",
                    len(reasoning),
                )
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "length" and text:
            logger.warning(
                "LLM response truncated by token limit! finish_reason=length, "
                "response_length=%d chars. Consider raising max_tokens.",
                len(text),
            )
        return text

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract JSON from LLM output that may contain markdown fences or prose."""
        # 空响应明确报错
        if not text or not text.strip():
            raise LLMOutputParseError("LLM returned empty response (0 characters)")

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
        # Fallback: extract the first balanced {…} or […] span.
        # LLM 常在 JSON 前后附加解释文字，直接找最外层括号片段可救回大多数情况。
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end > start:
                candidate = text[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
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
        prompt = list(messages)
        for attempt in range(2):
            try:
                logger.debug("LLM sync call | model=%s | msgs=%d | attempt=%d",
                             self.model, len(prompt), attempt + 1)
                resp = self._sync_client.chat.completions.create(
                    messages=prompt, **self._call_kwargs(**extra)
                )
                text = self._extract_text(resp)
                logger.debug("LLM response length=%d", len(text))
                if text.strip():
                    return text
                # 空响应：重试
                if attempt == 0:
                    logger.warning("LLM returned empty, retrying after 1.5s backoff")
                    _time.sleep(1.5)
                    # Reasoning models may burn all tokens on reasoning_content and
                    # return empty content; nudge the retry to emit content directly.
                    prompt = self._nudge_messages(prompt, _EMPTY_RESPONSE_NUDGE)
                    continue
                # 两次都空：记录详细错误信息
                logger.error(
                    "LLM returned empty after 2 attempts | model=%s | prompt_len=%d",
                    self.model, len(str(prompt))
                )
                return text  # 返回空字符串
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
        """Synchronous chat + JSON parse.

        Uses ``response_format=json_object`` when the provider supports it (falls
        back to plain text), and mirrors the async ``chat_json`` retry behavior:
        on empty/non-JSON output it appends a direct-output instruction and retries.
        """
        text = self._chat_with_json_mode_sync(messages, **extra)
        try:
            return self._parse_json(text)
        except LLMOutputParseError:
            logger.warning(
                "[llm] JSON parse failed, retrying once (head: %s)", text[:80]
            )
            text = self._chat_with_json_mode_sync(
                self._nudge_messages(messages, _EMPTY_JSON_NUDGE), **extra
            )
            return self._parse_json(text)

    def _chat_with_json_mode_sync(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> str:
        """``response_format=json_object`` on sync calls; degrade gracefully."""
        try:
            return self.chat_sync(
                messages, response_format={"type": "json_object"}, **extra
            )
        except (BadRequestError, UnprocessableEntityError):
            logger.warning(
                "[llm] response_format=json_object unsupported (sync), retrying without it"
            )
            return self.chat_sync(messages, **extra)

    # ── async interface ───────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> str:
        """Async chat completion. Returns the assistant text.

        Retries once with a direct-output instruction when the model returns an
        empty response (common for reasoning models that burn tokens on thinking).
        """
        try:
            logger.debug("LLM async call | model=%s | msgs=%d", self.model, len(messages))
            resp = await self._async_client.chat.completions.create(
                messages=messages, **self._call_kwargs(**extra)
            )
            text = self._extract_text(resp)
            if text.strip():
                return text
            logger.warning(
                "LLM async returned empty, retrying once with direct-output instruction"
            )
            resp = await self._async_client.chat.completions.create(
                messages=self._nudge_messages(messages, _EMPTY_RESPONSE_NUDGE),
                **self._call_kwargs(**extra),
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
        """Async chat + JSON parse.

        - 默认请求 ``response_format=json_object``（模型/网关不支持时自动降级重试）；
        - JSON 解析失败时整体重试一次（模型偶发输出非 JSON 内容）。
        """
        text = await self._chat_with_json_mode(messages, **extra)
        try:
            return self._parse_json(text)
        except LLMOutputParseError:
            logger.warning(
                "[llm] JSON parse failed, retrying once (head: %s)", text[:80]
            )
            text = await self._chat_with_json_mode(
                self._nudge_messages(messages, _EMPTY_JSON_NUDGE), **extra
            )
            return self._parse_json(text)

    async def _chat_with_json_mode(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ) -> str:
        """带 ``response_format=json_object`` 请求；不支持时（400/422）去掉参数重试。"""
        try:
            return await self.chat(
                messages, response_format={"type": "json_object"}, **extra
            )
        except (BadRequestError, UnprocessableEntityError):
            logger.warning(
                "[llm] response_format=json_object unsupported, retrying without it"
            )
            return await self.chat(messages, **extra)

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

    async def chat_stream_events(
        self,
        messages: List[Dict[str, str]],
        **extra,
    ):
        """流式事件生成器 — yield ``{"type": "content"|"thought", "text": ...}``。

        - ``content``: 正式回答的增量文本（与 ``chat_stream`` 一致）。
        - ``thought``: 模型的思考 token 增量（DeepSeek 类推理模型的
          ``reasoning_content``）。其他模型/网关不支持时安全降级——
          只产生 content 事件，不影响既有调用方。

        用于 SSE 端点把"生成阶段"的思维链也实时推给前端，让用户看到
        模型在最终回答前思考了什么。
        """
        try:
            logger.debug("LLM stream events | model=%s | msgs=%d", self.model, len(messages))
            stream = await self._async_client.chat.completions.create(
                messages=messages, stream=True, **self._call_kwargs(**extra)
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # DeepSeek 推理模型：思考 token 在 delta.reasoning_content；
                # openai SDK 可能把它放进 model_extra，双保险取值。
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is None:
                    reasoning = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
                if reasoning:
                    yield {"type": "thought", "text": reasoning}
                content = getattr(delta, "content", None)
                if content:
                    yield {"type": "content", "text": content}
        except APITimeoutError as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LLMClientError(f"LLM API error: {exc}") from exc


# Request-local model selection. Sync work dispatched to a thread must enter
# ``use_chat_model`` again because contextvars are not copied to executor
# threads automatically.
_active_chat_model: ContextVar[Optional[ChatModelProfile]] = ContextVar(
    "active_chat_model", default=None
)


@contextmanager
def use_chat_model(
    model: Optional[Union[str, ChatModelProfile]],
) -> Iterator[Optional[ChatModelProfile]]:
    """Apply a validated chat model selection within the current context."""
    if model is None:
        yield None
        return

    if isinstance(model, ChatModelProfile):
        profile = model
    else:
        from app.llm.models import get_chat_model_profile

        profile = get_chat_model_profile(model)
    token = _active_chat_model.set(profile)
    try:
        yield profile
    finally:
        _active_chat_model.reset(token)


def get_active_chat_model_profile() -> Optional[ChatModelProfile]:
    """Return the complete request-local model profile, if one was selected."""
    return _active_chat_model.get()


def get_active_chat_model_id() -> Optional[str]:
    """Return the model selected for the current request context, if any."""
    profile = get_active_chat_model_profile()
    return profile.id if profile else None


# Module-level singletons (lazy, per tier/profile)
_tier_clients: Dict[str, LLMClient] = {}


def evict_chat_model_clients(model_id: str) -> None:
    """Forget cached clients for a deleted or replaced custom profile."""
    prefix = f"chat:{model_id}:"
    for cache_key in [key for key in _tier_clients if key.startswith(prefix)]:
        _tier_clients.pop(cache_key, None)


def get_llm_client(
    tier: str = "main",
    model_id: Optional[str] = None,
    profile: Optional[ChatModelProfile] = None,
) -> LLMClient:
    """Return the process-level LLM client for the given tier (created once per tier).

    tier="main" — 主模型（默认），用于 ReAct 推理、答案生成等核心任务。
    tier="fast" — 快速模型，用于标题生成/意图识别/记忆提取等辅助任务。
                  未配置 LLM_FAST_* 时回退到主模型（不影响现有行为）。

    分级接口为阶段 1 引入, 本期所有调用点仍用 main; 后续成本优化时按需切 fast。
    """
    cfg = get_settings()
    selected_profile = profile or get_active_chat_model_profile()
    if selected_profile is None and model_id:
        from app.llm.models import get_chat_model_profile

        selected_profile = get_chat_model_profile(model_id)
    if selected_profile is not None:
        profile_hash = hashlib.sha256(
            "\0".join((
                selected_profile.id,
                selected_profile.base_url,
                selected_profile.model,
                selected_profile.api_key,
            )).encode("utf-8")
        ).hexdigest()[:12]
        cache_key = f"chat:{selected_profile.id}:{profile_hash}"
        if cache_key not in _tier_clients:
            _tier_clients[cache_key] = LLMClient(
                base_url=selected_profile.base_url,
                api_key=selected_profile.api_key,
                model=selected_profile.model,
                temperature=selected_profile.temperature,
            )
            logger.info(
                "[llm] chat model client created: id=%s provider=%s model=%s",
                selected_profile.id,
                selected_profile.provider,
                selected_profile.model,
            )
        return _tier_clients[cache_key]

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
