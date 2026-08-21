"""DeepAgents 集成 — LangChain ChatModel 适配。

项目自研 LLMClient 直接面向 OpenAI 兼容 HTTP API（DashScope / DeepSeek 等），
而 langchain create_react_agent 需要 langchain BaseChatModel。由于所有端点
都是 OpenAI 兼容协议，用 ``ChatOpenAI`` 指向现有配置即可（零新依赖，配置
单一来源：app/core/config.py）。

2026-08-21（S8）：DeepSeek 思考模式（reasoning）模型在响应中返回
``reasoning_content``，OpenAI 兼容协议要求多轮对话把上一轮 assistant 消息的
``reasoning_content`` 原样回传，否则报 400 ``The reasoning_content in the
thinking mode must be passed back to the API``。langchain-openai 1.4.1 的
消息转换（``_convert_dict_to_message`` / ``_convert_message_to_dict``）都会
丢弃该字段，导致 create_react_agent 多轮工具调用（调用 kb_search/web_search
之后）必然失败。``DeepSeekChatOpenAI`` 在响应侧保存、请求侧回传该字段；
对非 reasoning 模型行为与 ChatOpenAI 完全一致。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _extract_reasoning_content(response: Any) -> Optional[str]:
    """从 chat/completions 响应（dict 或 openai.BaseModel）提取 reasoning_content。"""
    try:
        if isinstance(response, dict):
            msg = response["choices"][0]["message"]
            rc = msg.get("reasoning_content")
            if rc is not None:
                return rc
            extra = msg.get("model_extra")
            return extra.get("reasoning_content") if isinstance(extra, dict) else None
        msg = response.choices[0].message
        rc = getattr(msg, "reasoning_content", None)
        if rc is not None:
            return rc
        extra = getattr(msg, "model_extra", None)
        return extra.get("reasoning_content") if isinstance(extra, dict) else None
    except Exception:
        return None


def _message_to_dict_with_reasoning(message: Any, _ob: Any) -> dict:
    """``_convert_message_to_dict`` + assistant 消息回传 reasoning_content。"""
    from langchain_core.messages import AIMessage

    if isinstance(message, AIMessage):
        rc = message.additional_kwargs.get("reasoning_content")
        converted = _ob._convert_from_v1_to_chat_completions(message)
        d = _ob._convert_message_to_dict(converted)
        if rc:
            d["reasoning_content"] = rc
        return d
    return _ob._convert_message_to_dict(message)


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI 适配子类：DeepSeek 思考模式多轮回传。

    仅覆盖 chat/completions 路径（DeepSeek 端点）；responses API 路径
    原样走基类，不受影响。对非 reasoning 模型行为与 ChatOpenAI 完全一致。
    """

    def _create_chat_result(self, response, generation_info=None):  # noqa: N802
        result = super()._create_chat_result(response, generation_info)
        rc = _extract_reasoning_content(response)
        if rc and result.generations:
            first = result.generations[0].message
            if getattr(first, "type", "") == "ai":
                first.additional_kwargs["reasoning_content"] = rc
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:  # noqa: N802
        # 复刻 BaseChatOpenAI._get_request_payload，但对 assistant 消息补回
        # additional_kwargs["reasoning_content"]（基类转换会丢弃该字段）。
        # responses API 路径直接走基类。
        from langchain_openai.chat_models import base as _ob

        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop
        payload = {**self._default_params, **kwargs}
        if self._use_responses_api(payload):
            return super()._get_request_payload(input_, stop=stop, **kwargs)
        payload["messages"] = [
            _message_to_dict_with_reasoning(m, _ob) for m in messages
        ]
        return payload


@lru_cache(maxsize=8)
def get_langchain_model(model_name: str | None = None, temperature: float | None = None):
    """返回指向项目 LLM 配置的 langchain ChatModel（进程级缓存）。

    Args:
        model_name: 覆盖模型名（默认 cfg.LLM_MODEL）。
        temperature: 覆盖温度（默认 cfg.LLM_TEMPERATURE）。

    与 ``app/llm/client.py`` 共用 base_url / api_key / model / temperature，
    保证 DeepAgents 路径与现有单 Agent / 多智能体路径使用同一模型配置。
    使用 ``DeepSeekChatOpenAI``（含 reasoning_content 多轮回传适配）。
    """
    cfg = get_settings()
    model = DeepSeekChatOpenAI(
        base_url=cfg.LLM_BASE_URL,
        api_key=cfg.LLM_API_KEY or "not-configured",
        model=model_name or cfg.LLM_MODEL,
        temperature=temperature if temperature is not None else cfg.LLM_TEMPERATURE,
        max_tokens=cfg.LLM_MAX_TOKENS,
        timeout=cfg.LLM_TIMEOUT,
        max_retries=cfg.LLM_MAX_RETRIES,
    )
    logger.debug(
        "[deepagents] langchain model: %s via %s", model.model_name, cfg.LLM_BASE_URL
    )
    return model
