"""DeepAgents 集成 — LangChain ChatModel 适配。

项目自研 LLMClient 直接面向 OpenAI 兼容 HTTP API（DashScope / DeepSeek 等），
而 langchain create_agent 需要 langchain BaseChatModel。由于所有端点
都是 OpenAI 兼容协议，用 ``ChatOpenAI`` 指向现有配置即可（零新依赖，配置
单一来源：app/core/config.py）。

DeepSeek 思考模式（reasoning）模型在响应中返回 ``reasoning_content``，
OpenAI 兼容协议要求多轮对话把上一轮 assistant 消息的 ``reasoning_content``
原样回传，否则报 400 ``The reasoning_content in the thinking mode must be
passed back to the API``。

langchain-openai 1.x 官方文档明确：非标准响应字段（``reasoning_content``、
``reasoning_details``）**不会被提取或保留**，官方建议改用厂商专属包
（ChatDeepSeek 等）——但那会引入 langchain-deepseek 依赖并绑定其模型路由，
与项目"OpenAI 兼容端点 + 自有配置"的形态不符，因此保留自研适配子类。

2026-09-02（阶段 1，langchain-openai 1.x）：适配点从 0.3 的
``_convert_message_to_dict``（实例方法）迁移到 1.x 的公开结构——

- 响应侧：仍覆写 ``_create_chat_result``（1.x 中保留），从原始响应提取
  ``reasoning_content`` 存入 AIMessage.additional_kwargs；
- 请求侧：覆写 ``_get_request_payload``，先调 ``super()``（内部完成 v1
  消息 → chat/completions dict 的全部转换，含 tool_calls/audio 等新逻辑），
  再按消息位置把 additional_kwargs 里的 ``reasoning_content`` 回填到
  payload["messages"] 的 assistant 消息上。位置一一对应成立：基类对
  messages 列表逐条转换，不过滤不合并。

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

    def _get_request_payload(  # noqa: N802
        self, input_, *, stop=None, **kwargs
    ) -> dict:
        # 先让基类完成全部消息转换（v1 content blocks / tool_calls / audio），
        # 再按位置把 assistant 消息的 reasoning_content 回填进 payload。
        # responses API 路径的 payload["messages"] 结构不同，直接走基类。
        from langchain_core.messages import AIMessage

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._use_responses_api({**self._default_params, **kwargs}):
            return payload

        messages = self._convert_input(input_).to_messages()
        payload_messages = payload.get("messages") or []
        if len(payload_messages) != len(messages):
            # 位置对齐前提被破坏（理论上不应发生）→ 放弃回填，保持可请求
            logger.warning(
                "[deep llm] payload/messages length mismatch (%d vs %d), "
                "skip reasoning_content passthrough",
                len(payload_messages), len(messages),
            )
            return payload
        for msg, msg_dict in zip(messages, payload_messages):
            if not isinstance(msg, AIMessage):
                continue
            rc = msg.additional_kwargs.get("reasoning_content")
            if rc:
                msg_dict["reasoning_content"] = rc
        return payload


@lru_cache(maxsize=8)
def get_langchain_model(model_name: str | None = None, temperature: float | None = None):
    """返回指向项目 LLM 配置的 langchain ChatModel（进程级缓存）。

    Args:
        model_name: 覆盖模型名（默认 cfg.LLM_MODEL）。
        temperature: 覆盖温度（默认 cfg.LLM_TEMPERATURE）。

    与 ``app/llm/client.py`` 共用 base_url / api_key / model / temperature，
    保证 DeepAgents 路径与 dynamic 路径使用同一模型配置。
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
