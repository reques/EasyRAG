"""DeepAgents 集成 — LangChain ChatModel 适配。

项目自研 LLMClient 直接面向 OpenAI 兼容 HTTP API（DashScope / DeepSeek 等），
而 langchain create_react_agent 需要 langchain BaseChatModel。由于所有端点
都是 OpenAI 兼容协议，直接用 ``ChatOpenAI`` 指向现有配置即可，无需自定义
adapter 类（零新依赖，配置单一来源：app/core/config.py）。
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def get_langchain_model(model_name: str | None = None, temperature: float | None = None):
    """返回指向项目 LLM 配置的 langchain ChatModel（进程级缓存）。

    Args:
        model_name: 覆盖模型名（默认 cfg.LLM_MODEL）。
        temperature: 覆盖温度（默认 cfg.LLM_TEMPERATURE）。

    与 ``app/llm/client.py`` 共用 base_url / api_key / model / temperature，
    保证 DeepAgents 路径与现有单 Agent / 多智能体路径使用同一模型配置。
    """
    from langchain_openai import ChatOpenAI

    cfg = get_settings()
    model = ChatOpenAI(
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
