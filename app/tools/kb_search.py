"""kb_search 工具 — 企业知识库检索（供 DeepAgents 主/子 Agent 与 ReAct 调用）。

背景（2026-08-21, DeepAgents 成熟化 S1）：注册表此前没有任何知识库检索工具，
DeepAgents 路径的模型只能依赖 `_run_deep` 注入的检索上下文；本工具让模型在
推理过程中按需补充检索（含 SubAgent 委派场景）。

授权模型：工具函数签名统一为 ``fn(**kwargs) -> str``，没有渠道接收请求上下文，
因此授权知识库范围从请求级 ContextVar 读取（``app/services/knowledge_context.py``，
由 ``_run_deep`` 以 ``with use_authorised_kb_ids(...)`` 设置，同步调用链内
对主/子 Agent 均可见）。未设置授权时拒绝检索并返回提示，避免越权。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from app.core.logger import get_logger
from app.tools.registry import ToolDefinition

logger = get_logger(__name__)


def _kb_search(query: str, progress_callback: Optional[Callable[..., Any]] = None) -> str:
    """在企业知识库中检索与 query 相关的文档片段并返回（含来源标注）。

    progress_callback（阶段 5，可选）：长耗时进度上报，由 registry.invoke
    注入并同步转发统一事件流。
    """
    from app.services.knowledge_context import get_authorised_kb_ids

    kb_ids = get_authorised_kb_ids()
    if not kb_ids:
        return "（当前请求未授权任何知识库，无法执行知识库检索；可改用 web_search 检索公开信息）"
    if progress_callback:
        progress_callback("知识库检索：查询规划与召回中…", percent=20)

    try:
        from app.rag.enhanced_retriever import (
            format_blocks_for_prompt,
            format_flat_for_prompt,
            get_enhanced_retriever,
        )

        result = get_enhanced_retriever().retrieve(
            query,
            history=None,
            knowledge_base_ids=kb_ids,
        )
        if progress_callback:
            progress_callback("知识库检索：命中内容，整理结果…", percent=90)
        if result.knowledge_blocks:
            return format_blocks_for_prompt(result.knowledge_blocks)
        if result.raw_docs:
            return format_flat_for_prompt(result.raw_docs)
        return "（知识库中未找到与问题相关的内容）"
    except Exception as exc:
        logger.warning("[kb_search] retrieval failed: %s", exc)
        return f"（知识库检索失败: {exc}）"


TOOL = ToolDefinition(
    name="kb_search",
    description=(
        "在企业知识库中检索与问题相关的文档片段。适用于查询内部资料、规章制度、"
        "产品文档、已上传知识库中的内容。检索结果含来源标注，回答时优先采用并引用来源。"
    ),
    fn=_kb_search,
    arg_schema={
        "query": ("string", "检索内容（问题或关键词）", True),
    },
    # 增强检索含查询分解（多次 LLM 调用），预算放大到 120s（2026-08-26 阶段 1）
    timeout_s=120,
    # 阶段 2：工具发现元数据（适用场景关键词 + 能力标签）
    metadata={
        "scenarios": ["知识库", "内部资料", "内部文档", "规章制度", "产品文档", "文档检索", "资料"],
        "tags": ["search", "kb", "retrieval", "knowledge"],
    },
)
