"""Run 级上下文装配 —— deep / dynamic 共用的消息链前缀。

**存在理由（2026-10 收口）**：deep 与 dynamic 两条路径各自重写了一遍
"知识库目录 → 用户事实/经历 → 本轮提问"的装配，此前只能靠注释约定对齐
（``_run_deep_inner`` 写着"复用 prepare_context 的注入链，保证行为一致"，
``dynamic.py`` 写着"对齐 prepare_context / _run_deep 的注入链"）。
"靠注释保证一致"正是历史事故的来源：``_run_deep`` 曾经**从不检索知识库**，
而系统提示却声称"检索结果会作为上下文提供"，知识库问答静默退化成纯 LLM 生成
（见 ``agent_service.py`` S1 注释）。

本模块把公共装配抽成一处实现。各路径只负责自己**特有**的部分：
- deep：生成前的知识库前置检索（``context:retrieval`` 系统消息）；
- dynamic：不前置检索，检索交给模型自行调用 ``kb_search`` 工具。

Skill 区块**不在**这里注入 —— 由 ``SkillsMiddleware`` 每轮按
``activated_skills`` 动态渲染（渐进式披露），手工拼接会与之重复。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

#: 本轮无知识库目录时的占位文案（与历史实现一致）。
EMPTY_CATALOG_PLACEHOLDER = "当前会话没有已授权的知识库。"


@dataclass
class AssembledRunContext:
    """公共消息链的装配结果。"""

    messages: List[Any]
    """本轮输入消息（知识库目录 → 记忆 → 由调用方追加本轮提问）。"""

    history: List[Dict[str, Any]]
    """规范化后的对话历史（供 checkpoint 轮次 ID 与后续检索使用）。"""

    has_checkpoint: bool
    """本轮是否存在可恢复的未完成 checkpoint（决定替代历史与占位注入）。"""


def resolve_working_history(
    context: Any,
    *,
    session_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """解析本次 Run 使用的原始历史（session 兜底仅在有兜底来源时生效）。"""
    history = list(context.history)
    if not history and session_history:
        history = list(session_history)
    return history


def assemble_base_run_context(
    *,
    query: str,
    context: Any,
    checkpoint_kind: str,
    session_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> AssembledRunContext:
    """装配 deep / dynamic 共用的消息前缀。

    顺序（两个调用方都依赖该顺序，勿单独调整）：

    1. ``context:knowledge-catalog`` —— 已授权知识库目录（有目录，或恢复轮次）；
    2. 用户事实 / 相关经历 / 历史 —— ``assemble_memory_context`` 的产物；
    3. （由调用方追加）生成前检索块（仅 deep 路径），随后是本轮提问。

    ``checkpoint_kind``：checkpoint 命名空间（``"deep"`` / ``"dynamic"``）。
    必须由调用方显式给出 —— 同一个 context 可能被不同路径执行，不能靠
    ``context`` 自身推断，否则恢复语义会串道。

    ``session_history``：history 为空时的兜底来源（AgentService 内存
    SessionStore）。deep 路径依赖它；dynamic 路径历史由调用方预先解析。

    知识库前置检索（deep 特有）**不在这里**：它需要在检索过程中实时透出
    ``on_step`` / ``on_artifact`` 进度，因此留在调用方，插入位置由调用方用
    ``len(messages)`` 取证（见 ``_run_deep_inner``）。
    """
    from app.memory.context import assemble_memory_context

    session_id = context.thread_id
    history = resolve_working_history(context, session_history=session_history)

    memory_context = assemble_memory_context(
        history,
        context.user_id,
        query=query,
        exclude_trailing_user=True,
    )
    history = [dict(item) for item in memory_context.history]

    # checkpoint 判定必须在装配消息之前：它决定是否回放历史、是否注入占位。
    from app.agents.checkpointing import begin_checkpoint_run

    has_checkpoint = begin_checkpoint_run(session_id, checkpoint_kind)

    messages: List[Any] = []
    if context.knowledge_catalog or has_checkpoint:
        from app.services.knowledge_catalog import format_knowledge_catalog

        catalog = list(context.knowledge_catalog)
        messages.append({
            "role": "system",
            "id": "context:knowledge-catalog",
            "content": (
                format_knowledge_catalog(catalog)
                if catalog
                else EMPTY_CATALOG_PLACEHOLDER
            ),
        })

    messages.extend(
        memory_context.as_dict_messages(
            include_history=not has_checkpoint,
            replace_empty_facts=has_checkpoint,
        )
    )

    return AssembledRunContext(
        messages=messages,
        history=history,
        has_checkpoint=has_checkpoint,
    )


def to_langchain_messages(messages: Sequence[Any]) -> List[Any]:
    """把 dict 形态的装配结果转成 LangChain 消息（保留 role 与 id）。

    已有 LangChain 消息原样透传，便于调用方混合追加自己的消息。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    converted: List[Any] = []
    for item in messages:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        role = item.get("role")
        content = item.get("content", "")
        message_id = item.get("id")
        if role == "system":
            converted.append(SystemMessage(content=content, id=message_id))
        elif role == "user":
            converted.append(HumanMessage(content=content, id=message_id))
        elif role == "tool":
            # 持久化历史没有可靠的 tool_call_id，作为明确标注的系统上下文恢复。
            converted.append(SystemMessage(
                content=f"[历史工具结果]\n{content}", id=message_id
            ))
        else:
            converted.append(AIMessage(content=content, id=message_id))
    return converted


__all__ = [
    "EMPTY_CATALOG_PLACEHOLDER",
    "AssembledRunContext",
    "assemble_base_run_context",
    "resolve_working_history",
    "to_langchain_messages",
]
