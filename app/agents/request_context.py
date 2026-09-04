"""请求上下文门面 — 把声明式 ChatContext 写入全部传播层 ContextVar。

阶段 2（Yuxi 参照重构，见 ``docs/plans/2026-09-02-agentstate-refactor-yuxi.md``
§2.2）：``use_chat_model`` / ``use_skill_context`` / ``use_authorised_kb_ids`` /
``use_request_trace`` / ``use_task_observers`` 各自独立存在（传播层 —— 工具
签名统一为 ``fn(**kwargs) -> str``，工具线程只能从 ContextVar 读取），但路由
层不再逐个进入 —— 本模块提供唯一入口 ``use_request_context(ctx)``，一次进入、
五层齐设、退出全恢复。

子 Agent / spawn_tasks 的 ThreadPoolExecutor 分派继续用
``events.snapshot_request_context()`` + ``run_with_request_context()``
（contextvars 快照重放），无需感知本门面。

skill 解析说明（2026-09-04 Skill 重构）：路由层已把 slug 解析为
``SkillDefinition`` 有效集合（含个人 Skill 的 owner 校验与
``skill_dependencies`` 闭包展开），此处直接注入。为 None 时按
``ctx.skill_ids`` 回查**内置目录**——个人 Skill 需要 owner 上下文与数据库
索引，回查不到即跳过（这条路径只用于测试与内部调用）。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence

from app.agents.context import ChatContext
from app.skills.loader import SkillDefinition


@contextmanager
def use_request_context(
    ctx: ChatContext,
    skill_definitions: Optional[Sequence[SkillDefinition]] = None,
    model_profile: Optional[Any] = None,
) -> Iterator[ChatContext]:
    """进入一次 Run 的请求上下文：声明层 → 传播层一次性铺设。

    写入的传播层（均为各自模块既有实现，语义不变）：
    - ``use_chat_model(ctx.model_id)`` —— 请求级模型选择（LLM client / 审计读取）；
    - ``use_skill_context`` —— Skill 有效集合与激活状态（渐进式披露门控依据）；
    - ``use_authorised_kb_ids`` —— kb_search 授权边界（空元组 = 显式无授权）；
    - ``use_request_trace`` —— 请求级 trace + 事件日志（随响应返回）；
    - ``use_task_observers`` —— SSE 步骤/工件回调透传（on_step/on_artifact）。

    ``skill_definitions``：路由层已解析好的有效集合（``SkillDefinition``）直接
    注入；为 None 时按 ``ctx.skill_ids`` 回查内置目录。

    ``preload_skill_slugs``（``ctx``）：对齐 Yuxi ``preload_skills`` —— 这些
    Skill 在 Run 开始时就进激活集（首轮展开正文与工具），不走渐进式披露。

    ``model_profile``：路由层已解析的 ChatModelProfile（含 custom:* 的 DB
    回查）直接注入；为 None 时按 ``ctx.model_id`` 走静态目录回查。

    model_id 为空时模型层按"未选择"进入（回退项目配置模型，与旧行为一致）。
    """
    from app.agents.deep.observe import use_task_observers
    from app.agents.events import use_request_trace
    from app.llm.client import use_chat_model
    from app.services.knowledge_context import use_authorised_kb_ids
    from app.skills.runtime import SkillRuntimeContext, use_skill_context

    definitions = (
        skill_definitions
        if skill_definitions is not None
        else _resolve_skill_definitions(ctx.skill_ids)
    )
    skill_runtime = SkillRuntimeContext.from_definitions(definitions)
    for slug in getattr(ctx, "preload_skill_slugs", ()) or ():
        skill_runtime.activate(slug)

    with (
        use_chat_model(model_profile if model_profile is not None else (ctx.model_id or None)),
        use_skill_context(skill_runtime),
        use_authorised_kb_ids(list(ctx.knowledge_base_ids) or None),
        use_request_trace(session_id=ctx.thread_id),
        use_task_observers(ctx.on_step, ctx.on_artifact),
    ):
        yield ctx


def _resolve_skill_definitions(skill_ids) -> List[SkillDefinition]:
    """按 slug 解析内置 Skill（个人 Skill 由路由层预解析后注入）。"""
    from app.skills.registry import get_skill

    definitions: List[SkillDefinition] = []
    for slug in skill_ids or ():
        definition = get_skill(slug)
        if definition is not None:
            definitions.append(definition)
    return definitions


def apply_request_context(
    ctx: ChatContext,
    skill_definitions: Optional[Sequence[SkillDefinition]] = None,
    model_profile: Optional[Any] = None,
) -> ChatContext:
    """把 ctx 铺进当前线程的传播层（非 with 场景：executor 线程入口）。

    路由层把同步执行分派到 executor 线程时，在线程函数体内先调用本函数，
    再执行 Agent —— 等价于 ``with use_request_context(ctx):``，只是作用域
    是整个线程生命周期、无需退出恢复。返回传入的 ctx 便于链式书写。
    """
    return use_request_context(ctx, skill_definitions, model_profile).__enter__()
