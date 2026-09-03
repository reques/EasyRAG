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

skill 解析说明：路由层已把 skill_ids 解析为 SkillProfile（含 custom:* 的
owner 校验），此处按 id 回查内置目录只为了传播层需要的 profile 对象；custom
Skill 的 instructions/tool_names 已由路由层校验，传播层只需要白名单与 prompt
—— 因此回查不到的 id（custom:*）跳过即可，不报错。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence

from app.agents.context import ChatContext
from app.skills.catalog import SkillProfile


@contextmanager
def use_request_context(
    ctx: ChatContext,
    skill_profiles: Optional[Sequence[SkillProfile]] = None,
    model_profile: Optional[Any] = None,
) -> Iterator[ChatContext]:
    """进入一次 Run 的请求上下文：声明层 → 传播层一次性铺设。

    写入的传播层（均为各自模块既有实现，语义不变）：
    - ``use_chat_model(ctx.model_id)`` —— 请求级模型选择（LLM client / 审计读取）；
    - ``use_skill_context`` —— Skill 白名单与 prompt 注入（按 skill_ids 解析）；
    - ``use_authorised_kb_ids`` —— kb_search 授权边界（空元组 = 显式无授权）；
    - ``use_request_trace`` —— 请求级 trace + 事件日志（随响应返回）；
    - ``use_task_observers`` —— SSE 步骤/工件回调透传（on_step/on_artifact）。

    ``skill_profiles``：路由层已解析好的 SkillProfile（含 custom:* 的 owner
    校验）直接注入；为 None 时按 ``ctx.skill_ids`` 回查内置目录。custom:* 在
    目录中不存在，跳过即可（其 instructions/tool_names 已由路由层校验）。

    ``model_profile``：路由层已解析的 ChatModelProfile（含 custom:* 的 DB
    回查）直接注入；为 None 时按 ``ctx.model_id`` 走静态目录回查。

    model_id 为空时模型层按"未选择"进入（回退项目配置模型，与旧行为一致）。
    """
    from app.agents.deep.observe import use_task_observers
    from app.agents.events import use_request_trace
    from app.llm.client import use_chat_model
    from app.services.knowledge_context import use_authorised_kb_ids
    from app.skills.context import SkillRuntimeContext, use_skill_context

    profiles = skill_profiles if skill_profiles is not None else _resolve_skill_profiles(ctx.skill_ids)
    with (
        use_chat_model(model_profile if model_profile is not None else (ctx.model_id or None)),
        use_skill_context(SkillRuntimeContext.from_profiles(profiles)),
        use_authorised_kb_ids(list(ctx.knowledge_base_ids) or None),
        use_request_trace(session_id=ctx.thread_id),
        use_task_observers(ctx.on_step, ctx.on_artifact),
    ):
        yield ctx


def _resolve_skill_profiles(skill_ids) -> List:
    """按 id 解析 SkillProfile（内置目录；custom:* 由路由层预解析后注入）。"""
    from app.skills.catalog import get_builtin_skill

    profiles = []
    for sid in skill_ids or ():
        profile = get_builtin_skill(sid)
        if profile is not None:
            profiles.append(profile)
    return profiles


def apply_request_context(
    ctx: ChatContext,
    skill_profiles: Optional[Sequence[SkillProfile]] = None,
    model_profile: Optional[Any] = None,
) -> ChatContext:
    """把 ctx 铺进当前线程的传播层（非 with 场景：executor 线程入口）。

    路由层把同步执行分派到 executor 线程时，在线程函数体内先调用本函数，
    再执行 Agent —— 等价于 ``with use_request_context(ctx):``，只是作用域
    是整个线程生命周期、无需退出恢复。返回传入的 ctx 便于链式书写。
    """
    return use_request_context(ctx, skill_profiles, model_profile).__enter__()
