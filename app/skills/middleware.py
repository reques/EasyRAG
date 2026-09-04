"""SkillsMiddleware — Skill 注入与工具门控挂在 ``create_agent`` 上。

参照 Yuxi 的 ``middlewares/skills.py``。替代此前散在三处的手工拼接
（``agent_service.py`` 两处 + ``dynamic.py`` + ``graph/nodes.py``）：那些位置
各自调 ``get_active_skill_prompt()`` 拼 system message，功能与执行路径耦合，
且无法按 Agent 组装。

## 为什么用 wrap_model_call 而不是 dynamic_prompt

需要同时改两样东西：``system_message``（Skill 区块）与 ``tools``（过滤未激活
Skill 的工具）。``dynamic_prompt`` 只能改前者。``ModelRequest.override()``
是不可变模式（返回新实例），不污染原请求。

## 三个钩子的分工

| 钩子 | 职责 |
|---|---|
| ``before_agent`` | Run 开始时把 ContextVar 的激活集回灌进 L1 State（preload 场景） |
| ``wrap_model_call`` | State ↔ ContextVar 双向对齐；渲染 Skill 区块；过滤工具列表 |
| ``wrap_tool_call`` | 第一层门控：未激活 Skill 的工具直接拒绝，不进 handler |

第二层门控在 ``ToolRegistry.invoke``（ContextVar 侧），覆盖子 Agent 线程、
graph 节点与 MCP 桥接 —— 它们绕过 middleware。两层的判据是同一个
``SkillRuntimeContext.allows_tool``，因此不会出现两套真相。
"""
from __future__ import annotations

from typing import Annotated, Any, Callable, Sequence

from app.core.logger import get_logger
from app.skills.runtime import (
    READ_SKILL_TOOL_NAME,
    SkillRuntimeContext,
    get_active_skill_context,
)

logger = get_logger(__name__)


def _merge_activated(left: Sequence[str] | None, right: Sequence[str] | None) -> list[str]:
    """``activated_skills`` 的 reducer：去重并集，只增不减，保序。

    只增不减是有意的：同一 thread 的后续消息继承已激活集合（用户追问时不该
    让模型重新激活）。若要按消息重置，把状态字段换成 EphemeralValue 语义 ——
    见规划文档 §6.3。
    """
    merged = list(left or ())
    for slug in right or ():
        if slug not in merged:
            merged.append(slug)
    return merged


def _skill_aware_state_schema() -> type:
    """构造带 ``activated_skills`` 的 State schema（延迟导入 langchain）。"""
    from langchain.agents.middleware import AgentState

    class SkillAwareState(AgentState):
        activated_skills: Annotated[list[str], _merge_activated]

    return SkillAwareState


def _tool_name(tool: Any) -> str:
    """从 langchain 工具或 dict 形态的工具声明中取名字。"""
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(tool, dict):
        return str(
            tool.get("name")
            or (tool.get("function") or {}).get("name")
            or ""
        )
    return ""


def build_skills_middleware() -> Any:
    """构造 SkillsMiddleware 实例（挂到 ``create_agent(middleware=[...])``）。

    每次调用返回新实例。Agent 构建有进程级缓存（``_main_agent_cache`` 等），
    因此实例本身不能持有请求态 —— 请求态全在 ContextVar 与 State 里。
    """
    from langchain.agents.middleware import AgentMiddleware

    from app.skills.read_tool import build_read_skill_tool

    state_schema = _skill_aware_state_schema()
    read_skill_tool = build_read_skill_tool()

    class SkillsMiddleware(AgentMiddleware):
        """Skill 区块注入 + 渐进式工具门控。"""

        name = "SkillsMiddleware"

        def __init__(self) -> None:
            super().__init__()
            self.state_schema = state_schema
            self.tools = [read_skill_tool]

        # ── Run 开始：ContextVar → State（preload_skills 场景）─────────────
        def before_agent(self, state: dict, runtime: Any) -> dict | None:
            skill_rt = get_active_skill_context()
            if not skill_rt.active:
                return None
            preloaded = list(skill_rt.activated_slugs)
            if not preloaded:
                return None
            known = set(state.get("activated_skills") or ())
            fresh = [slug for slug in preloaded if slug not in known]
            if not fresh:
                return None
            logger.info("[skills] preloaded into state: %s", fresh)
            return {"activated_skills": fresh}

        # ── 每轮模型调用：prompt 注入 + 工具过滤 ──────────────────────────
        def wrap_model_call(self, request: Any, handler: Callable) -> Any:
            skill_rt = get_active_skill_context()
            if not skill_rt.active:
                # 未启用 Skill：不改 prompt、不过滤工具，但要摘掉 read_skill
                # （没有 Skill 可读，暴露它只会诱导无意义调用）
                return handler(
                    request.override(tools=_without_read_skill(request.tools))
                )

            # State ↔ ContextVar 双向对齐：State 是持久真相（跨轮次/checkpoint），
            # ContextVar 是工具线程侧真相。先回灌 State→ContextVar，再取并集。
            state_activated = (request.state or {}).get("activated_skills") or ()
            skill_rt.sync_activated(list(state_activated))

            overrides: dict[str, Any] = {
                "tools": _filter_tools(request.tools, skill_rt),
            }
            block = skill_rt.render_prompt()
            if block:
                overrides["system_message"] = _append_block(
                    request.system_message, block
                )
            return handler(request.override(**overrides))

        # ── 第一层工具门控 ────────────────────────────────────────────────
        def wrap_tool_call(self, request: Any, handler: Callable) -> Any:
            from langchain_core.messages import ToolMessage

            skill_rt = get_active_skill_context()
            name = str((request.tool_call or {}).get("name") or "")
            if not name or skill_rt.allows_tool(name):
                return handler(request)

            pending = _skill_for_tool(skill_rt, name)
            if pending:
                detail = (
                    f"工具 {name} 属于尚未读取的 Skill「{pending.name}」。"
                    f"请先调用 {READ_SKILL_TOOL_NAME}('{pending.slug}') 读取其指令，"
                    "下一轮即可使用该工具。"
                )
            else:
                detail = (
                    f"工具 {name} 不在本次请求启用的 Skill 授权范围内。"
                    f"可用 Skill：{'、'.join(skill_rt.effective_slugs) or '（无）'}"
                )
            logger.info("[skills] tool %r blocked by skill gate", name)
            return ToolMessage(
                content=detail,
                tool_call_id=str((request.tool_call or {}).get("id") or ""),
                name=name,
                status="error",
            )

    return SkillsMiddleware()


def _without_read_skill(tools: Sequence[Any] | None) -> list[Any]:
    return [t for t in (tools or ()) if _tool_name(t) != READ_SKILL_TOOL_NAME]


def _filter_tools(
    tools: Sequence[Any] | None, skill_rt: SkillRuntimeContext
) -> list[Any]:
    """按门控过滤工具列表（未激活 Skill 的工具不出现在 schema 里）。

    与 ``wrap_tool_call`` 的关系：这里让模型"看不到"，那里防"伪造工具名硬调"。
    两层都必要 —— 模型可以凭训练记忆调用未出现在 schema 中的工具名。
    """
    kept: list[Any] = []
    dropped: list[str] = []
    for tool in tools or ():
        name = _tool_name(tool)
        if not name or skill_rt.allows_tool(name):
            kept.append(tool)
        else:
            dropped.append(name)
    if dropped:
        logger.debug("[skills] tools hidden this turn: %s", dropped)
    return kept


def _skill_for_tool(
    skill_rt: SkillRuntimeContext, tool_name: str
) -> Any | None:
    """找出声明了该工具、但尚未激活的 Skill（用于给模型可执行的引导）。"""
    activated = set(skill_rt.activated_slugs)
    for skill in skill_rt.effective:
        if skill.slug not in activated and tool_name in skill.tool_dependencies:
            return skill
    return None


def _append_block(system_message: Any, block: str) -> Any:
    """把 Skill 区块追加到既有 system message 之后（保留 Agent 自身的 prompt）。"""
    from langchain_core.messages import SystemMessage

    if system_message is None:
        return SystemMessage(content=block)
    existing = getattr(system_message, "content", None)
    if isinstance(existing, str):
        return SystemMessage(content=f"{existing}\n\n{block}")
    # 多模态/结构化 content：追加为独立文本块，不破坏原结构
    if isinstance(existing, list):
        return SystemMessage(content=[*existing, {"type": "text", "text": block}])
    return SystemMessage(content=block)


__all__ = ["build_skills_middleware"]
