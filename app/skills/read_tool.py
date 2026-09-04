"""``read_skill`` 工具 — 渐进式披露的激活入口。

模型在首轮只看到 Skill 的摘要行（name + description + slug），判断某个
Skill 与任务相关时调用 ``read_skill(slug)``，拿到 ``SKILL.md`` 全文，同时
该 Skill 进入 ``activated_skills``，其 ``tool_dependencies`` 在**下一轮**
模型调用时变为可用。

"下一轮才生效"由数据流自然保证，不需要额外延迟机制：本轮的工具列表在
``wrap_model_call`` 里已经算完并发给了模型，``read_skill`` 通过
``Command(update=...)`` 写 State 之后，只有下一次 ``wrap_model_call``
才会读到新的激活集。

两处状态同时写：

- **L1 State**（``activated_skills``）：给 middleware 的 ``wrap_model_call``
  过滤工具与渲染 prompt；随 checkpoint 持久化，同 thread 的后续消息继承。
- **ContextVar**（``SkillRuntimeContext``）：给子 Agent 线程、graph 节点、
  MCP 桥接的 ``ToolRegistry.invoke`` 门控 —— 它们拿不到 State。

本工具不注册进 ``ToolRegistry``（不放 ``app/tools/``）：它是 middleware 自带
工具，只在挂了 ``SkillsMiddleware`` 的 Agent 上出现。子 Agent 不挂该
middleware，因此拿不到 read_skill，只能继承主 Agent 的激活集 —— 对齐 Yuxi
的"子智能体不可用 install_skill"。
"""
from __future__ import annotations

from typing import Annotated, Any

# langchain 在模块顶层导入（不同于项目其他模块的惰性导入约定）：
# ``from __future__ import annotations`` 把注解变成字符串，``@tool`` 装饰器
# 在构造 args_schema 时按**模块全局**解析 ``Annotated[..., InjectedState]``，
# 函数内的局部导入解析不到（NameError）。本模块只被 middleware 在构建 Agent
# 时惰性导入，因此顶层导入不会拖慢应用启动。
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.core.logger import get_logger
from app.skills.runtime import READ_SKILL_TOOL_NAME, get_active_skill_context

logger = get_logger(__name__)

_READ_SKILL_DESCRIPTION = """读取一个 Skill 的完整工作指令。

当「可用 Skill」清单中某个 Skill 与当前任务相关时调用本工具，参数传该 Skill
的 slug（清单中标注在方括号里）。读取后：
- 该 Skill 的完整指令会加入你的系统上下文，你必须遵循它；
- 它声明的工具会在下一轮对话中变为可用（本轮还不能调用）。

只读取确实相关的 Skill，不要为了"以防万一"把清单里的都读一遍。"""


def build_read_skill_tool() -> Any:
    """构造 read_skill 的 langchain 工具（带 State 注入与 Command 返回）。

    用 ``InjectedState`` 拿当前 ``activated_skills`` 做幂等判断，用
    ``Command(update=...)`` 写回 —— 这是 langgraph 里工具修改图状态的标准
    通道（对比项目其他工具的 ``fn(**kwargs) -> str`` 签名：那些工具不需要
    写状态，因此走 ToolRegistry 的统一包装）。
    """

    @tool(READ_SKILL_TOOL_NAME, description=_READ_SKILL_DESCRIPTION)
    def read_skill(
        slug: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        runtime = get_active_skill_context()
        target = (slug or "").strip()

        if not runtime.active:
            return _reply("当前请求没有启用任何 Skill，无需读取。", tool_call_id)

        definition = runtime.get(target)
        if definition is None:
            logger.info("[skills] read_skill miss: %r", target)
            return _reply(
                f"没有名为 {target!r} 的可用 Skill。"
                f"本次可用的 slug：{'、'.join(runtime.effective_slugs) or '（无）'}",
                tool_call_id,
            )

        already = target in (state.get("activated_skills") or ())
        # ContextVar 侧同步（子 Agent / graph 节点 / MCP 桥接的门控依据）
        runtime.activate(target)

        # 只提"真正因此解锁"的工具：公共工具本来就一直可用，说成"已解锁"
        # 会让模型误以为之前不能用（见 runtime.allows_tool 第 3 条）。
        unlocked = runtime.gated_tools_of(definition)
        header = f"# {definition.name}（slug: {definition.slug}）"
        footer_parts = []
        if unlocked:
            footer_parts.append(
                f"本 Skill 声明的工具已解锁，下一轮可用：{'、'.join(unlocked)}"
            )
        if definition.skill_dependencies:
            footer_parts.append(
                "关联 Skill（需要时同样用 read_skill 读取）："
                + "、".join(definition.skill_dependencies)
            )
        if already:
            footer_parts.append("（此 Skill 之前已读取过）")
        footer = "\n".join(f"> {line}" for line in footer_parts)

        body = definition.body.strip() or definition.description
        content = f"{header}\n\n{body}" + (f"\n\n{footer}" if footer else "")
        logger.info(
            "[skills] activated %s (unlocks: %s)", definition.slug, list(unlocked)
        )
        if not already:
            _emit_activation(definition, unlocked)
        return _reply(content, tool_call_id, activated=[definition.slug])

    return read_skill


def _emit_activation(definition: Any, unlocked: tuple) -> None:
    """把激活动作透出到统一事件流 + SSE 步骤回调（前端任务状态栏）。

    两条通道都走：``emit`` 进请求 trace / 事件日志（随响应返回、可落库），
    ``on_step`` 是 SSE 的实时桥（``use_task_observers`` 设置，见
    ``app/agents/deep/observe.py``）—— 二者消费者不同，缺一前端就看不到
    "模型激活了哪个 Skill"。

    best-effort：无上下文时是 no-op，异常一律吞掉 —— 可观测性不能影响
    Skill 激活这条主路径。
    """
    detail = f"已激活 Skill：{definition.name}"
    if unlocked:
        detail += f"（解锁工具：{'、'.join(unlocked)}）"

    try:
        from app.agents.events import emit

        emit(
            "skill",
            "skill_activated",
            definition.name,
            detail,
            slug=definition.slug,
            unlocked_tools=list(unlocked),
        )
    except Exception:
        pass

    try:
        from app.agents.deep.observe import get_task_observers

        observers = get_task_observers()
        if observers and observers[0]:
            observers[0]("skill_activated", detail)
    except Exception:
        pass


def _reply(
    content: str,
    tool_call_id: str,
    *,
    activated: list[str] | None = None,
) -> Any:
    """构造 Command：ToolMessage + 可选的 activated_skills 增量更新。"""
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
    }
    if activated:
        update["activated_skills"] = activated
    return Command(update=update)


__all__ = ["build_read_skill_tool"]
