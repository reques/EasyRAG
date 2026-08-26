"""DeepAgents 集成 — ``task`` 委派工具。

主 Agent 通过 ``task(description, subagent_type)`` 把独立子任务交给配置的
SubAgent。设计参考 DeepAgents SubAgentMiddleware / Yuxi subagent_task：

- 工具描述动态注入可用 SubAgent 名册（模型看到可选集，自动路由）
- 子 Agent 独立 state 运行（上下文隔离），返回最终结果文本
- 结果作为 ToolMessage 回到主 Agent，主 Agent 继续推理
- 未知 subagent_type → 抛错回主 Agent；子 Agent 异常 → 返回错误消息回主 Agent（可自我修正）
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from app.agents.deep.subagents import (
    get_subagent_config,
    get_subagents,
    run_subagent,
    subagents_prompt,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── S5 委派熔断（2026-08-26，阶段 1）────────────────────────────────────
# 按 (session_id, subagent_type) 记录连续失败：达到阈值后拒绝再委派该子智能体
# （返回提示让主 Agent 自行处理，避免死循环式反复委派）；成功即清零；条目带
# TTL，长期无活动自动过期，避免旧故障永久封禁。session_id 取自请求级 trace
# （无 trace 上下文时退化为进程级单键——测试/脚本场景）。
TASK_FAIL_LIMIT = 3
TASK_BREAKER_TTL_S = 600.0
_task_failures: Dict[Tuple[str, str], Dict[str, float]] = {}


def reset_task_breaker() -> None:
    """清空熔断状态（测试用）。"""
    _task_failures.clear()


def _breaker_session() -> str:
    from app.agents.events import get_trace

    trace = get_trace()
    return trace.session_id if trace else ""


def _breaker_check(subagent_type: str) -> Optional[str]:
    """熔断检查：连续失败达到阈值时返回给主 Agent 的提示文本，否则 None。"""
    now = time.time()
    key = (_breaker_session(), subagent_type)
    entry = _task_failures.get(key)
    if not entry:
        return None
    if now - entry["last_ts"] > TASK_BREAKER_TTL_S:
        _task_failures.pop(key, None)
        return None
    if entry["fails"] >= TASK_FAIL_LIMIT:
        return (
            f"子智能体 '{subagent_type}' 近期已连续失败 {entry['fails']} 次，"
            "委派已被熔断；请基于已有信息自行完成任务，或换用其他子智能体。"
        )
    return None


def _breaker_record(subagent_type: str, ok: bool) -> None:
    """记录一次委派结果：成功清零，失败累加（窗口外重计）。"""
    key = (_breaker_session(), subagent_type)
    now = time.time()
    if ok:
        _task_failures.pop(key, None)
        return
    entry = _task_failures.get(key)
    if entry and now - entry["last_ts"] <= TASK_BREAKER_TTL_S:
        entry["fails"] += 1
    else:
        entry = {"fails": 1}
    entry["last_ts"] = now
    _task_failures[key] = entry

TASK_SYSTEM_PROMPT = """## `task`（子智能体委派工具）

你可以使用 `task` 工具把复杂、独立、需要隔离上下文的子任务交给已配置的子智能体。
子智能体只返回最终结果，你看不到它的中间步骤。

使用原则：
- 任务足够复杂、可独立完成、或需要隔离上下文时使用。
- 简单问题或少量直接工具调用不要委派，直接自己处理。
- 调用时选择下方可用的 `subagent_type`，并在 `description` 中写清目标、背景和期望输出。
- 不要通过 shell、curl 或命令行间接调用子智能体；需要子智能体时必须使用 `task` 工具。

可用子智能体：
{subagents}
"""

TASK_TOOL_DESCRIPTION = """把独立子任务委派给已配置的子智能体执行，返回其最终结果。

参数：
- description: 任务描述，写清目标、背景上下文和期望输出（必填）。
- subagent_type: 子智能体名称，必须是下方可用的之一。
"""


def build_task_tool(model=None, recursion_limit: Optional[int] = None) -> Any:
    """构建 ``task`` StructuredTool（同步执行 SubAgent）。

    model: 测试可注入 mock（透传给子 Agent）；None = 项目真实模型。
    recursion_limit: 子 Agent 的 LangGraph recursion_limit（None =
    DEEP_SUBAGENT_RECURSION_LIMIT 配置）。
    """
    from langchain_core.tools import StructuredTool
    from app.core.config import get_settings

    if recursion_limit is None:
        recursion_limit = get_settings().DEEP_SUBAGENT_RECURSION_LIMIT

    def _task(description: str, subagent_type: str) -> str:
        cfg = get_subagent_config(subagent_type)
        if cfg is None:
            available = ", ".join(c.name for c in get_subagents())
            raise ValueError(
                f"未知子智能体类型 '{subagent_type}'，可用: {available}"
            )
        # S5 熔断：连续失败的子智能体直接拒绝委派
        tripped = _breaker_check(subagent_type)
        if tripped:
            logger.warning(
                "[deepagents] task -> subagent=%s circuit OPEN, rejecting",
                subagent_type,
            )
            return tripped
        logger.info(
            "[deepagents] task -> subagent=%s description=%r",
            subagent_type, description[:100],
        )
        # 统一事件流：委派事件携带 trace（无 trace 上下文时 no-op）
        from app.agents.events import emit, use_span
        from app.observability.tracing import trace_span

        emit("delegation", "task_start", f"委派 {subagent_type}", description[:200],
             task_key=subagent_type, subagent_type=subagent_type)
        try:
            # S3 步骤透传：主 Agent 的 SSE 回调（use_task_observers 设置）
            # → 子 Agent 观察者（run_subagent 的 stream 循环读取）
            from app.agents.deep.observe import get_task_observers, use_subagent_observers

            # use_span：子 Agent 层事件以 subagent/<name> 标识（同 trace）
            # trace_span：阶段 5 遥测（OTel 未安装时 no-op）
            with use_span(f"subagent/{subagent_type}"), trace_span(
                f"subagent.{subagent_type}", subagent=subagent_type
            ):
                on_step, on_artifact = get_task_observers() or (None, None)
                if on_step is None and on_artifact is None:
                    result = run_subagent(
                        cfg, description, model=model, recursion_limit=recursion_limit
                    )
                else:
                    with use_subagent_observers(on_step, on_artifact):
                        result = run_subagent(
                            cfg, description, model=model, recursion_limit=recursion_limit
                        )
            _breaker_record(subagent_type, True)
            emit("delegation", "task_end", f"{subagent_type} 完成", str(result)[:200],
                 task_key=subagent_type, subagent_type=subagent_type)
            return result
        except Exception as e:
            _breaker_record(subagent_type, False)
            logger.warning("[deepagents] task -> subagent=%s failed: %s", subagent_type, e)
            emit("delegation", "task_error", f"{subagent_type} 失败", str(e)[:200],
                 task_key=subagent_type, subagent_type=subagent_type)
            return f"子智能体执行失败: {e}"

    return StructuredTool.from_function(
        func=_task,
        name="task",
        description=TASK_TOOL_DESCRIPTION + "\n可用 subagent_type:\n" + subagents_prompt(),
        infer_schema=True,
    )


def task_system_prompt() -> str:
    """主 Agent system prompt 的 task 工具说明段。"""
    return TASK_SYSTEM_PROMPT.format(subagents=subagents_prompt())
