"""DeepAgents 集成 — ``task`` 委派工具。

主 Agent 通过 ``task(description, subagent_type)`` 把独立子任务交给配置的
SubAgent。设计参考 DeepAgents SubAgentMiddleware / Yuxi subagent_task：

- 工具描述动态注入可用 SubAgent 名册（模型看到可选集，自动路由）
- 子 Agent 独立 state 运行（上下文隔离），返回最终结果文本
- 结果作为 ToolMessage 回到主 Agent，主 Agent 继续推理
- 未知 subagent_type / 子 Agent 异常 → 抛错回主 Agent（可自我修正）
"""
from __future__ import annotations

from typing import Any

from app.agents.deep.subagents import (
    get_subagent_config,
    get_subagents,
    run_subagent,
    subagents_prompt,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

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


def build_task_tool(model=None, recursion_limit: int = 20) -> Any:
    """构建 ``task`` StructuredTool（同步执行 SubAgent）。

    model: 测试可注入 mock（透传给子 Agent）；None = 项目真实模型。
    """
    from langchain_core.tools import StructuredTool

    def _task(description: str, subagent_type: str) -> str:
        cfg = get_subagent_config(subagent_type)
        if cfg is None:
            available = ", ".join(c.name for c in get_subagents())
            raise ValueError(
                f"未知子智能体类型 '{subagent_type}'，可用: {available}"
            )
        logger.info(
            "[deepagents] task -> subagent=%s description=%r",
            subagent_type, description[:100],
        )
        return run_subagent(cfg, description, model=model, recursion_limit=recursion_limit)

    return StructuredTool.from_function(
        func=_task,
        name="task",
        description=TASK_TOOL_DESCRIPTION + "\n可用 subagent_type:\n" + subagents_prompt(),
        infer_schema=True,
    )


def task_system_prompt() -> str:
    """主 Agent system prompt 的 task 工具说明段。"""
    return TASK_SYSTEM_PROMPT.format(subagents=subagents_prompt())
