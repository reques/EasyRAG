"""DeepAgents 集成 — 主 Agent 构建。

主 Agent = langgraph ``create_react_agent``：
- 项目全量工具（ToolRegistry → StructuredTool，含技能白名单）
- ``task`` 委派工具（→ 配置化 SubAgent）
- ``spawn_tasks`` 批量委派工具（阶段 3：DAG 依赖 + 分层并发）
- ``revise_plan`` 计划修订工具（阶段 4：追加/取消/细化重发）
- system prompt 注入：任务工具说明 + 可用子智能体名册

架构对照 DeepAgents ``create_deep_agent``（底层同为 create_react_agent +
工具集 + 委派机制），但因项目钉死 langchain 0.3.26 / langgraph 0.6.x
（deepagents 官方包要求 langchain>=1.0），此处自组装等价能力。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agents.deep.subagents import subagents_prompt
from app.agents.deep.task_tool import build_task_tool, task_system_prompt
from app.agents.deep.tools import registry_to_langchain_tools
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

MAIN_SYSTEM_PROMPT = """你是一名智能助理，运行在 EasyRAG 企业知识库问答平台上。

你可以：
1. 直接回答用户问题（知识库检索结果会作为上下文提供）。
2. 调用可用工具获取实时信息（搜索/计算等）。
3. 把复杂、独立的子任务通过 `task` 工具委派给子智能体，并在其结果基础上继续推理。
4. 复杂多域任务：先规划拆解，用 `spawn_tasks` 一次声明多个子任务及其依赖，
   调度器会按依赖层级并发执行并把上游产出注入下游任务。
5. 执行过程中用 `revise_plan` 动态追加、取消或细化子任务，直至信息充分。

{subagents_section}

复杂长链路任务（多步骤 / 跨领域 / 需要持续迭代）：
- 先规划再执行：把目标拆成可并行或有依赖的子任务，为每个子任务写明目标、所需上下文
  与「交付物」（期望输出的最终形态，如结论、清单、代码、报告段落）；
- 用 `spawn_tasks` 一次声明全部子任务与依赖（depends_on），调度器会按依赖层级
  并发执行，并把上游交付物注入下游任务；
- 每轮执行后检查各子任务的交付物与遗留关注（concerns / suggested_followup），
  需要补信息时用 `revise_plan` 精准追加，不要无脑重跑或无限循环；
- 收尾时必须把各子任务交付物汇总成一份完整、自洽、可直接使用的最终回答，
  标注关键来源；若某子任务失败，明确说明影响与替代结论。

工作原则：
- 简单问题直接回答；需要外部信息时先调工具；任务复杂且可独立时用 `task` 委派。
- 多个相互关联的子任务（有先后依赖、需并行）时用 `spawn_tasks` 表达依赖；
  单一子任务用 `task`。简单委派不要过度拆解。
- 用户能实时看到你的思考、工具调用与子任务交付物：让每一步动作都有明确目的，
  避免重复解释或低价值中间评论，把思考直接用于选择下一步动作。
- 委派后基于子智能体返回的结果继续推理，给出最终完整回答。
- 回答使用中文，引用来源时注明。
"""


# 生产路径缓存（model=None 时缓存；测试注入 mock 绕过缓存）
_main_agent_cache: Optional[Any] = None


def build_main_agent(
    model=None,
    subagent_model=None,
    recursion_limit: Optional[int] = None,
):
    """构建主 Agent compiled graph。

    model: 测试可注入 mock；subagent_model: task 委派的子 Agent 模型
    （测试隔离用；None 回退到 model/项目配置）。注意：任务委派工具的
    recursion_limit 在构建时绑定（None = DEEP_SUBAGENT_RECURSION_LIMIT）；
    主 Agent 自身 recursion_limit 由调用方 invoke 时传入。
    """
    from langgraph.prebuilt import create_react_agent

    from app.agents.deep.llm import get_langchain_model
    from app.agents.deep.planner import build_revise_plan_tool, build_spawn_tasks_tool

    global _main_agent_cache
    cacheable = model is None  # 修复（2026-08-26 阶段 3）：model 随后会被重赋值，
    # 旧版 `if model is None:` 存入分支恒不成立——主 Agent 缓存从不生效
    if cacheable and _main_agent_cache is not None:
        return _main_agent_cache
    if model is None:
        model = get_langchain_model()
    if recursion_limit is None:
        recursion_limit = get_settings().DEEP_SUBAGENT_RECURSION_LIMIT
    tools = registry_to_langchain_tools()  # 全量（技能白名单生效）
    tools.append(build_task_tool(model=subagent_model or model, recursion_limit=recursion_limit))
    tools.append(build_spawn_tasks_tool(model=subagent_model or model))
    tools.append(build_revise_plan_tool(model=subagent_model or model))
    prompt = MAIN_SYSTEM_PROMPT.format(subagents_section=task_system_prompt())
    logger.info(
        "[deepagents] main agent built: %d tools (incl. task, spawn_tasks, revise_plan)",
        len(tools),
    )
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name="easyrag_deep_agent",
    )
    if cacheable:
        _main_agent_cache = agent
    return agent


def get_main_agent():
    """进程级单例主 Agent。"""
    return build_main_agent()


def get_agent_tool_names() -> List[str]:
    """主 Agent 可用工具名（含 task），供状态展示/日志。"""
    from app.agents.deep.subagents import get_subagents

    names = [t.name for t in registry_to_langchain_tools()]
    names.append("task")
    names.append("spawn_tasks")
    names.append("revise_plan")
    return names
