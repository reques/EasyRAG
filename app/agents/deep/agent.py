"""DeepAgents 集成 — 主 Agent 构建。

主 Agent = langgraph ``create_react_agent``：
- 项目全量工具（ToolRegistry → StructuredTool，含技能白名单）
- ``task`` 委派工具（→ 配置化 SubAgent）
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

{subagents_section}

工作原则：
- 简单问题直接回答；需要外部信息时先调工具；任务复杂且可独立时用 `task` 委派。
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

    global _main_agent_cache
    if model is None and _main_agent_cache is not None:
        return _main_agent_cache
    if model is None:
        model = get_langchain_model()
    if recursion_limit is None:
        recursion_limit = get_settings().DEEP_SUBAGENT_RECURSION_LIMIT
    tools = registry_to_langchain_tools()  # 全量（技能白名单生效）
    tools.append(build_task_tool(model=subagent_model or model, recursion_limit=recursion_limit))
    prompt = MAIN_SYSTEM_PROMPT.format(subagents_section=task_system_prompt())
    logger.info("[deepagents] main agent built: %d tools (incl. task)", len(tools))
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name="easyrag_deep_agent",
    )
    if model is None:
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
    return names
