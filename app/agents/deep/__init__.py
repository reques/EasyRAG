"""DeepAgents 风格 SubAgent 集成（langchain 0.3 / langgraph 0.6 自组装）。

因项目钉死 langchain==0.3.26（deepagents 官方包要求 langchain>=1.0），
本模块参考 DeepAgents 的 SubAgent 架构（task 委派工具 + 配置化子智能体 +
上下文隔离），用 langchain create_agent 自组装等价能力。
"""
from app.agents.deep.agent import build_main_agent, get_main_agent
from app.agents.deep.llm import get_langchain_model
from app.agents.deep.subagents import (
    DEFAULT_SUBAGENTS,
    SubAgentConfig,
    build_subagent,
    get_subagent_config,
    get_subagents,
    load_subagents,
    run_subagent,
    subagents_prompt,
)
from app.agents.deep.task_tool import build_task_tool, task_system_prompt
from app.agents.deep.tools import registry_to_langchain_tools, tools_prompt

__all__ = [
    "DEFAULT_SUBAGENTS",
    "SubAgentConfig",
    "build_main_agent",
    "build_subagent",
    "build_task_tool",
    "get_langchain_model",
    "get_main_agent",
    "get_subagent_config",
    "get_subagents",
    "load_subagents",
    "registry_to_langchain_tools",
    "run_subagent",
    "subagents_prompt",
    "task_system_prompt",
    "tools_prompt",
]
