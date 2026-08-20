"""DeepAgents 集成 — SubAgent 配置与构建。

配置化的子智能体：``name / description / system_prompt / tools``。
主 Agent 通过 ``task(description, subagent_type)`` 工具按描述选择 SubAgent
（模型自动路由，业务代码零 if/else）。

SubAgent 用 langgraph ``create_react_agent`` 构建（DeepAgents 底层同款
harness），每次 invoke 独立 state —— 子 Agent 上下文天然与主 Agent 隔离，
结果以纯文本返回，不污染主 Agent state。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SubAgentConfig:
    """单个子智能体配置。"""

    name: str                    # 唯一标识（task 工具的 subagent_type）
    description: str             # 能力描述（供主 Agent 选择）
    system_prompt: str           # 子 Agent 系统提示
    tools: Tuple[str, ...] = ()  # 工具白名单（注册表内名称；tuple 保证 hashable）


# ── 内置默认 SubAgent（可被 DEEP_SUBAGENTS_FILE 覆盖）────────────────────────

DEFAULT_SUBAGENTS: List[SubAgentConfig] = [
    SubAgentConfig(
        name="research-agent",
        description="负责资料搜索与深入研究：联网检索、查证事实、收集资料并输出结构化研究结论。",
        system_prompt=(
            "你是一名资深研究助理。你的职责是围绕任务目标进行资料搜索与深入研究。\n"
            "规则：\n"
            "1. 需要外部信息时优先使用 web_search 工具检索，不要编造事实。\n"
            "2. 检索后归纳关键结论，标注来源。\n"
            "3. 最终用中文输出结构化研究结果（要点列表 + 简短总结），"
            "不要提及工具调用过程，直接给结论。"
        ),
        tools=("web_search", "text_tool", "datetime_tool"),
    ),
    SubAgentConfig(
        name="coding-agent",
        description="负责代码分析与实现：阅读/生成代码、解释逻辑、编写脚本、排查代码问题。",
        system_prompt=(
            "你是一名资深软件工程师。你的职责是分析、编写、解释和排查代码。\n"
            "规则：\n"
            "1. 代码必须可直接运行或语义完整，关键步骤给出注释。\n"
            "2. 需要计算时使用 calculator 工具。\n"
            "3. 最终用中文输出：实现思路（简述）+ 代码 + 使用说明。"
        ),
        tools=("text_tool", "calculator", "datetime_tool"),
    ),
]


def _load_subagents_file(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """从 JSON/YAML 文件读取 SubAgent 配置（可选覆盖）。"""
    if not path or not os.path.isfile(path):
        return None
    import json

    try:
        import yaml  # type: ignore

        loader = yaml.safe_load
    except ImportError:
        loader = None

    text = open(path, encoding="utf-8").read()
    data = None
    if path.endswith((".yaml", ".yml")) and loader:
        data = loader(text)
    else:
        data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("subagents", [])
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else None


def load_subagents() -> List[SubAgentConfig]:
    """加载 SubAgent 配置：外部文件（可选）→ 内置默认。"""
    cfg = get_settings()
    override = _load_subagents_file(getattr(cfg, "DEEP_SUBAGENTS_FILE", None))
    if override:
        configs = []
        for item in override:
            configs.append(
                SubAgentConfig(
                    name=str(item["name"]),
                    description=str(item.get("description", "")),
                    system_prompt=str(item.get("system_prompt", "")),
                    tools=tuple(str(t) for t in item.get("tools", [])),
                )
            )
        logger.info("[deepagents] loaded %d subagents from %s", len(configs), cfg.DEEP_SUBAGENTS_FILE)
        return configs
    return list(DEFAULT_SUBAGENTS)


@lru_cache(maxsize=1)
def get_subagents() -> List[SubAgentConfig]:
    """进程级缓存的 SubAgent 配置列表。"""
    return load_subagents()


def get_subagent_config(name: str) -> Optional[SubAgentConfig]:
    """按名称查找 SubAgent 配置（未命中返回 None，task 工具抛错提示可选集）。"""
    for cfg in get_subagents():
        if cfg.name == name:
            return cfg
    return None


def subagents_prompt() -> str:
    """可用的 SubAgent 名册文本（注入主 Agent system prompt）。"""
    lines = []
    for cfg in get_subagents():
        tools = "、".join(cfg.tools) if cfg.tools else "无工具"
        lines.append(f"- {cfg.name}: {cfg.description}（工具: {tools}）")
    return "\n".join(lines) or "（无可用子智能体）"


# 生产路径缓存（按 subagent name；测试注入 mock 时绕过缓存）
_subagent_cache: Dict[str, Any] = {}


def build_subagent(config: SubAgentConfig, model=None):
    """构建 SubAgent 的 langgraph compiled graph（create_react_agent）。

    - 工具子集 = 配置的 tools 白名单（经注册表转换，技能/可用性检查生效）
    - 每次 invoke 独立 state → 子 Agent 上下文隔离
    - model: 测试可注入 mock（此时不缓存）；None = 项目配置的真实模型
    """
    from langgraph.prebuilt import create_react_agent

    from app.agents.deep.llm import get_langchain_model
    from app.agents.deep.tools import registry_to_langchain_tools

    if model is None and config.name in _subagent_cache:
        return _subagent_cache[config.name]
    if model is None:
        model = get_langchain_model()
    tools = registry_to_langchain_tools(config.tools)
    logger.info(
        "[deepagents] build subagent %s: %d tools", config.name, len(tools)
    )
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=config.system_prompt,
        name=f"subagent_{config.name}",
    )
    if model is None:
        _subagent_cache[config.name] = agent
    return agent


def run_subagent(
    config: SubAgentConfig,
    task_description: str,
    model=None,
    recursion_limit: int = 20,
) -> str:
    """同步运行 SubAgent，返回最终回答文本。

    Args:
        config: SubAgent 配置。
        task_description: 委派的任务描述（含目标/上下文/期望输出）。
        model: 测试可注入 mock；None = 项目配置的真实模型。
        recursion_limit: langgraph recursion_limit。
    """
    agent = build_subagent(config, model=model)
    result = agent.invoke(
        {"messages": [("user", task_description)]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result.get("messages") or []
    if not messages:
        return "（子智能体未返回任何内容）"
    last = messages[-1]
    content = getattr(last, "content", "") or ""
    if isinstance(content, list):  # multimodal 内容块
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        content = "".join(parts)
    return str(content)
