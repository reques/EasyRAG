"""DeepAgents 集成 — SubAgent 配置与构建。

配置化的子智能体：``name / description / system_prompt / tools``。
主 Agent 通过 ``task(description, subagent_type)`` 工具按描述选择 SubAgent
（模型自动路由，业务代码零 if/else）。

SubAgent 用 langgraph ``create_react_agent`` 构建（DeepAgents 底层同款
harness），每次 invoke 独立 state —— 子 Agent 上下文天然与主 Agent 隔离，
结果以纯文本返回，不污染主 Agent state。

tools 声明语法（2026-08-26 阶段 2 扩展，见 ``resolve_tool_spec``）：
  - 普通名称 ``"web_search"``；  - ``"*"`` 全量；
  - ``"except:<name>"`` 排除；   - ``"@tag"`` 按 metadata 能力标签。
权限硬边界不变：候选池始终是 registry.list_all()（已过 check_fn + skills
白名单 + KB 授权 ContextVar），动态绑定只能收窄不能放大。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
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
    # 工具白名单声明：普通名 / "*" / "except:<name>" / "@tag"（tuple 保证 hashable）
    tools: Tuple[str, ...] = ()


def resolve_tool_spec(specs: Tuple[str, ...]) -> Tuple[str, ...]:
    """把 SubAgentConfig.tools 声明解析为实际工具名列表（阶段 2）。

    语义：
      - ``"*"``        → 全量可用工具；
      - ``"@tag"``     → metadata["tags"] 含该标签的工具；
      - ``"except:x"`` → 从结果中剔除（可与任意包含语法组合）；
      - 普通名       → 已注册且可用则保留，未注册静默忽略（与旧行为一致）。
    候选池为 ``registry.list_all()``——已过 check_fn / skills 白名单过滤，
    解析结果只能收窄权限，不能放大。返回按名称排序的 tuple（可作缓存指纹）。"""
    from app.tools.registry import get_tool_registry

    available = get_tool_registry().list_all()
    names = {t.name for t in available}
    include: set = set()
    exclude: set = set()
    for raw in specs:
        item = str(raw).strip()
        if not item:
            continue
        if item.startswith("except:"):
            exclude.add(item[len("except:"):].strip())
        elif item == "*":
            include |= names
        elif item.startswith("@"):
            tag = item[1:].lower()
            include |= {
                t.name
                for t in available
                if tag in {str(x).lower() for x in (t.metadata or {}).get("tags", [])}
            }
        elif item in names:
            include.add(item)
    return tuple(sorted(include - exclude))


# ── 内置默认 SubAgent（可被 DEEP_SUBAGENTS_FILE 覆盖）────────────────────────

DEFAULT_SUBAGENTS: List[SubAgentConfig] = [
    SubAgentConfig(
        name="research-agent",
        description="负责资料搜索与深入研究：联网检索、查证事实、收集资料并输出结构化研究结论。",
        system_prompt=(
            "你是一名资深研究助理。你的职责是围绕任务目标进行资料搜索与深入研究。\n"
            "规则：\n"
            "1. 需要外部信息时优先使用 web_search 工具检索，不要编造事实；\n"
            "   企业内部资料优先使用 kb_search 检索知识库。\n"
            "2. 检索后归纳关键结论，标注来源。\n"
            "3. 最终用中文输出结构化研究结果（要点列表 + 简短总结），"
            "不要提及工具调用过程，直接给结论。"
        ),
        tools=("web_search", "kb_search", "text_tool", "datetime_tool"),
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


def _load_subagents_file(path: str) -> Optional[List[Dict[str, Any]]]:
    """从 JSON/YAML 文件读取 SubAgent 配置（可选覆盖）。

    健壮性（2026-08-21, S2 修复）：文件不存在 / 解析失败 / 内容为空 →
    返回 None 并告警，调用方回退内置默认，不因坏配置崩启动。
    条目校验：缺少 name 或 name 为空 → 跳过（避免注册无名子智能体）。
    """
    if not path:
        return None
    if not os.path.isfile(path):
        logger.warning("[deepagents] DEEP_SUBAGENTS_FILE not found: %s (fallback to builtin)", path)
        return None
    try:
        import json

        try:
            import yaml  # type: ignore

            loader = yaml.safe_load
        except ImportError:
            loader = None

        with open(path, encoding="utf-8") as f:
            text = f.read()
        data = None
        if path.endswith((".yaml", ".yml")) and loader:
            data = loader(text)
        else:
            data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("subagents", [])
        if not isinstance(data, list):
            logger.warning(
                "[deepagents] DEEP_SUBAGENTS_FILE %s has no subagents list (fallback to builtin)",
                path,
            )
            return None
        items = []
        for d in data:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name", "") or "").strip()
            if not name:
                logger.warning("[deepagents] skip subagent entry without name in %s", path)
                continue
            items.append(d)
        return items if items else None
    except Exception as exc:
        logger.warning(
            "[deepagents] failed to load DEEP_SUBAGENTS_FILE %s: %s (fallback to builtin)",
            path, exc,
        )
        return None


def load_subagents() -> List[SubAgentConfig]:
    """加载 SubAgent 配置：外部文件（可选）→ 内置默认。"""
    cfg = get_settings()
    override = _load_subagents_file(cfg.DEEP_SUBAGENTS_FILE)
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
        logger.info(
            "[deepagents] loaded %d subagents from %s", len(configs), cfg.DEEP_SUBAGENTS_FILE
        )
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


# 生产路径缓存：key = (subagent 名, 解析后工具集指纹)——工具集变化时重建；
# 测试注入 mock 时绕过缓存（model 非 None）
_subagent_cache: Dict[Tuple[str, Tuple[str, ...]], Any] = {}


def build_subagent(config: SubAgentConfig, model=None):
    """构建 SubAgent 的 langgraph compiled graph（create_react_agent）。

    - 工具子集 = ``resolve_tool_spec(config.tools)`` 解析结果（经注册表转换，
      技能/可用性检查生效）
    - 缓存 key 含解析后的工具集指纹：工具集变化（配置改动 / 技能白名单 /
      MCP 动态注册）时自动重建（阶段 2）
    - 每次 invoke 独立 state → 子 Agent 上下文隔离
    - model: 测试可注入 mock（此时不缓存）；None = 项目配置的真实模型
    """
    from langgraph.prebuilt import create_react_agent

    from app.agents.deep.llm import get_langchain_model
    from app.agents.deep.tools import registry_to_langchain_tools

    tool_names = resolve_tool_spec(config.tools)
    cache_key = (config.name, tool_names)
    cacheable = model is None  # 修复（2026-08-26 阶段 2）：model 随后会被重赋值，
    # 旧版 `if model is None:` 存入分支恒不成立——缓存从不生效（每次重建）
    if cacheable and cache_key in _subagent_cache:
        return _subagent_cache[cache_key]
    if model is None:
        model = get_langchain_model()
    tools = registry_to_langchain_tools(list(tool_names))
    logger.info(
        "[deepagents] build subagent %s: %d tools %s",
        config.name, len(tools), list(tool_names),
    )
    agent = create_react_agent(
        model=model,
        tools=tools,
        # 阶段 4：统一追加结构化尾部约定（外部配置文件也自动生效）
        prompt=config.system_prompt + RESULT_TAIL_PROMPT,
        name=f"subagent_{config.name}",
    )
    if cacheable:
        _subagent_cache[cache_key] = agent
    return agent


def _maybe_narrow_tools_by_task(
    config: SubAgentConfig, task_description: str
) -> SubAgentConfig:
    """执行时动态收窄工具集（阶段 2，DEEP_DYNAMIC_TOOLS 开关，默认关闭）。

    按任务描述 ``discover()`` 出相关工具，与配置解析结果取交集（只收窄）；
    discover 无命中或交集为空时保留原配置（避免子智能体无工具可用）。"""
    if not get_settings().DEEP_DYNAMIC_TOOLS:
        return config
    from app.tools.registry import get_tool_registry

    static_names = set(resolve_tool_spec(config.tools))
    discovered = {t.name for t in get_tool_registry().discover(task_description)}
    if not discovered:
        return config
    narrowed = tuple(sorted(static_names & discovered))
    if not narrowed:
        return config
    if tuple(narrowed) == tuple(sorted(static_names)):
        return config
    logger.info(
        "[deepagents] dynamic tool binding for %s: %s -> %s",
        config.name, sorted(static_names), list(narrowed),
    )
    return replace(config, tools=narrowed)


# ── 结构化尾部（2026-08-26 阶段 4）：供主 Agent 决策 replan ─────────────

RESULT_TAIL_PROMPT = """

输出约定：回答正文结束后，另起一行追加一个 JSON 尾块（不要包在代码块里）：
{"status": "completed|partial|failed", "concerns": "遗留问题/不确定点，无则空字符串", "suggested_followup": "建议的后续动作，无则空字符串"}
"""


def parse_result_tail(text: str) -> Dict[str, Any]:
    """解析子智能体回答的结构化 JSON 尾块（阶段 4）。

    约定：回答末尾追加 ``{"status": ..., "concerns": ..., "suggested_followup": ...}``。
    解析失败回退纯文本（不阻断主流程）：status="unknown"，raw 携带尾部原文。"""
    import json as _json
    import re as _re

    tail = (text or "").strip()
    if not tail:
        return {"status": "unknown", "concerns": "", "suggested_followup": "", "raw": ""}
    # 取最后一个以 } 结尾的 {...} 片段（容忍尾块前有多余文本）
    candidates = _re.findall(r"\{[^{}]*\"status\"[^{}]*\}", tail, flags=_re.DOTALL)
    for cand in reversed(candidates):
        try:
            parsed = _json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, dict) and "status" in parsed:
            return {
                "status": str(parsed.get("status", "unknown")),
                "concerns": str(parsed.get("concerns", "") or ""),
                "suggested_followup": str(parsed.get("suggested_followup", "") or ""),
                "raw": cand,
            }
    return {"status": "unknown", "concerns": "", "suggested_followup": "",
            "raw": tail[-300:]}


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

    2026-08-21（S3）：改为 ``agent.stream`` 循环执行。有请求级观察者
    （``use_subagent_observers`` 设置）时，把子 Agent 的推理/工具调用/
    工具返回步骤以 ``{subagent_name}/step`` 形式透传给 SSE 回调；无观察者
    时行为与原来一致（跳过解析，开销不变）。
    """
    from app.agents.deep.observe import get_subagent_observers

    # 阶段 2：执行时动态收窄工具集（开关默认关闭；只收窄不放大）
    config = _maybe_narrow_tools_by_task(config, task_description)
    agent = build_subagent(config, model=model)
    on_step, on_artifact = get_subagent_observers() or (None, None)
    final_state = None
    for chunk in agent.stream(
        {"messages": [("user", task_description)]},
        config={"recursion_limit": recursion_limit},
        stream_mode="values",
    ):
        final_state = chunk
        if on_step is None and on_artifact is None:
            continue
        msgs = chunk.get("messages") or []
        if not msgs:
            continue
        last = msgs[-1]
        mtype = getattr(last, "type", "")
        tc = getattr(last, "tool_calls", None)
        if tc:
            tool_name = tc[0].get("name", "")
            # act and reasoning：思考内容作为独立 reason 步骤透出
            thought = str(getattr(last, "content", "") or "").strip()
            _thought = " ".join(thought.split())[:200]
            if len(thought) > 200:
                _thought += "…"
            _args = tc[0].get("args") or {}
            try:
                import json
                _args_text = json.dumps(_args, ensure_ascii=False)
            except Exception:
                _args_text = str(_args)
            if on_step:
                if _thought:
                    on_step(f"{config.name}/reason", _thought)
                _args_short = " ".join(_args_text.split())[:120]
                if len(_args_text) > 120:
                    _args_short += "…"
                on_step(
                    f"{config.name}/tool",
                    f"调用 {tool_name} {_args_short}".rstrip(),
                )
            if on_artifact:
                # 阶段 6：子智能体的工具调用（含参数）与推理一并实时透传，
                # 前端在同一会话内可见子 Agent 的完整动作链。
                on_artifact(
                    "thought", f"{config.name}/reason", "子智能体推理", _thought
                )
                on_artifact(
                    "tool", f"{config.name}/tool",
                    f"调用 {tool_name}", _args_text[:800],
                )
        elif mtype == "tool":
            t_content = str(getattr(last, "content", "") or "")
            if on_step:
                on_step(f"{config.name}/tool_done", f"工具返回: {t_content[:120]}")
            if on_artifact:
                on_artifact(
                    "tool_result", f"{config.name}/tool", "工具返回",
                    " ".join(t_content.split())[:300],
                )
        elif mtype == "ai" and getattr(last, "content", ""):
            if on_step:
                on_step(f"{config.name}/generate", "子智能体生成回答中...")
    messages = (final_state or {}).get("messages") or []
    if not messages:
        return "（子智能体未返回任何内容）"
    last = messages[-1]
    content = getattr(last, "content", "") or ""
    if isinstance(content, list):  # multimodal 内容块
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        content = "".join(parts)
    return str(content)
