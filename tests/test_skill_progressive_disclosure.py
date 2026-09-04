"""渐进式披露的端到端行为 — SkillsMiddleware 挂 create_agent 的真实回路。

对齐 Yuxi 的核心约束："未激活 Skill 的工具即使已注册到 ToolNode 也不能被
模型调用"。这里用脚本化的假模型驱动 langchain ``create_agent``，逐轮检查
绑定给模型的工具集与 system prompt。
"""
from __future__ import annotations

from typing import Any, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.skills.loader import SkillDefinition
from app.skills.middleware import build_skills_middleware
from app.skills.registry import list_builtin_skills
from app.skills.runtime import SkillRuntimeContext, use_skill_context

# 每轮模型调用的观测记录。用模块级状态而不是模型字段：pydantic v2 会在校验
# 时**复制** dict/list 字段，实例里的对象与测试持有的不是同一个引用。
TURNS: List[dict] = []
CURSOR: List[int] = [0]


@pytest.fixture(autouse=True)
def _reset_recorder():
    TURNS.clear()
    CURSOR[0] = 0
    yield


class ScriptedModel(BaseChatModel):
    """按脚本逐轮返回 AIMessage，并记录每轮绑定的工具与 system prompt。"""

    script: List[AIMessage] = []
    bound_tools: Optional[List[str]] = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        clone = self.model_copy()
        clone.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return clone

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system_text = next(
            (str(m.content) for m in messages if m.type == "system"), ""
        )
        TURNS.append({"tools": list(self.bound_tools or []), "system": system_text})
        index = CURSOR[0]
        CURSOR[0] = index + 1
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"WEB:{query}"


@tool
def kb_search(query: str) -> str:
    """Search the knowledge base."""
    return f"KB:{query}"


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _agent(script: List[AIMessage]):
    from langchain.agents import create_agent

    return create_agent(
        model=ScriptedModel(script=script),
        tools=[web_search, kb_search],
        system_prompt="BASE PROMPT",
        middleware=[build_skills_middleware()],
    )


def test_tool_unlocks_only_after_read_skill():
    """核心回路：硬调被拒 → read_skill → 下一轮工具出现并可执行。"""
    script = [
        _call("web_search", {"query": "early"}, "c0"),          # 未激活，应被拒
        _call("read_skill", {"slug": "web-research"}, "c1"),    # 激活
        _call("web_search", {"query": "now"}, "c2"),            # 已解锁
        AIMessage(content="最终回答"),
    ]
    runtime = SkillRuntimeContext.from_selection(["web-research"], list_builtin_skills())
    with use_skill_context(runtime):
        result = _agent(script).invoke({"messages": [("user", "查最新消息")]})

    turns = TURNS
    # 前两轮：web_search 不在工具表里（模型看不到）
    assert "web_search" not in turns[0]["tools"]
    assert "read_skill" in turns[0]["tools"]
    # 激活之后：出现在工具表里
    assert "web_search" in turns[2]["tools"]

    contents = [str(m.content) for m in result["messages"]]
    assert any("尚未读取的 Skill" in c for c in contents), "硬调应被 wrap_tool_call 拦下"
    assert any("请先调用 read_skill" in c for c in contents), "拒绝信息要给出可执行下一步"
    assert any(c.startswith("WEB:now") for c in contents), "激活后工具应真正执行"
    assert result["activated_skills"] == ["web-research"]


def test_prompt_switches_from_summary_to_body():
    script = [
        _call("read_skill", {"slug": "web-research"}, "c1"),
        AIMessage(content="done"),
    ]
    runtime = SkillRuntimeContext.from_selection(["web-research"], list_builtin_skills())
    with use_skill_context(runtime):
        _agent(script).invoke({"messages": [("user", "q")]})

    turns = TURNS
    assert "尚未读取" in turns[0]["system"]
    assert "BASE PROMPT" in turns[0]["system"], "Agent 自身 prompt 必须保留"
    assert "已读取的 Skill 指令" in turns[1]["system"]
    assert "尚未读取" not in turns[1]["system"]


def test_dependency_closure_is_visible_but_locked():
    """legal-analysis 依赖 knowledge-research：依赖只进描述范围，工具仍锁着。"""
    script = [_call("kb_search", {"query": "条款"}, "c0"), AIMessage(content="done")]
    runtime = SkillRuntimeContext.from_selection(["legal-analysis"], list_builtin_skills())
    assert runtime.effective_slugs == ("legal-analysis", "knowledge-research")

    with use_skill_context(runtime):
        result = _agent(script).invoke({"messages": [("user", "q")]})

    summary = TURNS[0]["system"]
    assert "legal-analysis" in summary and "knowledge-research" in summary
    # kb_search 是公共工具 → 即使 knowledge-research 未激活也放行
    assert "kb_search" in TURNS[0]["tools"]
    assert any(str(m.content).startswith("KB:") for m in result["messages"])


def test_preloaded_skill_is_active_from_first_turn():
    script = [_call("web_search", {"query": "x"}, "c0"), AIMessage(content="done")]
    runtime = SkillRuntimeContext.from_selection(
        ["web-research"], list_builtin_skills(), preload=["web-research"]
    )
    with use_skill_context(runtime):
        result = _agent(script).invoke({"messages": [("user", "q")]})

    assert "web_search" in TURNS[0]["tools"], "preload 应首轮解锁"
    assert "已读取的 Skill 指令" in TURNS[0]["system"]
    assert any(str(m.content).startswith("WEB:") for m in result["messages"])
    assert result["activated_skills"] == ["web-research"]


def test_read_skill_hidden_when_no_skill_selected():
    """未启用 Skill 时不该暴露 read_skill（没有可读的东西）。"""
    script = [AIMessage(content="直接回答")]
    with use_skill_context(SkillRuntimeContext()):
        _agent(script).invoke({"messages": [("user", "q")]})

    turn = TURNS[0]
    assert "read_skill" not in turn["tools"]
    assert set(turn["tools"]) == {"web_search", "kb_search"}
    assert "可用 Skill" not in turn["system"]


def test_read_skill_reports_unknown_slug():
    script = [
        _call("read_skill", {"slug": "no-such-skill"}, "c1"),
        AIMessage(content="done"),
    ]
    runtime = SkillRuntimeContext.from_selection(["web-research"], list_builtin_skills())
    with use_skill_context(runtime):
        result = _agent(script).invoke({"messages": [("user", "q")]})

    contents = [str(m.content) for m in result["messages"]]
    assert any("没有名为" in c and "web-research" in c for c in contents)
    assert result.get("activated_skills") in (None, [], ())


def test_personal_skill_definition_participates_in_gate():
    """个人 Skill（source=personal）与内置在门控里完全同构。"""
    personal = SkillDefinition(
        slug="my-research", name="我的研究法", description="自定义研究流程",
        body="先列假设再查证。", tool_dependencies=("web_search",),
        source="personal", owner_id="u1",
    )
    script = [
        _call("web_search", {"query": "a"}, "c0"),
        _call("read_skill", {"slug": "my-research"}, "c1"),
        _call("web_search", {"query": "b"}, "c2"),
        AIMessage(content="done"),
    ]
    with use_skill_context(SkillRuntimeContext.from_definitions([personal])):
        result = _agent(script).invoke({"messages": [("user", "q")]})

    assert "web_search" not in TURNS[0]["tools"]
    assert "web_search" in TURNS[2]["tools"]
    contents = [str(m.content) for m in result["messages"]]
    assert any("先列假设再查证" in c for c in contents)
    assert any(c.startswith("WEB:b") for c in contents)
