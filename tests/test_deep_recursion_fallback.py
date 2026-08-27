"""阶段 1：DeepAgents 超限降级（S4）— GraphRecursionError → forced answer。

用假主 Agent（mock stream）触发 GraphRecursionError / 普通异常，不调用真实 LLM。
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from app.services.agent_service import AgentService


class _FakeDeepAgent:
    """模拟 create_react_agent 的 stream 行为。

    chunks: 每个 yield 一个全量 state dict；之后 raise 指定异常。
    """

    def __init__(self, chunks: List[Dict[str, Any]], error: Exception):
        self._chunks = chunks
        self._error = error

    def stream(self, messages, config=None, stream_mode=None):
        for chunk in self._chunks:
            yield chunk
        raise self._error


def _partial_state_chunk():
    """一个典型的"部分执行状态"：用户消息 + AI 工具调用 + 工具结果（无最终回答）。"""
    return {
        "messages": [
            HumanMessage(content="帮我调研 A 和 B 并对比"),
            AIMessage(content="", tool_calls=[{
                "name": "task",
                "args": {"description": "调研 A", "subagent_type": "research-agent"},
                "id": "call_1",
            }]),
            ToolMessage(content="A 的初步调研产出：要点 1、要点 2", tool_call_id="call_1"),
        ]
    }


def _service() -> AgentService:
    svc = object.__new__(AgentService)
    svc._sessions = None
    return svc


def _run(svc: AgentService, fake_agent: _FakeDeepAgent, monkeypatch) -> Dict[str, Any]:
    monkeypatch.setattr(
        "app.agents.deep.agent.get_main_agent", lambda: fake_agent
    )
    return svc._run_deep(
        "帮我调研 A 和 B 并对比",
        session_id="conv-test",
        history=[],           # 跳过 SessionStore
        user_id=None,          # 跳过用户 facts
        knowledge_base_ids=None,   # 跳过前置检索
        knowledge_catalog=None,
    )


# ── S4：超限降级 ───────────────────────────────────────────────────────────


def test_recursion_limit_forces_answer_from_partial_state(monkeypatch):
    """GraphRecursionError → 基于已有工具结果收尾，不再直接返回错误。"""
    svc = _service()
    fake = _FakeDeepAgent([_partial_state_chunk()], GraphRecursionError("limit reached"))
    result = _run(svc, fake, monkeypatch)

    assert result["is_fallback"] is False
    assert result["degraded"] is True
    assert result["final_answer"].startswith("基于已有信息")
    assert "要点 1" in result["final_answer"]
    # 步骤记录超限事件
    assert any("recursion limit" in s for s in result["steps"])
    assert any("fallback" in s for s in result["steps"])


def test_recursion_limit_with_final_ai_answer_kept(monkeypatch):
    """部分状态里已有完整 AI 回答 → 直接沿用，不强改格式。"""
    svc = _service()
    chunk = {
        "messages": [
            HumanMessage(content="问题"),
            AIMessage(content="这是已经生成好的完整回答。"),
        ]
    }
    fake = _FakeDeepAgent([chunk], GraphRecursionError("limit reached"))
    result = _run(svc, fake, monkeypatch)
    assert result["degraded"] is True
    assert result["final_answer"] == "这是已经生成好的完整回答。"


def test_recursion_limit_without_any_observations(monkeypatch):
    """超限且无任何工具产出 → 给出明确的降级提示（而非空回答/报错）。"""
    svc = _service()
    chunk = {"messages": [HumanMessage(content="问题")]}
    fake = _FakeDeepAgent([chunk], GraphRecursionError("limit reached"))
    result = _run(svc, fake, monkeypatch)
    assert result["is_fallback"] is False
    assert "推理步数上限" in result["final_answer"]


# ── 统一事件流：trace + events 随响应返回 ─────────────────────────────────


def test_run_deep_returns_trace_and_events(monkeypatch):
    """响应携带 trace_id 与结构化事件列表（步骤/工件进同一事件流）。"""
    svc = _service()
    chunk = {
        "messages": [
            HumanMessage(content="问题"),
            AIMessage(content="最终回答内容。"),
        ]
    }

    class _NormalAgent:
        def stream(self, messages, config=None, stream_mode=None):
            yield chunk

    monkeypatch.setattr("app.agents.deep.agent.get_main_agent", lambda: _NormalAgent())
    result = svc._run_deep(
        "问题", session_id="conv-evt", history=[], user_id=None,
        knowledge_base_ids=None, knowledge_catalog=None,
    )

    assert result["trace_id"]
    assert isinstance(result["events"], list) and result["events"]
    stages = [ev["stage"] for ev in result["events"]]
    assert "understand" in stages           # _step 发出
    assert "generate_done" in stages
    assert all(ev["trace_id"] == result["trace_id"] for ev in result["events"])


# ── 兜底：普通异常仍走错误回退（既有行为不变）──────────────────────────────


def test_generic_error_still_returns_fallback(monkeypatch):
    svc = _service()
    fake = _FakeDeepAgent([_partial_state_chunk()], RuntimeError("模型服务不可用"))
    result = _run(svc, fake, monkeypatch)
    assert result["is_fallback"] is True
    assert "处理请求时发生错误" in result["final_answer"]
    assert result["degraded"] is False
