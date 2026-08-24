"""P1 上下文注入测试：graph 入口改写 + ReAct/多智能体历史注入。

覆盖 2026-08-15 P1 修复：
- query_rewrite 节点：graph 路径（/chat/send）与 SSE 快速路径统一做指代消解
- REACT_REASONING 模板：ReAct 循环拿到最近对话历史（新增 {history} 占位符）
- TaskBrief.history：子任务携带最近对话，Worker 注入"对话背景"上下文
- Orchestrator._decompose：拆解 prompt 注入历史，追问可被正确理解
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.agents.workers.base import TaskBrief
from app.agents.workers.rag_worker import RagWorker
from app.prompts.templates import REACT_REASONING


# ── query_rewrite 节点（graph 入口）──────────────────────────────────────
def test_query_rewrite_node_rewrites_and_steps(monkeypatch):
    import app.graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "rewrite_query_with_history", lambda q, h: "无锡明天天气如何"
    )
    out = nodes.query_rewrite({
        "query": "那明天呢",
        "history": [{"role": "user", "content": "无锡今天天气"}],
        "steps": [],
    })
    assert out["query"] == "无锡明天天气如何"
    assert out["steps"] == ["query_rewrite -> 无锡明天天气如何"]


def test_query_rewrite_node_keeps_original_when_no_rewrite(monkeypatch):
    import app.graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "rewrite_query_with_history", lambda q, h: q
    )
    out = nodes.query_rewrite({"query": "什么是民法典", "history": [], "steps": []})
    assert out["query"] == "什么是民法典"
    assert out["steps"] == ["query_rewrite -> 什么是民法典"]


def test_graph_compiles_with_query_rewrite_entry():
    from app.graph.workflow import build_graph

    g = build_graph()
    names = {n for n in g.get_graph().nodes}
    assert "query_rewrite" in names
    assert "intent_recognition" in names


# ── REACT_REASONING 模板注入历史 ─────────────────────────────────────────
def test_react_template_formats_with_history():
    prompt = REACT_REASONING.format(
        tools="[web_search]",
        observations="1. 思考: x | 工具: - | 结果: y",
        query="那第二个呢",
        history="用户: 帮我查民法典第一条\n助手: 已为你找到",
    )
    assert "对话历史" in prompt
    assert "帮我查民法典第一条" in prompt
    assert "那第二个呢" in prompt


# ── TaskBrief.history 与 Worker 注入 ─────────────────────────────────────
def test_task_brief_history_defaults_empty():
    brief = TaskBrief(task_id="task-1", goal="g")
    assert brief.history == []


def test_history_context_message_shapes():
    from app.agents.workers.base import BaseWorker

    brief = TaskBrief(
        task_id="task-1", goal="g",
        history=[
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ],
    )
    msg = BaseWorker._history_context_message(brief)
    assert msg is not None
    assert msg["role"] == "system"
    assert "对话背景" in msg["content"]
    assert "之前的问题" in msg["content"]

    empty = BaseWorker._history_context_message(TaskBrief(task_id="t", goal="g"))
    assert empty is None


def test_rag_worker_injects_history_into_messages():
    class _FakeLLM:
        def __init__(self):
            self.calls: List[List[Dict[str, str]]] = []

        def chat_sync(self, messages, **kwargs):
            self.calls.append(messages)
            return "基于上下文的回答"

    fake = _FakeLLM()
    worker = RagWorker()
    worker.llm = fake
    worker._retriever = MagicMock()
    worker._retriever.retrieve.return_value = []

    brief = TaskBrief(
        task_id="task-1", goal="回答第二个问题",
        history=[{"role": "user", "content": "第一个问题"}, {"role": "assistant", "content": "第一个回答"}],
    )
    report = worker.run(brief)
    assert report.ok(), report.error
    assert fake.calls, "worker 应调用 LLM"
    received = fake.calls[0]
    assert any(
        m.get("role") == "system" and "对话背景" in m.get("content", "")
        for m in received
    )


def test_worker_without_history_skips_injection():
    class _FakeLLM:
        def __init__(self):
            self.calls: List[List[Dict[str, str]]] = []

        def chat_sync(self, messages, **kwargs):
            self.calls.append(messages)
            return "ok"

    fake = _FakeLLM()
    worker = RagWorker()
    worker.llm = fake
    worker._retriever = MagicMock()
    worker._retriever.retrieve.return_value = []

    brief = TaskBrief(task_id="task-1", goal="无历史问题")
    report = worker.run(brief)
    assert report.ok()
    received = fake.calls[0]
    # 无历史时不注入"对话背景"
    for m in received:
        assert "对话背景" not in m.get("content", "")


# ── Orchestrator._decompose 注入历史 ─────────────────────────────────────
def test_decompose_injects_history_into_prompt():
    from app.agents.orchestrator import Orchestrator

    class _FakeLLM:
        def __init__(self):
            self.messages = None

        def chat_json_sync(self, messages, **kwargs):
            self.messages = messages
            return {"needs_decomposition": False, "sub_tasks": [], "final_instruction": ""}

    fake = _FakeLLM()
    orch = Orchestrator()
    orch._llm = fake
    briefs, mode, inst = orch._decompose(
        "那第二个呢",
        [{"role": "user", "content": "第一个问题"}, {"role": "assistant", "content": "第一个回答"}],
    )
    assert briefs == []  # 单一意图 → 不拆解
    user_content = fake.messages[1]["content"]
    assert "对话历史" in user_content
    assert "第一个问题" in user_content
    assert "那第二个呢" in user_content
