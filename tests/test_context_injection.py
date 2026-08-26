"""P1 上下文注入测试：graph 入口改写 + ReAct 历史注入。

覆盖 2026-08-15 P1 修复：
- query_rewrite 节点：graph 路径（/chat/send）与 SSE 快速路径统一做指代消解
- REACT_REASONING 模板：ReAct 循环拿到最近对话历史（新增 {history} 占位符）

注：TaskBrief/Worker/Orchestrator 相关用例随多智能体统一到 DeepAgents 而退役，
对话历史改由 deep agent 的 checkpointer 会话记忆承接。
"""
from __future__ import annotations

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
