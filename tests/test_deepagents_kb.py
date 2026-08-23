"""DeepAgents 知识库接入（S1）测试：kb_search 工具 + _run_deep 前置检索注入。

2026-08-21 修复：DeepAgents 路径此前从不检索知识库（系统提示却声称"检索结果
会作为上下文提供"）——知识库问答退化成纯 LLM 生成。本次：
1. 新增注册表工具 kb_search（请求级授权范围经 ContextVar 注入）；
2. _run_deep 生成前执行增强检索并注入 system 上下文，收集 sources。

全部用 mock（不调用真实 LLM / Milvus）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _fake_retriever(expected_kb_ids=("kb-1",)):
    class _Fake:
        def retrieve(self, query, history=None, knowledge_base_ids=None):
            assert knowledge_base_ids == list(expected_kb_ids), (
                f"retriever 应收到授权范围 {expected_kb_ids}，实际 {knowledge_base_ids}"
            )
            return SimpleNamespace(
                knowledge_blocks=[],
                raw_docs=[
                    SimpleNamespace(
                        content="员工手册规定：入职满一年享年假 5 天。",
                        metadata={"source": "员工手册.pdf"},
                    )
                ],
                sources=[{"title": "员工手册.pdf", "url": "", "type": "kb"}],
            )

    return _Fake()


# ── kb_search 工具 ─────────────────────────────────────────────────────────
def test_kb_search_tool_registered():
    from app.tools.registry import get_tool_registry

    tool = get_tool_registry().get("kb_search")
    assert tool.name == "kb_search"
    assert "query" in tool.arg_schema
    assert tool.arg_schema["query"][2] is True  # query 必填


def test_kb_search_denies_without_scope():
    from app.services.knowledge_context import get_authorised_kb_ids
    from app.tools.registry import get_tool_registry

    assert get_authorised_kb_ids() is None  # 测试进程默认无授权
    out = get_tool_registry().invoke("kb_search", query="年假")
    assert "未授权" in out


def test_kb_search_with_scope(monkeypatch):
    monkeypatch.setattr(
        "app.rag.enhanced_retriever.get_enhanced_retriever", lambda: _fake_retriever()
    )
    from app.services.knowledge_context import use_authorised_kb_ids
    from app.tools.registry import get_tool_registry

    with use_authorised_kb_ids(["kb-1"]):
        out = get_tool_registry().invoke("kb_search", query="年假")
    assert "员工手册" in out
    assert "来源" in out


# ── 请求级授权上下文 ───────────────────────────────────────────────────────
def test_kb_context_scope_set_and_restore():
    from app.services.knowledge_context import (
        get_authorised_kb_ids,
        use_authorised_kb_ids,
    )

    assert get_authorised_kb_ids() is None
    with use_authorised_kb_ids(["kb-1", "kb-2"]):
        assert get_authorised_kb_ids() == ["kb-1", "kb-2"]
    assert get_authorised_kb_ids() is None  # 退出恢复

    # 显式空授权：不应继承外层遗留值
    with use_authorised_kb_ids(["kb-1"]):
        with use_authorised_kb_ids(None):
            assert get_authorised_kb_ids() is None


# ── _run_deep 前置检索注入 ─────────────────────────────────────────────────
def test_run_deep_injects_kb_context(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.services.agent_service import AgentService, SessionStore

    captured: dict = {}

    class _FakeAgent:
        def stream(self, inputs, config=None, stream_mode="values"):
            captured["messages"] = inputs["messages"]
            yield {"messages": [AIMessage(content="根据员工手册，年假为 5 天。")]}

    monkeypatch.setattr("app.agents.deep.agent.get_main_agent", lambda: _FakeAgent())
    monkeypatch.setattr(
        "app.rag.enhanced_retriever.get_enhanced_retriever", lambda: _fake_retriever()
    )

    svc = object.__new__(AgentService)
    svc._sessions = SessionStore(ttl=3600)
    result = svc._run_deep(
        "年假几天",
        history=[],
        user_id=None,
        knowledge_base_ids=["kb-1"],
    )

    assert result["final_answer"] == "根据员工手册，年假为 5 天。"
    assert result["sources"] == [{"title": "员工手册.pdf", "url": "", "type": "kb"}]
    # 检索上下文已注入 system 消息
    sys_msgs = [
        m for m in captured["messages"]
        if m.get("role") == "system" and "知识库检索到" in m.get("content", "")
    ]
    assert sys_msgs, "messages 应包含知识库检索上下文 system 消息"
    assert "员工手册" in sys_msgs[0]["content"]
    steps_text = "\n".join(result["steps"])
    assert "知识库命中 1 条" in steps_text


def test_run_deep_without_kb_skips_retrieval(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.services.agent_service import AgentService, SessionStore

    def _boom(*a, **k):
        raise AssertionError("无授权知识库时不应触发检索")

    class _FakeAgent:
        def stream(self, inputs, config=None, stream_mode="values"):
            yield {"messages": [AIMessage(content="你好！")]}

    monkeypatch.setattr("app.agents.deep.agent.get_main_agent", lambda: _FakeAgent())
    monkeypatch.setattr("app.rag.enhanced_retriever.get_enhanced_retriever", _boom)

    svc = object.__new__(AgentService)
    svc._sessions = SessionStore(ttl=3600)
    # 无 knowledge_base_ids：跳过检索（检索器被调即断言失败），主 Agent 正常执行
    result = svc._run_deep("你好", history=[], user_id=None, knowledge_base_ids=None)
    assert result["final_answer"] == "你好！"
    assert result["sources"] == []


# ── SubAgent 白名单 ────────────────────────────────────────────────────────
def test_research_agent_has_kb_search():
    from app.agents.deep.subagents import get_subagent_config

    cfg = get_subagent_config("research-agent")
    assert cfg is not None
    assert "kb_search" in cfg.tools
    assert "kb_search" in cfg.system_prompt  # 提示已引导使用
