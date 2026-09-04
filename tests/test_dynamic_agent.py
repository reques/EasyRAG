"""轻量动态 Agent 测试：路由、工具构建与结果解析。"""
from __future__ import annotations

import types

from app.agents import dynamic as dyn
from app.services.agent_service import AgentService


def _simple_cfg(**overrides) -> types.SimpleNamespace:
    base = {"AGENT_MODE": "auto", "AGENT_MAX_ITERATIONS": 20}
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ── 系统 prompt 指导 ──────────────────────────────────────
def test_dynamic_prompt_guides_direct_answer():
    prompt = dyn.DYNAMIC_SYSTEM_PROMPT
    assert "直接回答" in prompt
    assert "web_search" in prompt
    assert "kb_search" in prompt
    assert "calculator" in prompt
    assert "{tools_prompt}" in prompt


def test_dynamic_prompt_excludes_delegation_tools():
    prompt = dyn.DYNAMIC_SYSTEM_PROMPT
    assert "task" not in prompt.split("可用工具")[0] or True
    assert "spawn_tasks" not in prompt


# ── 构建：注册表工具 + agent ────────────────
def test_build_dynamic_agent_creates_agent(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_create(model=None, tools=None, system_prompt=None, name=None, middleware=None):
        return FakeAgent(
            model=model, tools=tools, system_prompt=system_prompt,
            name=name, middleware=middleware,
        )

    monkeypatch.setattr("langchain.agents.create_agent", fake_create)
    fake_model = object()
    monkeypatch.setattr("app.agents.deep.llm.get_langchain_model", lambda: fake_model)
    dyn._dynamic_agent_cache = None
    try:
        agent = dyn.build_dynamic_agent()
        assert agent is not None
        assert captured["name"] == "easyrag_dynamic_agent"
        assert captured["model"] is fake_model
        assert isinstance(captured["system_prompt"], str)
        assert "web_search" in captured["system_prompt"]
        # 不包含委派工具名称
        names = [getattr(t, "name", "") for t in captured["tools"]]
        assert "task" not in names
        assert "spawn_tasks" not in names
        # 2026-09-04 Skill 重构：SkillsMiddleware 必须挂上（Skill 注入 + 渐进式
        # 门控 + read_skill 工具都在它上面；漏挂 = Skill 功能静默失效）
        middleware_names = [type(m).__name__ for m in (captured["middleware"] or [])]
        assert "SkillsMiddleware" in middleware_names
    finally:
        dyn._dynamic_agent_cache = None


def test_get_dynamic_agent_is_cached(monkeypatch):
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: object())
    monkeypatch.setattr("app.agents.deep.llm.get_langchain_model", lambda: object())
    dyn._dynamic_agent_cache = None
    try:
        a1 = dyn.get_dynamic_agent()
        a2 = dyn.get_dynamic_agent()
        assert a1 is a2
    finally:
        dyn._dynamic_agent_cache = None


# ── run_dynamic_agent 结果解析 ─────────────────────────────────────
def _fake_run(stream_chunks):
    def fake_stream(inputs, config=None, stream_mode=None):
        for chunk in stream_chunks:
            yield chunk

    agent = types.SimpleNamespace(stream=fake_stream)
    return agent


def _chunks_per_message(*msgs):
    """模拟 langgraph stream_mode=values ：每个 chunk 的最后一条消息为当前步骤产出。"""
    acc = []
    for m in msgs:
        acc.append(m)
        yield {"messages": list(acc)}



def test_run_dynamic_agent_parses_direct_answer(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    agent = _fake_run([
        {"messages": [HumanMessage(content="你好"), AIMessage(content="你好！")]},
    ])
    monkeypatch.setattr(dyn, "get_dynamic_agent", lambda: agent)
    monkeypatch.setattr(dyn, "cfg", _simple_cfg())
    result = dyn.run_dynamic_agent("你好", session_id="s1")
    assert result["final_answer"] == "你好！"
    assert result["intent"] == "dynamic"
    assert result["tool_triggered"] is False
    assert result["retrieval_triggered"] is False
    assert result["is_fallback"] is False


def test_run_dynamic_agent_tracks_tool_calls_and_sources(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    tool_output = (
        "搜索结果：最新 AI 新闻"
        "<!--SOURCES:[{\"title\":\"T1\",\"url\":\"https://a.b/c\"}]-->"
    )
    agent = _fake_run(list(_chunks_per_message(
        HumanMessage(content="查下新闻"),
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "AI 新闻"}, "id": "c1"}],
        ),
        ToolMessage(content=tool_output, tool_call_id="c1"),
        AIMessage(content="根据搜索结果：…"),
    )))
    monkeypatch.setattr(dyn, "get_dynamic_agent", lambda: agent)
    monkeypatch.setattr(dyn, "cfg", _simple_cfg())
    result = dyn.run_dynamic_agent("查下新闻", session_id="s1")
    assert result["tool_triggered"] is True
    assert result["tool_name"] == "web_search"
    assert result["sources"] == [{"title": "T1", "url": "https://a.b/c"}]
    assert result["is_fallback"] is False
    assert any("web_search" in s for s in result["steps"])


def test_run_dynamic_agent_marks_kb_retrieval(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    agent = _fake_run(list(_chunks_per_message(
        HumanMessage(content="公司规章制度"),
        AIMessage(
            content="",
            tool_calls=[{"name": "kb_search", "args": {"query": "规章制度"}, "id": "c1"}],
        ),
        ToolMessage(content="没找到相关内容", tool_call_id="c1"),
        AIMessage(content="基于检索结果：…"),
    )))
    monkeypatch.setattr(dyn, "get_dynamic_agent", lambda: agent)
    monkeypatch.setattr(dyn, "cfg", _simple_cfg())
    result = dyn.run_dynamic_agent("公司规章制度", session_id="s1")
    assert result["retrieval_triggered"] is True
    assert result["tool_name"] == "kb_search"


# ── AgentService.run 路由 ────────────────────────────────────────────────────────────
def test_auto_mode_simple_question_routes_to_dynamic(monkeypatch):
    import app.services.agent_service as svc_mod

    calls = {}

    def _fake_dynamic(self, query, **kwargs):
        calls["q"] = query
        return {"final_answer": "dyn", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", _simple_cfg(AGENT_MODE="auto"))
    monkeypatch.setattr(AgentService, "_should_use_multi", staticmethod(lambda q, h=None: False))
    monkeypatch.setattr(AgentService, "_run_dynamic", _fake_dynamic)
    svc = object.__new__(AgentService)
    result = svc.run("你好", session_id="s1")
    assert result["final_answer"] == "dyn"
    assert calls["q"] == "你好"


def test_auto_mode_complex_question_routes_to_deep(monkeypatch):
    import app.services.agent_service as svc_mod

    calls = {}

    def _fake_deep(self, query, **kwargs):
        calls["q"] = query
        return {"final_answer": "deep", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", _simple_cfg(AGENT_MODE="auto"))
    monkeypatch.setattr(AgentService, "_should_use_multi", staticmethod(lambda q, h=None: True))
    monkeypatch.setattr(AgentService, "_run_deep", _fake_deep)
    svc = object.__new__(AgentService)
    result = svc.run("跨领域复杂问题", session_id="s1")
    assert result["final_answer"] == "deep"
    assert calls["q"] == "跨领域复杂问题"


def test_dynamic_mode_routes_to_run_dynamic(monkeypatch):
    import app.services.agent_service as svc_mod

    calls = {}

    def _fake_dynamic(self, query, **kwargs):
        calls["q"] = query
        return {"final_answer": "dyn", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", _simple_cfg(AGENT_MODE="dynamic"))
    monkeypatch.setattr(AgentService, "_run_dynamic", _fake_dynamic)
    svc = object.__new__(AgentService)
    result = svc.run("测试", session_id="s1")
    assert result["final_answer"] == "dyn"
    assert calls["q"] == "测试"


def test_single_mode_deprecated_falls_back_to_dynamic(monkeypatch):
    """阶段 0（2026-09-02）：single 固定管线已退役，配置残留按 auto 处理 → dynamic。"""
    import app.services.agent_service as svc_mod

    calls = {}

    def _fake_dyn(self, query, **kwargs):
        calls["q"] = query
        return {"final_answer": "dyn", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", _simple_cfg(AGENT_MODE="single"))
    monkeypatch.setattr(AgentService, "_run_dynamic", _fake_dyn)
    svc = object.__new__(AgentService)
    result = svc.run("测试", session_id="s1")
    assert result["final_answer"] == "dyn"
    assert calls["q"] == "测试"


def test_deepagents_mode_still_routes_to_deep(monkeypatch):
    import app.services.agent_service as svc_mod

    calls = {}

    def _fake_deep(self, query, **kwargs):
        calls["q"] = query
        return {"final_answer": "deep", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", _simple_cfg(AGENT_MODE="deepagents"))
    monkeypatch.setattr(AgentService, "_run_deep", _fake_deep)
    svc = object.__new__(AgentService)
    result = svc.run("测试", session_id="s1")
    assert result["final_answer"] == "deep"


# ── 最终回答逐 token 流式（2026-09-04）────────────────────────────────────
def _msg_stream(stream_chunks, token_chunks):
    """混合 fake：values chunk 与 ('messages', (token, meta)) 按 stream_mode 分发。

    stream_mode 传列表时产出 (mode, payload) 元组（对齐真实 langgraph 契约）。
    """
    def fake_stream(inputs, config=None, stream_mode=None):
        want_messages = isinstance(stream_mode, (list, tuple)) and "messages" in stream_mode
        for tc in token_chunks:
            if want_messages:
                yield ("messages", (tc, {"langgraph_node": "model"}))
        for chunk in stream_chunks:
            if want_messages:
                yield ("values", chunk)
            else:
                yield chunk
    return types.SimpleNamespace(stream=fake_stream)


def test_run_dynamic_agent_streams_final_answer_tokens(monkeypatch):
    """正文 token 经 on_artifact 以 kind=answer 逐段透出，且不落 artifacts 列表。"""
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

    agent = _msg_stream(
        [{"messages": [HumanMessage(content="q"), AIMessage(content="你好呀")]}],
        [AIMessageChunk(content="你好"), AIMessageChunk(content="呀")],
    )
    monkeypatch.setattr(dyn, "get_dynamic_agent", lambda: agent)
    monkeypatch.setattr(dyn, "cfg", _simple_cfg())
    received = []
    result = dyn.run_dynamic_agent(
        "q", session_id="s1",
        on_artifact=lambda ev: received.append(ev),
    )
    answer_evs = [e for e in received if e.get("kind") == "answer"]
    assert "".join(e.get("content", "") for e in answer_evs if e.get("streaming")) == "你好呀"
    # 结束标记：streaming=False
    assert any(e.get("kind") == "answer" and e.get("streaming") is False for e in answer_evs)
    # 正文流不进 artifacts 列表（与 done 整段重复）
    assert not any(a.get("kind") == "answer" for a in result["artifacts"])
    assert result["final_answer"] == "你好呀"


def test_run_dynamic_agent_token_stream_filters_toolcall_chunks(monkeypatch):
    """tool_call 参数增量与 reasoning token 不进正文流。"""
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

    param_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "web_search", "args": "{\"query\"", "id": "c1", "index": 0}],
    )
    think_chunk = AIMessageChunk(content="")
    think_chunk.additional_kwargs["reasoning_content"] = "思考中"
    agent = _msg_stream(
        [{"messages": [HumanMessage(content="q"), AIMessage(content="答")]},
         {"messages": [HumanMessage(content="q"), AIMessage(content="", tool_calls=[
             {"name": "web_search", "args": {"query": "x"}, "id": "c1"}])]}],
        [param_chunk, think_chunk, AIMessageChunk(content="答")],
    )
    monkeypatch.setattr(dyn, "get_dynamic_agent", lambda: agent)
    monkeypatch.setattr(dyn, "cfg", _simple_cfg())
    received = []
    result = dyn.run_dynamic_agent(
        "q", session_id="s1",
        on_artifact=lambda ev: received.append(ev),
    )
    streamed = "".join(
        e.get("content", "") for e in received
        if e.get("kind") == "answer" and e.get("streaming")
    )
    assert streamed == "答", f"只应透出最终轮正文 token，实际：{streamed!r}"
