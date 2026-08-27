"""DeepAgents 子 Agent 步骤透传（S3）测试。

2026-08-21 修复：task 委派的 SubAgent 执行过程此前是黑盒（前端只能看到
"调用 task(...)"）。本次新增请求级观察者（app/agents/deep/observe.py）：
_run_deep 设置 task 观察者 → task 工具转发为子 Agent 观察者 → run_subagent
的 stream 循环把子 Agent 的推理/工具调用/工具返回透传 SSE（带子 Agent 名前缀）。

全部用 mock（不调用真实 LLM）。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage


def _fake_subagent_agent(chunks):
    class _FakeAgent:
        def stream(self, inputs, config=None, stream_mode="values"):
            for chunk in chunks:
                yield chunk

    return _FakeAgent()


# ── 观察者 ContextVar 设置/恢复 ───────────────────────────────────────────
def test_observers_scope_set_and_restore():
    from app.agents.deep.observe import (
        get_subagent_observers,
        get_task_observers,
        use_subagent_observers,
        use_task_observers,
    )

    assert get_task_observers() is None
    assert get_subagent_observers() is None
    step = lambda s, d: None  # noqa: E731
    with use_task_observers(step):
        assert get_task_observers() == (step, None)
        with use_subagent_observers(step, step):
            assert get_subagent_observers() == (step, step)
        assert get_subagent_observers() is None
    assert get_task_observers() is None


# ── run_subagent 观察者透传 ───────────────────────────────────────────────
def test_run_subagent_passes_through_steps(monkeypatch):
    from app.agents.deep.observe import use_subagent_observers
    from app.agents.deep.subagents import SubAgentConfig, run_subagent

    cfg = SubAgentConfig(name="research-agent", description="d",
                         system_prompt="p", tools=("web_search",))

    agent = _fake_subagent_agent([
        {"messages": [AIMessage(
            content="先搜一下",
            tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "x"}}],
        )]},
        {"messages": [ToolMessage(content="检索结果", tool_call_id="call_1")]},
        {"messages": [AIMessage(content="最终研究结论")]},
    ])
    monkeypatch.setattr("app.agents.deep.subagents.build_subagent",
                        lambda cfg_, model=None: agent)

    steps, artifacts = [], []
    with use_subagent_observers(
        lambda s, d: steps.append((s, d)),
        lambda k, st, t, c: artifacts.append((k, st, t, c)),
    ):
        out = run_subagent(cfg, "查资料")

    assert out == "最终研究结论"
    # act and reasoning：思考内容独立成 reason 步骤 + 工具调用带参数
    assert ("research-agent/reason", "先搜一下") in steps
    assert ("research-agent/tool", '调用 web_search {"query": "x"}') in steps
    assert any(s == "research-agent/tool_done" for s, _ in steps)
    assert ("research-agent/generate", "子智能体生成回答中...") in steps
    thought_artifacts = [a for a in artifacts if a[0] == "thought"]
    assert thought_artifacts and thought_artifacts[0][1] == "research-agent/reason"
    tool_artifacts = [a for a in artifacts if a[0] == "tool"]
    assert tool_artifacts and tool_artifacts[0][1] == "research-agent/tool"
    assert '{"query": "x"}' in tool_artifacts[0][3]
    tool_result_artifacts = [a for a in artifacts if a[0] == "tool_result"]
    assert tool_result_artifacts and tool_result_artifacts[0][2] == "工具返回"
    assert "检索结果" in tool_result_artifacts[0][3]


def test_run_subagent_without_observers_unchanged(monkeypatch):
    from app.agents.deep.subagents import SubAgentConfig, run_subagent

    cfg = SubAgentConfig(name="coding-agent", description="d",
                         system_prompt="p", tools=("calculator",))
    agent = _fake_subagent_agent([
        {"messages": [AIMessage(content="完成", tool_calls=[])]},
    ])
    monkeypatch.setattr("app.agents.deep.subagents.build_subagent",
                        lambda cfg_, model=None: agent)
    assert run_subagent(cfg, "算一下") == "完成"


# ── task 工具观察者转发链路 ───────────────────────────────────────────────
def test_task_forwards_observers_when_present(monkeypatch):
    import app.agents.deep.task_tool as tt

    seen: dict = {"sub_obs": "not-called"}

    def _fake_run(cfg, desc, model=None, recursion_limit=None):
        from app.agents.deep.observe import get_subagent_observers
        seen["sub_obs"] = get_subagent_observers()
        return "子结果"

    monkeypatch.setattr("app.agents.deep.task_tool.run_subagent", _fake_run)

    from app.agents.deep.observe import use_task_observers

    tool = tt.build_task_tool(model=object())
    with use_task_observers(lambda s, d: None):
        tool.invoke({"description": "x", "subagent_type": "research-agent"})
    assert seen["sub_obs"] is not None, "有 task 观察者时子 Agent 观察者应被设置"


def test_task_no_observers_no_forward(monkeypatch):
    import app.agents.deep.task_tool as tt

    seen: dict = {"sub_obs": "not-called"}

    def _fake_run(cfg, desc, model=None, recursion_limit=None):
        from app.agents.deep.observe import get_subagent_observers
        seen["sub_obs"] = get_subagent_observers()
        return "子结果"

    monkeypatch.setattr("app.agents.deep.task_tool.run_subagent", _fake_run)
    tool = tt.build_task_tool(model=object())
    tool.invoke({"description": "x", "subagent_type": "research-agent"})
    assert seen["sub_obs"] is None, "无 task 观察者时不应设置子 Agent 观察者"


# ── _run_deep 端到端：委派过程可见 ────────────────────────────────────────
def test_run_deep_delegation_steps_visible(monkeypatch):
    import app.agents.deep.task_tool as tt
    from app.services.agent_service import AgentService, SessionStore

    def _fake_run(cfg, desc, model=None, recursion_limit=None):
        from app.agents.deep.observe import get_subagent_observers
        obs = get_subagent_observers()
        if obs:
            obs[0](f"{cfg.name}/tool", "调用 web_search(...)")
            obs[0](f"{cfg.name}/generate", "子智能体生成回答中...")
        return "研究结果：消费法规定"

    monkeypatch.setattr("app.agents.deep.task_tool.run_subagent", _fake_run)
    tool = tt.build_task_tool(model=object())

    class _FakeMainAgent:
        def stream(self, inputs, config=None, stream_mode="values"):
            # 模拟主 Agent ReAct：先委派 task，再基于结果回答
            out = tool.invoke({"description": "查消费法", "subagent_type": "research-agent"})
            yield {"messages": [AIMessage(
                content="委派给研究助理",
                tool_calls=[{"id": "task_1", "name": "task",
                             "args": {"description": "查消费法",
                                      "subagent_type": "research-agent"}}],
            )]}
            yield {"messages": [ToolMessage(content=out, tool_call_id="task_1")]}
            yield {"messages": [AIMessage(content="根据研究结果：" + out)]}

    monkeypatch.setattr("app.agents.deep.agent.get_main_agent", lambda: _FakeMainAgent())

    svc = object.__new__(AgentService)
    svc._sessions = SessionStore(ttl=3600)
    result = svc._run_deep("消费法有什么规定", history=[], user_id=None,
                           knowledge_base_ids=None)

    steps_text = "\n".join(result["steps"])
    assert "research-agent/tool: 调用 web_search(...)" in steps_text
    assert "research-agent/generate: 子智能体生成回答中..." in steps_text
    assert result["final_answer"] == "根据研究结果：研究结果：消费法规定"
