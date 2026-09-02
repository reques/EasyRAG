"""阶段 1：DeepAgents 委派熔断（S5）— 连续失败计数 / 成功清零 / TTL 过期。

用 monkeypatch 替换 run_subagent，不调用真实 LLM。
"""
from __future__ import annotations

import time

import pytest

import app.agents.deep.task_tool as tt


@pytest.fixture(autouse=True)
def _clean_breaker():
    tt.reset_task_breaker()
    yield
    tt.reset_task_breaker()


def _build_tool(monkeypatch, outcomes):
    """outcomes: 每次委派的结果（str）或异常（Exception 实例）。"""
    seq = list(outcomes)

    def _fake_run(cfg, desc, model=None, recursion_limit=None):
        if not seq:
            raise AssertionError("unexpected extra delegation")
        out = seq.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(tt, "run_subagent", _fake_run)
    return tt.build_task_tool(model=object())


def test_consecutive_failures_trip_breaker(monkeypatch):
    """连续 3 次失败 → 第 4 次委派被熔断拒绝（不再执行子智能体）。"""
    boom = RuntimeError("子智能体崩溃")
    tool = _build_tool(monkeypatch, [boom, boom, boom])  # 只有 3 个结果

    for i in range(3):
        out = tool.invoke({"description": f"任务{i}", "subagent_type": "research-agent"})
        assert "子智能体执行失败" in out

    # 第 4 次：熔断打开，直接返回提示（run_subagent 不再被调用）
    out = tool.invoke({"description": "任务4", "subagent_type": "research-agent"})
    assert "熔断" in out
    assert "连续失败 3 次" in out


def test_success_resets_failure_counter(monkeypatch):
    """失败 2 次 + 成功 1 次 → 计数清零，再失败 1 次仍不熔断且可再次成功。"""
    boom = RuntimeError("flaky")
    tool = _build_tool(
        monkeypatch,
        [boom, boom, "成功结果", boom, "最终成功"],
    )
    assert "子智能体执行失败" in tool.invoke(
        {"description": "a", "subagent_type": "research-agent"})
    assert "子智能体执行失败" in tool.invoke(
        {"description": "b", "subagent_type": "research-agent"})
    assert tool.invoke({"description": "c", "subagent_type": "research-agent"}) == "成功结果"
    # 清零后：再失败 1 次（计数 1，不熔断），随后成功
    out_fail = tool.invoke({"description": "d", "subagent_type": "research-agent"})
    assert "熔断" not in out_fail
    assert "子智能体执行失败" in out_fail
    out = tool.invoke({"description": "e", "subagent_type": "research-agent"})
    assert "熔断" not in out
    assert out == "最终成功"


def test_breaker_isolated_by_subagent_type(monkeypatch):
    """A 子智能体熔断不影响 B 子智能体。"""
    boom = RuntimeError("boom")
    tool = _build_tool(monkeypatch, [boom, boom, boom, "B 的结果"])
    for i in range(3):
        tool.invoke({"description": f"t{i}", "subagent_type": "research-agent"})
    tripped = tool.invoke({"description": "t4", "subagent_type": "research-agent"})
    assert "熔断" in tripped
    # coding-agent 未失败，正常执行
    assert tool.invoke({"description": "t5", "subagent_type": "coding-agent"}) == "B 的结果"


def test_breaker_entry_expires_after_ttl(monkeypatch):
    """条目超过 TTL 无活动 → 自动过期，熔断解除。"""
    boom = RuntimeError("boom")
    tool = _build_tool(monkeypatch, [boom, boom, boom, "恢复后的结果"])
    for i in range(3):
        tool.invoke({"description": f"t{i}", "subagent_type": "research-agent"})
    assert "熔断" in tool.invoke({"description": "t4", "subagent_type": "research-agent"})

    # 回拨时间：所有条目 last_ts 拨回 TTL 之外
    now = time.time()
    for entry in tt._task_failures.values():
        entry["last_ts"] = now - tt.TASK_BREAKER_TTL_S - 1
    out = tool.invoke({"description": "t5", "subagent_type": "research-agent"})
    assert out == "恢复后的结果"


def test_breaker_scoped_by_session(monkeypatch):
    """不同会话的失败互不影响（session_id 来自请求级 trace）。"""
    from app.agents.events import use_request_trace

    boom = RuntimeError("boom")
    tool = _build_tool(monkeypatch, [boom, boom, boom, "另一会话的结果"])
    with use_request_trace(session_id="conv-A"):
        for i in range(3):
            tool.invoke({"description": f"t{i}", "subagent_type": "research-agent"})
        assert "熔断" in tool.invoke(
            {"description": "t4", "subagent_type": "research-agent"})
    # 会话 B：无失败记录，正常执行
    with use_request_trace(session_id="conv-B"):
        out = tool.invoke({"description": "t5", "subagent_type": "research-agent"})
    assert out == "另一会话的结果"


def test_delegation_events_emitted(monkeypatch):
    """委派过程发结构化事件（delegation task_start/task_end）。"""
    from app.agents.events import use_request_trace

    tool = _build_tool(monkeypatch, ["研究结果"])
    with use_request_trace(session_id="conv-E") as rt:
        tool.invoke({"description": "研究 X", "subagent_type": "research-agent"})
    kinds = [(ev["kind"], ev["stage"]) for ev in rt.events]
    assert ("delegation", "task_start") in kinds
    assert ("delegation", "task_end") in kinds
    start_ev = next(ev for ev in rt.events if ev["stage"] == "task_start")
    assert start_ev["span"] == "main"
    assert "研究 X" in start_ev["content"]
