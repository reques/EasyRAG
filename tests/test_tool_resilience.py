"""阶段 1：工具层弹性 — 超时 / 重试 / 结构化事件 / contextvar 传播。

全部使用本地 mock 工具（无网络、无外部服务）。
"""
from __future__ import annotations

import time

import pytest

from app.agents.events import use_request_trace
from app.core.exceptions import ToolExecutionError, ToolTimeoutError
from app.tools.registry import ToolDefinition, ToolRegistry


def _flaky_tool(fail_first: int = 0):
    """前 fail_first 次抛 RuntimeError，之后返回 'ok'。"""
    calls = {"n": 0}

    def fn(**kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            raise RuntimeError("transient failure")
        return "ok"

    return fn, calls


# ── 超时 ───────────────────────────────────────────────────────────────────


def test_invoke_times_out_slow_tool():
    """慢工具超过 timeout_s → ToolTimeoutError（不再无限阻塞）。"""
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="slow",
        description="sleeps",
        fn=lambda **kw: time.sleep(0.8) or "done",
        timeout_s=0.2,
    ))
    started = time.perf_counter()
    with pytest.raises(ToolTimeoutError, match="timed out after"):
        reg.invoke("slow")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.7  # 未陪跑到工具自然结束（0.8s）


def test_timeout_error_is_not_retried():
    """超时属于不可重试错误：max_retries=1 也只执行一次。"""
    reg = ToolRegistry()
    fn, calls = _flaky_tool()
    # 第一次尝试就超时（fn 实际很快返回，用极小 timeout 模拟超时路径不可行，
    # 因此这里直接注册一个总是超时的工具）
    def slow(**kw):
        calls["n"] += 1
        time.sleep(0.5)
        return "done"

    reg.register(ToolDefinition(
        name="slow2", description="d", fn=slow, timeout_s=0.1, max_retries=1,
    ))
    with pytest.raises(ToolTimeoutError):
        reg.invoke("slow2")
    assert calls["n"] == 1  # 超时不重试


def test_zero_timeout_executes_directly():
    """timeout_s=0（如 MCP 桥接）不包外层超时，直接执行。"""
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="direct", description="d", fn=lambda **kw: "ok", timeout_s=0,
    ))
    assert reg.invoke("direct") == "ok"


# ── 重试 ───────────────────────────────────────────────────────────────────


def test_transient_failure_retried():
    """瞬时失败 + max_retries=1 → 第二次成功。"""
    reg = ToolRegistry()
    fn, calls = _flaky_tool(fail_first=1)
    reg.register(ToolDefinition(
        name="flaky", description="d", fn=fn, timeout_s=5, max_retries=1,
    ))
    assert reg.invoke("flaky") == "ok"
    assert calls["n"] == 2


def test_exhausted_retries_raise_tool_execution_error():
    """重试耗尽 → 包装为 ToolExecutionError 并携带原始错误。"""
    reg = ToolRegistry()
    fn, calls = _flaky_tool(fail_first=99)
    reg.register(ToolDefinition(
        name="always-fails", description="d", fn=fn, timeout_s=5, max_retries=1,
    ))
    with pytest.raises(ToolExecutionError, match="unexpected error.*transient"):
        reg.invoke("always-fails")
    assert calls["n"] == 2  # 1 次原始 + 1 次重试


def test_no_retry_by_default():
    """默认 max_retries=0：失败一次即抛。"""
    reg = ToolRegistry()
    fn, calls = _flaky_tool(fail_first=99)
    reg.register(ToolDefinition(name="plain", description="d", fn=fn, timeout_s=5))
    with pytest.raises(ToolExecutionError):
        reg.invoke("plain")
    assert calls["n"] == 1


# ── 结构化事件 ─────────────────────────────────────────────────────────────


def test_invoke_emits_tool_events_into_trace():
    """请求 trace 内调用工具 → tool_start/tool_end 事件携带 trace_id 与耗时。"""
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="echo", description="d",
        fn=lambda **kw: f"echo:{kw.get('msg', '')}", timeout_s=5,
    ))
    with use_request_trace(session_id="sess-1") as rt:
        result = reg.invoke("echo", msg="hi")
    assert result == "echo:hi"

    stages = [ev["stage"] for ev in rt.events]
    assert stages == ["tool_start", "tool_end"]
    start_ev, end_ev = rt.events
    assert start_ev["kind"] == "tool" and start_ev["tool"] == "echo"
    assert "hi" in start_ev["content"]  # 参数摘要
    assert end_ev["elapsed_ms"] >= 0
    assert all(ev["trace_id"] == rt.trace.trace_id for ev in rt.events)
    assert all(ev["session_id"] == "sess-1" for ev in rt.events)


def test_invoke_emits_tool_error_event():
    """工具失败 → tool_error 事件包含错误摘要。"""
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="boom", description="d",
        fn=lambda **kw: (_ for _ in ()).throw(RuntimeError("kaboom")),
        timeout_s=5,
    ))
    with use_request_trace() as rt:
        with pytest.raises(ToolExecutionError):
            reg.invoke("boom")
    stages = [ev["stage"] for ev in rt.events]
    assert stages == ["tool_start", "tool_error"]
    assert "kaboom" in rt.events[-1]["content"]


def test_emit_noop_without_trace_context():
    """无 trace/sink/log 上下文时 emit 为 no-op（不抛错）。"""
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="quiet", description="d", fn=lambda **kw: "ok", timeout_s=5,
    ))
    assert reg.invoke("quiet") == "ok"  # 不抛、不产生副作用


# ── contextvar 传播 ────────────────────────────────────────────────────────


def test_timeout_execution_preserves_contextvars():
    """限时执行线程内可见请求级 contextvar（kb_search 授权依赖此行为）。

    ThreadPoolExecutor 不自动复制上下文——registry 显式 copy_context()。
    """
    from app.services.knowledge_context import (
        get_authorised_kb_ids,
        use_authorised_kb_ids,
    )

    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="ctx-probe", description="d",
        fn=lambda **kw: get_authorised_kb_ids(), timeout_s=5,
    ))
    with use_authorised_kb_ids(["kb-1", "kb-2"]):
        assert reg.invoke("ctx-probe") == ["kb-1", "kb-2"]
    # 作用域外恢复
    assert reg.invoke("ctx-probe") is None
