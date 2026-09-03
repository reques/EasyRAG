"""阶段 5a 测试 — 工具进度回调 + OpenTelemetry 可选遥测（no-op / 真实 tracer）。

覆盖：
- registry.invoke 的 progress_callback：事件流转发 + 原回调转发 + 参数隔离
- 不接受 progress_callback 的工具函数不注入该参数
- 回调自身异常不传播
- trace_span no-op（未安装）与真实路径（fake tracer 注入）
- instrument_app / OTelASGIMiddleware（fake tracer）
- task / spawn_tasks 执行路径上的遥测 span
"""
from __future__ import annotations

import asyncio

import pytest

import app.observability.tracing as tracing_mod
from app.observability.tracing import (
    OTEL_AVAILABLE,
    OTelASGIMiddleware,
    instrument_app,
    trace_span,
)
from app.tools.registry import ToolDefinition, ToolRegistry


# ── fixtures ────────────────────────────────────────────────────────────────


class _FakeSpan:
    def __init__(self, name: str, attributes: dict):
        self.name = name
        self.attributes = dict(attributes)
        self.exceptions = []
        self.status = None
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class _FakeTracer:
    def __init__(self):
        self.spans: list[_FakeSpan] = []

    def start_span(self, name, **kwargs):
        span = _FakeSpan(name, kwargs.get("attributes") or {})
        self.spans.append(span)
        return span


@pytest.fixture
def fake_otel(monkeypatch):
    """把 tracing 模块切到"已安装 OTel"状态并注入 fake tracer。"""
    fake = _FakeTracer()

    class _FakeStatus:
        def __init__(self, code, description=""):
            self.code = code
            self.description = description

    monkeypatch.setattr(tracing_mod, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(
        tracing_mod, "_otel_trace",
        type("FakeTraceModule", (), {"get_tracer": staticmethod(lambda name: fake),
                                     "Status": _FakeStatus,
                                     "StatusCode": type("SC", (), {"ERROR": "ERROR"})}),
    )
    return fake


@pytest.fixture
def no_otel(monkeypatch):
    monkeypatch.setattr(tracing_mod, "OTEL_AVAILABLE", False)


def _make_registry() -> ToolRegistry:
    return ToolRegistry()


# ── 进度回调 ────────────────────────────────────────────────────────────────


def test_progress_callback_emits_event_and_forwards():
    from app.agents.events import use_request_trace

    received = []

    def fn(query, progress_callback=None):
        progress_callback("加载数据", percent=50, phase="load")
        return f"done:{query}"

    reg = _make_registry()
    reg.register(ToolDefinition(
        name="prog_tool", description="", fn=fn,
        arg_schema={"query": ("string", "", True)},
        timeout_s=10,
    ))
    with use_request_trace(session_id="conv-prog") as rt:
        out = reg.invoke(
            "prog_tool", query="q",
            progress_callback=lambda msg, percent=None, **extra: received.append(
                (msg, percent, extra)
            ),
        )
    assert out == "done:q"
    assert received == [("加载数据", 50, {"phase": "load"})]
    progress_events = [
        e for e in rt.events if e["kind"] == "tool" and e["stage"] == "progress"
    ]
    assert len(progress_events) == 1
    ev = progress_events[0]
    assert ev["tool"] == "prog_tool"
    assert ev["percent"] == 50
    assert "加载数据" in ev["content"]


def test_progress_callback_not_in_tool_kwargs_or_digest():
    from app.agents.events import use_request_trace

    seen = {}

    def fn(**kwargs):  # **kwargs 接受 progress_callback
        seen.update(kwargs)
        return "ok"

    reg = _make_registry()
    reg.register(ToolDefinition(name="kwargs_tool", description="", fn=fn, timeout_s=10))
    with use_request_trace() as rt:
        reg.invoke("kwargs_tool", a=1, progress_callback=lambda *args, **kw: None)
    # 工具收到的是包装后的回调，不是原回调；业务参数原样保留
    assert "a" in seen and seen["a"] == 1
    assert callable(seen["progress_callback"])
    # tool_start 摘要不含 progress_callback（不泄露调用方回调/日志干净）
    start = next(e for e in rt.events if e["stage"] == "tool_start")
    assert "progress_callback" not in start["content"]


def test_progress_callback_dropped_for_fn_without_param(no_otel):
    seen = {}

    def fn(query):  # 签名无 progress_callback，也不接受 **kwargs
        seen["query"] = query
        return "ok"

    reg = _make_registry()
    reg.register(ToolDefinition(
        name="no_prog", description="", fn=fn,
        arg_schema={"query": ("string", "", True)},
        timeout_s=0,  # 直调路径
    ))
    assert reg.invoke("no_prog", query="x", progress_callback=lambda *a: None) == "ok"
    assert seen == {"query": "x"}


def test_progress_callback_exception_swallowed():
    def fn(progress_callback=None):
        progress_callback("step1")
        return "ok"

    def bad_callback(*args, **kwargs):
        raise RuntimeError("sink boom")

    reg = _make_registry()
    reg.register(ToolDefinition(name="boom_cb", description="", fn=fn, timeout_s=10))
    # 回调异常不传播：工具仍正常返回
    assert reg.invoke("boom_cb", progress_callback=bad_callback) == "ok"


# ── trace_span：no-op 与真实路径 ────────────────────────────────────────────


def test_trace_span_noop_when_otel_missing(no_otel):
    with trace_span("any.span", foo="bar") as span:
        assert span is None


def test_trace_span_uses_fake_tracer(fake_otel):
    with trace_span("unit.span", tool="x") as span:
        span.set_attribute("k", "v")
    assert [s.name for s in fake_otel.spans] == ["unit.span"]
    s = fake_otel.spans[0]
    assert s.attributes["tool"] == "x" and s.attributes["k"] == "v"
    assert s.ended and not s.exceptions


def test_trace_span_records_exception(fake_otel):
    with pytest.raises(ValueError):
        with trace_span("unit.err") as span:
            raise ValueError("boom")
    s = fake_otel.spans[0]
    assert s.ended and len(s.exceptions) == 1
    assert s.status is not None


def test_registry_invoke_creates_span(fake_otel):
    reg = _make_registry()
    reg.register(ToolDefinition(
        name="spanned", description="", fn=lambda: "ok", timeout_s=0,
    ))
    assert reg.invoke("spanned") == "ok"
    assert any(s.name == "tool.invoke.spanned" for s in fake_otel.spans)


def test_registry_invoke_noop_without_otel(no_otel):
    reg = _make_registry()
    reg.register(ToolDefinition(
        name="plain", description="", fn=lambda: "ok", timeout_s=0,
    ))
    assert reg.invoke("plain") == "ok"  # 未安装时照常工作


# ── ASGI 中间件 ─────────────────────────────────────────────────────────────


def test_instrument_app_noop_without_otel(no_otel):
    class FakeApp:
        user_middleware = []

        def add_middleware(self, cls, **kw):
            self.user_middleware.append(cls)

    app = FakeApp()
    assert instrument_app(app) is app
    assert app.user_middleware == []


def test_instrument_app_adds_middleware_with_otel(fake_otel):
    class FakeApp:
        user_middleware = []

        def add_middleware(self, cls, **kw):
            self.user_middleware.append(cls)

    app = FakeApp()
    instrument_app(app)
    assert app.user_middleware == [OTelASGIMiddleware]


def test_asgi_middleware_root_span(fake_otel):
    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = OTelASGIMiddleware(inner_app)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(
        {"type": "http", "method": "GET", "path": "/api/v1/health"},
        receive, send,
    ))
    spans = [s for s in fake_otel.spans if s.name.startswith("GET ")]
    assert len(spans) == 1
    assert spans[0].attributes["http.method"] == "GET"
    assert spans[0].attributes["http.path"] == "/api/v1/health"
    assert spans[0].attributes["http.status_code"] == 204


def test_asgi_middleware_ignores_non_http(fake_otel):
    called = []

    async def inner_app(scope, receive, send):
        called.append(scope["type"])

    asyncio.run(OTelASGIMiddleware(inner_app)(
        {"type": "lifespan"}, None, None,
    ))
    assert called == ["lifespan"]
    assert fake_otel.spans == []


# ── task / spawn_tasks 执行路径的 span ──────────────────────────────────────


def test_task_tool_creates_subagent_span(fake_otel, monkeypatch):
    from app.agents.deep import task_tool as tt

    monkeypatch.setattr(tt, "run_subagent", lambda *a, **kw: "sub-result")
    tool = tt.build_task_tool()
    out = tool.invoke({"description": "d", "subagent_type": "research-agent"})
    assert out == "sub-result"
    names = [s.name for s in fake_otel.spans]
    assert "subagent.research-agent" in names


def test_spawn_tasks_creates_spans(fake_otel, monkeypatch):
    from app.agents.deep import planner
    from app.agents.deep import subagents
    from app.agents.deep.task_tool import reset_task_breaker

    reset_task_breaker()
    planner.reset_plans()
    monkeypatch.setattr(subagents, "run_subagent", lambda *a, **kw: "ok")
    specs = planner.parse_task_inputs([
        {"key": "a", "description": "do a", "subagent_type": "research-agent"},
        {"key": "b", "description": "do b", "subagent_type": "research-agent",
         "depends_on": ["a"]},
    ])
    results = planner.run_spawn_tasks(specs)
    assert results["a"].status == planner.STATUS_OK
    assert results["b"].status == planner.STATUS_OK
    names = [s.name for s in fake_otel.spans]
    assert "spawn_tasks" in names
    assert "spawn_task.a" in names and "spawn_task.b" in names
