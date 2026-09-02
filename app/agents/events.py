"""统一事件流 — 请求级 trace 上下文与结构化事件分发（2026-08-26，阶段 1）。

背景：智能体执行过程的中间上报此前散落在三套机制里——``_run_deep`` 的
_step/_artifact 闭包、observe.py 的双层观察者、registry 无任何事件。工具
调用的参数/结果/耗时没有统一的结构化记录，跨层（主 Agent → SubAgent →
工具）无法用同一 trace 串联，也无法可靠回放一次执行。

本模块提供进程内等价的"事件总线"（单进程部署，不引入 MQ）：

- ``use_request_trace``：请求级 trace（trace_id + session_id + span），
  作用域内所有 ``emit`` 进入同一事件列表（随响应返回；持久化后续阶段接入）；
- ``use_span``：同请求内切换 span（如 ``subagent/<name>``），trace_id 保持
  不变——跨线程（executor）执行时必须用它重放 contextvar（contextvars 不随
  线程自动传播）；
- ``use_event_sink``：注册事件消费者（如 SSE 适配器），可叠加；
- ``emit(kind, stage, title, content, **extra)``：统一事件入口。无 trace、
  无 sink、无日志上下文时为 no-op（测试/脚本直接调工具零开销）。

事件字段约定：
    kind    事件类别（step / artifact / tool / delegation / plan / ...）
    stage   阶段标识（tool_start / tool_end / tool_error / <step 名> ...）
    title   短标题（展示用，自动截断 80 字符）
    content 正文（思考摘要 / 工具参数摘要 / 错误信息等）
    extra   附加字段（elapsed_ms / artifact_kind / tool 等）
"""
from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterator, List, Optional

# emit 的 **extra 不允许覆盖的保留字段
_RESERVED_KEYS = frozenset(
    {"trace_id", "span", "session_id", "kind", "stage", "title", "content", "ts"}
)


@dataclass(frozen=True)
class TraceInfo:
    """请求级追踪信息。span 标识当前执行层（main / subagent/<name> / ...）。"""

    trace_id: str
    span: str = "main"
    session_id: str = ""


@dataclass
class RequestTrace:
    """一次请求的 trace + 事件日志（随响应返回，供诊断/前端消费）。"""

    trace: TraceInfo
    events: List[Dict[str, Any]]


_current_trace: ContextVar[Optional[TraceInfo]] = ContextVar(
    "agent_current_trace", default=None
)
_event_sinks: ContextVar[tuple] = ContextVar("agent_event_sinks", default=())
_event_log: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "agent_event_log", default=None
)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def get_trace() -> Optional[TraceInfo]:
    return _current_trace.get()


def get_trace_id() -> str:
    trace = _current_trace.get()
    return trace.trace_id if trace else ""


@contextmanager
def use_request_trace(session_id: str = "", span: str = "main") -> Iterator[RequestTrace]:
    """建立请求级 trace + 事件日志；作用域内所有 emit 进入同一事件流。"""
    trace = TraceInfo(trace_id=new_trace_id(), span=span, session_id=session_id)
    log: List[Dict[str, Any]] = []
    trace_token = _current_trace.set(trace)
    log_token = _event_log.set(log)
    try:
        yield RequestTrace(trace=trace, events=log)
    finally:
        _event_log.reset(log_token)
        _current_trace.reset(trace_token)


@contextmanager
def use_span(span: str) -> Iterator[None]:
    """同请求内切换 span（子层标识），trace_id/session_id 不变。

    跨线程（executor）执行时必须重新进入本上下文，否则子线程看不到
    trace/事件日志（contextvars 不随线程传播）。无外层 trace 时创建匿名
    trace，保证事件不丢（测试/脚本场景）。
    """
    parent = _current_trace.get()
    if parent is None:
        parent = TraceInfo(trace_id=new_trace_id(), span=span)
    token = _current_trace.set(replace(parent, span=span))
    try:
        yield
    finally:
        _current_trace.reset(token)


@contextmanager
def use_event_sink(sink: Callable[[Dict[str, Any]], None]) -> Iterator[None]:
    """注册事件消费者（如 SSE 适配器）；可叠加，退出时恢复。"""
    token = _event_sinks.set(_event_sinks.get() + (sink,))
    try:
        yield
    finally:
        _event_sinks.reset(token)


def emit(kind: str, stage: str, title: str, content: str = "", **extra: Any) -> None:
    """发出一条结构化事件（无 trace/sink/log 上下文时为 no-op）。"""
    trace = _current_trace.get()
    sinks = _event_sinks.get()
    log = _event_log.get()
    if trace is None and not sinks and log is None:
        return
    event: Dict[str, Any] = {
        "trace_id": trace.trace_id if trace else "",
        "span": trace.span if trace else "",
        "session_id": trace.session_id if trace else "",
        "kind": kind,
        "stage": stage,
        "title": str(title)[:80],
        "content": content or "",
        "ts": round(time.time(), 3),
    }
    for key, value in extra.items():
        if key not in _RESERVED_KEYS:
            event[key] = value
    if log is not None:
        log.append(event)
    for sink in sinks:
        try:
            sink(dict(event))
        except Exception:
            # 观察者故障不影响主流程
            pass


# ── 并发点的请求上下文重放（2026-08-26，阶段 3）───────────────────


def snapshot_request_context() -> "contextvars.Context":
    """在调度点（主线程）捕获请求级上下文快照。

    ThreadPoolExecutor 不自动传播 contextvars：trace/事件日志、skills 白名单、
    KB 授权、chat model 选择等全部丢失。每个并发任务需要各自一份快照（同一
    Context 对象不能被多线程同时 enter），调度点可多次调用本函数。
    """
    return contextvars.copy_context()


def run_with_request_context(
    context: "contextvars.Context", fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """在工作线程内用捕获的快照重放请求上下文执行 fn（消除手工重放）。"""
    return context.run(fn, *args, **kwargs)
