"""Request-scoped event bus and canonical execution trace events.

The public contract is independent from LangGraph. Existing
``kind/stage/title/content`` fields remain so older observers keep working,
while persistence and the UI consume:
``id, parent_id, type, status, timestamp, input, output, metadata``.
"""
from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

EVENT_TYPES = frozenset({
    "agent_start", "planning", "reasoning_summary", "tool_call", "tool_result",
    "file_operation", "code_execution", "error", "final_response",
})
_RESERVED_KEYS = frozenset({
    "id", "parent_id", "type", "status", "timestamp", "input", "output",
    "metadata", "trace_id", "span", "session_id", "kind", "stage", "title",
    "content", "ts",
})
_FILE_TOOLS = frozenset({
    "read_file", "write_file", "edit_file", "delete_file", "move_file",
    "copy_file", "search_file", "list_files", "apply_patch",
})
_CODE_TOOLS = frozenset({
    "execute_python", "execute_code", "execute_shell", "run_command", "shell",
    "terminal", "python",
})


@dataclass(frozen=True)
class TraceInfo:
    trace_id: str
    span: str = "main"
    session_id: str = ""


@dataclass
class RequestTrace:
    trace: TraceInfo
    events: List[Dict[str, Any]]


_current_trace: ContextVar[Optional[TraceInfo]] = ContextVar("agent_current_trace", default=None)
_event_sinks: ContextVar[tuple] = ContextVar("agent_event_sinks", default=())
_event_log: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("agent_event_log", default=None)
_trace_root: ContextVar[str] = ContextVar("agent_trace_root", default="")
_open_events: ContextVar[Optional[Dict[str, str]]] = ContextVar("agent_open_events", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def get_trace() -> Optional[TraceInfo]:
    return _current_trace.get()


def get_trace_id() -> str:
    trace = _current_trace.get()
    return trace.trace_id if trace else ""


@contextmanager
def use_request_trace(session_id: str = "", span: str = "main") -> Iterator[RequestTrace]:
    """Create one in-memory trace. Sinks stream events during execution."""
    trace = TraceInfo(trace_id=new_trace_id(), span=span, session_id=session_id)
    log: List[Dict[str, Any]] = []
    trace_token = _current_trace.set(trace)
    log_token = _event_log.set(log)
    root_token = _trace_root.set("")
    open_token = _open_events.set({})
    try:
        yield RequestTrace(trace=trace, events=log)
    finally:
        _open_events.reset(open_token)
        _trace_root.reset(root_token)
        _event_log.reset(log_token)
        _current_trace.reset(trace_token)


@contextmanager
def use_span(span: str) -> Iterator[None]:
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
    token = _event_sinks.set(_event_sinks.get() + (sink,))
    try:
        yield
    finally:
        _event_sinks.reset(token)


def _tool_event_type(tool: str, ending: bool) -> str:
    name = (tool or "").lower()
    if name in _FILE_TOOLS or any(part in name for part in ("file", "filesystem")):
        return "file_operation"
    if name in _CODE_TOOLS or any(part in name for part in ("python", "execute_code", "shell")):
        return "code_execution"
    return "tool_result" if ending else "tool_call"


def _canonical_type(kind: str, stage: str, artifact_kind: str, tool: str) -> str:
    if kind in EVENT_TYPES:
        return kind
    if kind == "agent":
        return "agent_start" if stage == "agent_start" else "final_response"
    if kind == "tool":
        return _tool_event_type(tool, stage in {"tool_end", "tool_error"})
    if kind == "artifact":
        if artifact_kind == "thought":
            return "reasoning_summary"
        if artifact_kind in {"tool", "delegate"}:
            return _tool_event_type(tool, False)
        if artifact_kind == "tool_result":
            return _tool_event_type(tool, True)
    if kind == "error" or stage in {"error", "tool_error", "task_error", "fallback"}:
        return "error"
    return "planning"


def _canonical_status(event_type: str, stage: str, explicit: Any) -> str:
    if explicit:
        return str(explicit)
    if event_type == "error" or stage.endswith("_error"):
        return "error"
    if stage.endswith("_start") or stage in {"progress", "understand", "reason", "tool", "generate"}:
        return "running"
    if stage.endswith("_end") or stage.endswith("_done") or event_type == "final_response":
        return "completed"
    return "info"


def _parent_for(event_type: str, stage: str, tool: str, event_id: str) -> str | None:
    opened = _open_events.get()
    root = _trace_root.get()
    span = get_trace().span if get_trace() else ""
    key = f"{span}:tool:{tool}"
    span_key = f"{span}:root"
    if event_type == "agent_start":
        _trace_root.set(event_id)
        return None
    if opened is not None and stage == "task_start":
        opened[span_key] = event_id
        return root or None
    if opened is not None and stage == "tool_start":
        opened[key] = event_id
        return opened.get(span_key) or root or None
    if opened is not None and stage in {"tool_end", "tool_error"}:
        return opened.pop(key, None) or opened.get(span_key) or root or None
    if opened is not None and stage in {"task_end", "task_error", "task_skip"}:
        return opened.pop(span_key, None) or root or None
    return (opened or {}).get(span_key) or root or None


def emit(kind: str, stage: str, title: str, content: str = "", **extra: Any) -> Dict[str, Any] | None:
    """Emit one event; no active trace/sink remains a zero-cost no-op."""
    trace = _current_trace.get()
    sinks = _event_sinks.get()
    log = _event_log.get()
    if trace is None and not sinks and log is None:
        return None

    artifact_kind = str(extra.get("artifact_kind") or "")
    tool = str(extra.get("tool") or "")
    event_type = _canonical_type(kind, stage, artifact_kind, tool)
    event_id = uuid.uuid4().hex
    parent_id = extra.get("parent_id") or _parent_for(event_type, stage, tool, event_id)
    status = _canonical_status(event_type, stage, extra.get("status"))
    timestamp = datetime.now(timezone.utc).isoformat()
    supplied_input = extra.get("input")
    supplied_output = extra.get("output")
    is_finished = stage in {"tool_end", "tool_error"}
    if supplied_input is None and (
        event_type in {"agent_start", "tool_call"}
        or (event_type in {"file_operation", "code_execution"} and not is_finished)
    ):
        supplied_input = content or None
    if supplied_output is None and (
        event_type in {"planning", "reasoning_summary", "tool_result", "final_response", "error"}
        or (event_type in {"file_operation", "code_execution"} and is_finished)
    ):
        supplied_output = content or None
    metadata = {
        "span": trace.span if trace else "",
        "session_id": trace.session_id if trace else "",
        "legacy_kind": kind,
        "stage": stage,
        "title": str(title)[:80],
    }
    for key, value in extra.items():
        if key not in _RESERVED_KEYS and key != "artifact_kind":
            metadata[key] = value
    if artifact_kind:
        metadata["artifact_kind"] = artifact_kind
    if extra.get("id"):
        metadata["source_event_id"] = str(extra["id"])

    event: Dict[str, Any] = {
        "id": event_id,
        "parent_id": parent_id,
        "type": event_type,
        "status": status,
        "timestamp": timestamp,
        "input": supplied_input,
        "output": supplied_output,
        "metadata": metadata,
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
            pass
    return event


def public_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Project an event onto the stable SSE/API contract."""
    return {
        key: event.get(key)
        for key in ("id", "parent_id", "type", "status", "timestamp", "input", "output", "metadata", "trace_id")
    }


def token_usage_from_message(message: Any) -> Dict[str, int]:
    """Normalize LangChain/OpenAI usage metadata across providers."""
    usage = getattr(message, "usage_metadata", None) or {}
    response = getattr(message, "response_metadata", None) or {}
    if not usage:
        usage = response.get("token_usage") or response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def add_token_usage(total: Dict[str, int], message: Any, seen: set[str]) -> None:
    """Add usage once per AI message (stream and value modes may repeat it)."""
    usage = token_usage_from_message(message)
    if not any(usage.values()):
        return
    key = str(getattr(message, "id", "") or id(message))
    if key in seen:
        return
    seen.add(key)
    for name, value in usage.items():
        total[name] = int(total.get(name, 0)) + value


def snapshot_request_context() -> "contextvars.Context":
    return contextvars.copy_context()


def run_with_request_context(
    context: "contextvars.Context", fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    return context.run(fn, *args, **kwargs)
