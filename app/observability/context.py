"""Execution-owned observation handles, propagated with ContextVar snapshots."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import threading
import time
import uuid

from app.observability import redaction


@dataclass
class ExecutionContext:
    execution_id: str
    trace_id: str
    local_trace_id: str = ""
    user_id: str = ""
    session_id: str = ""
    sampled: bool = False
    metadata: dict = field(default_factory=dict)
    handles: set = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
    closed: bool = False
    model_callback: object | None = field(default=None, repr=False)


_execution: ContextVar[ExecutionContext | None] = ContextVar("observability_execution", default=None)
_current: ContextVar[Observation | None] = ContextVar("observability_parent", default=None)


def execution_context():
    return _execution.get()


def current_observation():
    return _current.get()


class Observation:
    def __init__(self, name, *, kind="span", model=None, input=None, metadata=None):
        self.id = uuid.uuid4().hex[:16]
        self.raw = None
        self.ended = False
        self.status = "running"
        self.started = time.perf_counter()
        self.first_output = self.first_content = False
        self.execution = _execution.get()
        self.lock = threading.RLock()
        parent = _current.get()
        inherited = {key: value for key, value in (parent.metadata if parent else {}).items()
                     if key in {"agentId", "agentInvocationId", "taskId"}}
        self.metadata = {**inherited, **redaction.metadata(metadata)}
        context = self.execution
        if context is None:
            return
        with context.lock:
            if context.closed:
                return
            context.handles.add(self)
        if not context.sampled:
            return
        if parent is not None and (parent.raw is None or parent.ended):
            # Drop this subtree if its parent could not be recorded. Retrying
            # as a new SDK root would produce a misleading second root.
            return
        try:
            from app.observability.langfuse_client import get_client
            client = get_client()
            if client is None:
                return
            parent = _current.get()
            # Passing trace_context to every SDK call marks every observation as
            # a root in SDK 5. Use the parent's public factory for children.
            factory = parent.raw.start_observation if parent and parent.raw and not parent.ended else client.start_observation
            trace_args = {} if parent and parent.raw and not parent.ended else {"trace_context": {"trace_id": context.trace_id}}
            self.raw = factory(
                name=redaction.redact(name, 120), as_type=kind, **trace_args,
                model=model, input=redaction.content(input),
                metadata={**redaction.metadata(context.metadata),
                          "executionId": context.execution_id, "localTraceId": context.local_trace_id,
                          **self.metadata},
            )
            self.id = self.raw.id
        except Exception:
            self.raw = None

    def update(self, *, metadata=None, **values):
        with self.lock:
            if self.ended:
                return
            self.metadata.update(redaction.metadata(metadata))
            if self.raw is None:
                return
            try:
                if "input" in values:
                    values["input"] = redaction.content(values["input"])
                if "output" in values:
                    values["output"] = redaction.content(values["output"])
                self.raw.update(metadata=self.metadata, **values)
            except Exception:
                pass

    def token(self, *, content=False):
        with self.lock:
            if self.ended:
                return
            now = time.perf_counter()
            details = {}
            if not self.first_output:
                self.first_output = True
                details["firstOutputMs"] = round((now - self.started) * 1000, 2)
            if content and not self.first_content:
                self.first_content = True
                details["firstContentMs"] = round((now - self.started) * 1000, 2)
                self.update(completion_start_time=datetime.now(timezone.utc))
            if details:
                self.update(metadata=details)

    def error(self, exc=None, *, status="error", reason=None):
        with self.lock:
            if self.ended:
                return
            self.status = status
            self.update(level="ERROR" if status in {"error", "timed_out", "denied"} else "WARNING",
                        status_message=type(exc).__name__ if exc else status,
                        metadata={"status": status, "errorType": type(exc).__name__ if exc else None,
                                  "reason": reason, "workMayContinue": status == "timed_out"})

    def end(self, status=None):
        with self.lock:
            if self.ended:
                return
            self.status = status or ("completed" if self.status == "running" else self.status)
            self.update(metadata={"status": self.status, "durationMs": round((time.perf_counter() - self.started) * 1000, 2)})
            self.ended = True
            if self.raw:
                try:
                    self.raw.end()
                except Exception:
                    pass
        if self.execution:
            with self.execution.lock:
                self.execution.handles.discard(self)


@contextmanager
def observation_scope(name, **kwargs):
    observation = Observation(name, **kwargs)
    token = _current.set(observation)
    try:
        yield observation
    except BaseException as exc:
        from app.agents.cancellation import AgentRunCancelled
        status = "cancelled" if isinstance(exc, AgentRunCancelled) else (
            "interrupted" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "error")
        if type(exc).__name__ == "ToolTimeoutError":
            status = "timed_out"
        if observation.status in {"denied", "timed_out"}:
            status = observation.status
        observation.error(exc, status=status)
        raise
    finally:
        _current.reset(token)
        observation.end()


@contextmanager
def execution_scope(context):
    token = _execution.set(context)
    parent = _current.set(None)
    try:
        # Attribute propagation wraps observation creation and remains in the
        # copied execution context. It never sets as_baggage=True.
        from contextlib import nullcontext
        attrs = nullcontext()
        if context.sampled:
            try:
                from langfuse import propagate_attributes
                attrs = propagate_attributes(user_id=context.user_id, session_id=context.session_id)
            except ImportError:
                pass
        with attrs, observation_scope("execution.chat") as root:
            try:
                yield root
            except BaseException as exc:
                from app.agents.cancellation import AgentRunCancelled
                root.error(exc, status="cancelled" if isinstance(exc, AgentRunCancelled) else "error")
                raise
            finally:
                with context.lock:
                    context.closed = True
                    handles = list(context.handles)
                for handle in handles:
                    if handle is not root:
                        pending_status = root.status if root.status in {"cancelled", "interrupted", "error", "failed"} else "interrupted"
                        handle.error(status=pending_status)
                        handle.end()
    finally:
        _current.reset(parent)
        _execution.reset(token)


def record_error(exc=None, **kwargs):
    current = _current.get()
    if current:
        current.error(exc, **kwargs)


def detached_context():
    """Keep ambient application context, clear this run's trace/cancel signal."""
    from app.agents.cancellation import _cancel_event
    from app.agents.events import clear_trace_context
    ctx = copy_context()
    ctx.run(_execution.set, None)
    ctx.run(_current.set, None)
    ctx.run(_cancel_event.set, None)
    ctx.run(clear_trace_context)
    return ctx
