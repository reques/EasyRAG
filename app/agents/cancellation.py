"""Cooperative cancellation propagated to tools and child executor contexts."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class AgentRunCancelled(BaseException):
    """Control flow, deliberately outside Exception/fallback/retry handlers."""


_cancel_event: ContextVar[Event | None] = ContextVar("agent_cancel_event", default=None)


@contextmanager
def use_run_cancellation(event: Event | None):
    token = _cancel_event.set(event)
    try:
        check_run_cancelled()
        yield
    finally:
        _cancel_event.reset(token)


def check_run_cancelled():
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise AgentRunCancelled()
