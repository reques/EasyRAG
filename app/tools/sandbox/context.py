"""Context propagated with a request so policy decisions are attributable."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple


@dataclass(frozen=True)
class SandboxContext:
    """Immutable identity and execution scope for one agent run."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_mode: str = "chat"
    skill_ids: Tuple[str, ...] = ()


_sandbox_context: ContextVar[SandboxContext] = ContextVar(
    "tool_sandbox_context", default=SandboxContext()
)


def get_sandbox_context() -> SandboxContext:
    return _sandbox_context.get()


@contextmanager
def use_sandbox_context(context: SandboxContext) -> Iterator[SandboxContext]:
    token = _sandbox_context.set(context)
    try:
        yield context
    finally:
        _sandbox_context.reset(token)
