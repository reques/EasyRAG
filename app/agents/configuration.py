"""Request-local agent definition and tool boundary, propagated by ContextVars."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentConfiguration:
    id: str
    name: str
    mode: str
    system_prompt: str = ""
    tool_names: tuple[str, ...] | None = None
    subagents: tuple[Any, ...] = ()
    child_models: dict[str, Any] = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)


_configuration: ContextVar[AgentConfiguration | None] = ContextVar("agent_configuration", default=None)
_tools: ContextVar[frozenset[str] | None] = ContextVar("agent_tool_scope", default=None)


def get_agent_configuration():
    return _configuration.get()


def agent_tool_allowed(name):
    scope = _tools.get()
    return scope is None or name in scope


@contextmanager
def use_agent_configuration(configuration):
    token = _configuration.set(configuration)
    scope = configuration.tool_names if configuration is not None else None
    tools_token = _tools.set(frozenset(scope) if scope is not None else None)
    try:
        yield
    finally:
        _tools.reset(tools_token)
        _configuration.reset(token)


@contextmanager
def use_child_configuration(config):
    from app.agents.deep.subagents import resolve_tool_spec
    from app.llm.client import use_chat_model
    active = get_agent_configuration()
    # Child tool scope comes from the resolved authorized child definition;
    # Skill, KB and sandbox scopes remain inherited and can still deny it.
    scope = frozenset(resolve_tool_spec(config.tools))
    token = _tools.set(scope) if active is not None else None
    profile = active.child_models.get(config.name) if active else None
    try:
        with use_chat_model(profile) if profile is not None else nullcontext():
            yield
    finally:
        if token is not None:
            _tools.reset(token)
