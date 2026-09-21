"""Regression tests for the first-stage tool permission sandbox."""
from __future__ import annotations

import json

import pytest

from app.core.exceptions import ToolExecutionError
from app.tools.registry import ToolDefinition, ToolRegistry
from app.tools.sandbox.context import SandboxContext, get_sandbox_context, use_sandbox_context
from app.tools.sandbox.policy import PolicyEngine


def _tool(name: str, fn, capabilities):
    return ToolDefinition(
        name=name,
        description=name,
        fn=fn,
        metadata={"capabilities": capabilities},
    )


def test_enforce_mode_blocks_declared_capability_and_writes_redacted_audit(tmp_path):
    audit_file = tmp_path / "tool_audit.jsonl"
    engine = PolicyEngine({
        "defaults": {"mode": "enforce", "deny_capabilities": ["net.out"]},
    })
    registry = ToolRegistry(policy_engine=engine, audit_file=str(audit_file))
    called = {"value": False}
    registry.register(_tool(
        "network_tool",
        lambda **_: called.__setitem__("value", True) or "unreachable",
        ["net.out"],
    ))

    with use_sandbox_context(SandboxContext(user_id="u-1", session_id="s-1")):
        with pytest.raises(ToolExecutionError, match="denied by sandbox policy"):
            registry.invoke("network_tool", api_key="do-not-log")

    assert called["value"] is False
    entry = json.loads(audit_file.read_text(encoding="utf-8"))
    assert entry["allowed"] is False
    assert entry["outcome"] == "denied"
    assert entry["user_id"] == "u-1"
    assert "do-not-log" not in entry["arguments"]
    assert "[REDACTED]" in entry["arguments"]


def test_audit_mode_allows_call_but_records_would_deny(tmp_path):
    audit_file = tmp_path / "tool_audit.jsonl"
    engine = PolicyEngine({
        "defaults": {"mode": "audit", "deny_capabilities": ["proc.exec"]},
    })
    registry = ToolRegistry(policy_engine=engine, audit_file=str(audit_file))
    registry.register(_tool("executor", lambda **_: "ok", ["proc.exec"]))

    assert registry.invoke("executor") == "ok"

    entry = json.loads(audit_file.read_text(encoding="utf-8"))
    assert entry["allowed"] is True
    assert entry["would_allow"] is False
    assert entry["outcome"] == "success"
    assert entry["reason"].startswith("would deny:")


def test_sandbox_context_reaches_timeout_worker_thread(tmp_path):
    audit_file = tmp_path / "tool_audit.jsonl"
    registry = ToolRegistry(
        policy_engine=PolicyEngine({"defaults": {"mode": "enforce"}}),
        audit_file=str(audit_file),
    )
    registry.register(_tool(
        "context_probe",
        lambda **_: f"{get_sandbox_context().user_id}:{get_sandbox_context().session_id}",
        ["none"],
    ))

    with use_sandbox_context(SandboxContext(user_id="user-7", session_id="session-7")):
        assert registry.invoke("context_probe") == "user-7:session-7"

