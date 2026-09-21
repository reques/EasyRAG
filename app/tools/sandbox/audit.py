"""Append-only, redacted audit records for tool policy decisions."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.tools.sandbox.context import SandboxContext
from app.tools.sandbox.policy import Decision

_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "api_key", "authorization", "cookie")
_write_lock = threading.Lock()


def redact_arguments(arguments: Mapping[str, Any], max_chars: int = 400) -> str:
    """Return a small, non-secret JSON representation suitable for logs."""

    def _redact(value: Any, key: str = "") -> Any:
        if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {str(k): _redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_redact(item) for item in value]
        if isinstance(value, str) and len(value) > 200:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            return f"[text chars={len(value)} sha256={digest}]"
        return value

    try:
        text = json.dumps(_redact(arguments), ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = "[unserializable arguments]"
    return text[:max_chars]


def record_decision(
    path: str | Path,
    *,
    context: SandboxContext,
    tool_name: str,
    decision: Decision,
    arguments: Mapping[str, Any],
    elapsed_ms: float = 0.0,
    outcome: str = "policy",
    error: str = "",
) -> None:
    """Write one JSONL record. Audit failures never block a tool call."""
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "agent_mode": context.agent_mode,
            "skill_ids": list(context.skill_ids),
            "tool": tool_name,
            "capabilities": list(decision.capabilities),
            "mode": decision.mode,
            "allowed": decision.allowed,
            "would_allow": decision.would_allow,
            "reason": decision.reason,
            "arguments": redact_arguments(arguments),
            "elapsed_ms": round(elapsed_ms, 1),
            "outcome": outcome,
        }
        if error:
            record["error"] = error[:200]
        with _write_lock:
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        # A full disk or a malformed custom path must not take down chat.
        return
