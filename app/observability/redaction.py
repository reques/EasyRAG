"""Allowlisted, bounded telemetry. Never serialize arbitrary runtime objects."""
from __future__ import annotations

import re
import uuid
from typing import Any

_SECRET = re.compile(
    r"(?i)(?:bearer\s+[\w.\-]+|\b(?:sk|pk)-[\w\-]+|"
    r"(?:api[_-]?key|secret|password|token|authorization|cookie)[\"']?\s*[=:]\s*[\"']?[^\s,;\"']+|"
    r"\beyJ[\w-]+\.[\w-]+\.[\w-]+|https?://[^\s]+|data:[^\s\"'<>]+|"
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b\+?\d[\d -]{8,}\d\b)"
)
_KEYS = frozenset({
    "executionId", "localTraceId", "resumedFromTraceId", "resumedFromStreamId",
    "agentId", "agentMode", "agentInvocationId", "taskId", "delegationRunId",
    "modelProfileId", "modelCallId", "tool", "serverId", "invocationId", "attempt", "attempts",
    "status", "errorType", "failureStage", "cancelReason", "workMayContinue",
    "stream", "messageCount", "outputChars", "usageStatus", "finishReason",
    "firstOutputMs", "firstContentMs", "durationMs", "recovered", "subagent",
    "task_key", "tasks", "mode", "reason", "stage",
})
_IDS = frozenset({"executionId", "localTraceId", "resumedFromTraceId", "resumedFromStreamId",
                  "agentInvocationId", "modelCallId", "delegationRunId", "invocationId", "user.id", "session.id"})


def _safe_identity(key, value):
    leaf = key if key in {"user.id", "session.id"} else key.rsplit(".", 1)[-1]
    if leaf not in _IDS or not isinstance(value, str):
        return False
    raw = value.strip('"')  # SDK v5 metadata string attributes are JSON encoded.
    if leaf == "localTraceId":
        return bool(re.fullmatch(r"[0-9a-f]{16,32}", raw))
    try:
        uuid.UUID(raw)
        return True
    except ValueError:
        return False


def redact(value: str, limit: int = 2000) -> str:
    return _SECRET.sub("[REDACTED]", value[:max(0, limit)])


def metadata(values: dict | None) -> dict:
    result = {}
    for key, value in (values or {}).items():
        if key not in _KEYS or value is None:
            continue
        if isinstance(value, str):
            result[key] = value if _safe_identity(key, value) else redact(value, 198)
        elif isinstance(value, (bool, int, float)):
            result[key] = value
    return result


def content(value: Any) -> Any:
    from app.core.config import get_settings
    cfg = get_settings()
    if cfg.EASYRAG_OBSERVABILITY_CONTENT_MODE != "redacted":
        return None
    # Do not stringify dictionaries/messages (keys, images and reasoning can hide
    # credentials). Only explicitly selected plain text is allowed.
    if isinstance(value, str):
        return redact(value, cfg.EASYRAG_OBSERVABILITY_CONTENT_MAX_CHARS)
    return None


def mask_otel_spans(*, params):
    """Second boundary, applied in the export worker, without request-local state."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
    patches = {}
    for identifier, span in params.spans.items():
        values, deleted = {}, []
        for key, value in span.attributes.items():
            if key.startswith("exception.") or any(x in key.lower() for x in ("authorization", "cookie", "secret", "api_key")):
                deleted.append(key)
            elif isinstance(value, str):
                values[key] = value if _safe_identity(key, value) else redact(value, 8192)
        patches[identifier] = OtelSpanPatch(set_attributes=values, delete_attributes=tuple(deleted))
    return MaskOtelSpansResult(span_patches=patches)
