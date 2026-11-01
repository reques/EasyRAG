"""Allowlisted, bounded telemetry. Never serialize arbitrary runtime objects."""
from __future__ import annotations

import re
import uuid
import json
import math
from dataclasses import dataclass
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
    "knowledgeBaseIds", "retrievalResults", "topK", "scoreThreshold", "retrievalMode",
    "resultCount", "candidateCount", "evaluationRunId", "evaluationItemId", "datasetId",
})
_IDS = frozenset({"executionId", "localTraceId", "resumedFromTraceId", "resumedFromStreamId",
                  "agentInvocationId", "modelCallId", "delegationRunId", "invocationId", "user.id", "session.id",
                  "evaluationRunId", "evaluationItemId", "datasetId"})


def _safe_identity(key, value):
    if key in {"langfuse.experiment.id", "langfuse.experiment.dataset.id", "langfuse.experiment.item.id",
               "langfuse.experiment.item.root_observation_id"} and isinstance(value, str):
        return bool(re.fullmatch(r"[0-9a-f]{16,64}|[0-9a-f-]{36}", value))
    leaf = key if key in {"user.id", "session.id"} else key.rsplit(".", 1)[-1]
    if leaf not in _IDS or not isinstance(value, str):
        return False
    raw = value.strip('"')  # SDK v5 metadata string attributes are JSON encoded.
    if leaf in {"localTraceId", "evaluationItemId", "datasetId"}:
        return bool(re.fullmatch(r"[0-9a-f]{16,64}", raw))
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
        if key == "knowledgeBaseIds" and isinstance(value, (list, tuple)):
            result[key] = [redact(str(item), 64) for item in value[:20] if isinstance(item, (str, uuid.UUID))]
        elif key == "retrievalResults" and isinstance(value, list):
            result[key] = document_summaries(value)
        elif isinstance(value, str):
            result[key] = value if _safe_identity(key, value) else redact(value, 198)
        elif isinstance(value, (bool, int, float)):
            result[key] = value
    return result


def document_summaries(documents):
    """Only selected identities/ranking facts; never document text or filenames."""
    rows = []
    for rank, doc in enumerate(documents[:20], 1):
        row = {"rank": rank}
        for key in ("chunk_id", "file_id", "knowledge_base_id", "score", "rerank_score"):
            value = doc.get(key)
            if isinstance(value, str):
                row[key] = redact(value, 128)
            elif isinstance(value, (int, float)):
                import math
                if math.isfinite(value):
                    row[key] = value
        rows.append(row)
    return rows


@dataclass(frozen=True)
class SelectedContent:
    question: str | None = None
    contexts: tuple = ()


def selected_content(question=None, contexts=()):
    """An explicit schema, not permission to serialize arbitrary runtime objects."""
    return SelectedContent(question if isinstance(question, str) else None, tuple(contexts)[:20])


def content(value: Any) -> Any:
    from app.core.config import get_settings
    cfg = get_settings()
    if cfg.EASYRAG_OBSERVABILITY_CONTENT_MODE != "redacted":
        return None
    # Do not stringify dictionaries/messages (keys, images and reasoning can hide
    # credentials). Only explicitly selected plain text is allowed.
    if isinstance(value, str):
        return redact(value, cfg.EASYRAG_OBSERVABILITY_CONTENT_MAX_CHARS)
    if isinstance(value, SelectedContent):
        remaining = cfg.EASYRAG_OBSERVABILITY_CONTENT_MAX_CHARS
        def text(raw):
            nonlocal remaining
            if not isinstance(raw, str):
                return ""
            result = redact(raw, remaining)
            remaining = max(0, remaining - len(raw[:remaining]))
            return result
        result = {}
        if value.question is not None:
            result["question"] = text(value.question)
        if value.contexts:
            result["contexts"] = [
                {**summary, "content": text(doc.get("content"))}
                for doc, summary in zip(value.contexts, document_summaries(list(value.contexts)))
            ]
        return result
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
                values[key] = value if _safe_identity(key, value) else _mask_attribute(value)
        patches[identifier] = OtelSpanPatch(set_attributes=values, delete_attributes=tuple(deleted))
    return MaskOtelSpansResult(span_patches=patches)


def _mask_attribute(value: str) -> str:
    """Mask JSON string values without altering numbers or truncating JSON syntax."""
    if not value.lstrip().startswith(("{", "[")):
        return redact(value, 8192)
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return redact(value, 8192)
    remaining = 8192
    sensitive_keys = {"password", "secret", "token", "api_key", "apikey", "authorization", "cookie"}

    def sanitize(node, depth=0):
        nonlocal remaining
        if depth > 6:
            return None
        if isinstance(node, str):
            raw = node[:remaining]
            remaining -= len(raw)
            return redact(raw, len(raw))
        if isinstance(node, dict):
            return {key: "[REDACTED]" if key.lower() in sensitive_keys else sanitize(item, depth + 1)
                    for key, item in list(node.items())[:64]}
        if isinstance(node, list):
            return [sanitize(item, depth + 1) for item in node[:20]]
        if isinstance(node, float) and not math.isfinite(node):
            return None
        return node

    return json.dumps(sanitize(parsed), ensure_ascii=False, allow_nan=False)
