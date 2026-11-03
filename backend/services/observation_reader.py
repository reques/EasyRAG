"""Read one already-authorized Langfuse trace; never enumerate a project."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone

import httpx

from app.core.config import get_settings
from app.observability.langfuse_client import valid_base_url
from app.observability.redaction import metadata, redact

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_OBSERVATIONS = 500
_PRIVATE = re.compile(r"(?i)(api.?key|secret|password|authorization|cookie|(?:^|_)token$|reasoning|system.?prompt)")


def display_content(value, *, limit=8192):
    """Bounded JSON presentation; never render secrets, images or model reasoning."""
    remaining = limit

    def visit(node, depth=0):
        nonlocal remaining
        if depth > 6:
            return None
        if isinstance(node, str):
            raw = node[:remaining]
            remaining -= len(raw)
            return redact(raw, len(raw))
        if isinstance(node, dict):
            if node.get("role") == "system":
                return None
            return {str(key)[:100]: "[REDACTED]" if _PRIVATE.search(str(key)) else visit(item, depth + 1)
                    for key, item in list(node.items())[:40]}
        if isinstance(node, (list, tuple)):
            return [visit(item, depth + 1) for item in node[:30]]
        if node is None or isinstance(node, (bool, int)):
            return node
        if isinstance(node, float):
            return node if math.isfinite(node) else None
        return None

    return visit(value)


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def project_observations(payload, trace_id, *, user_id=None):
    if not isinstance(payload, dict) or payload.get("id") != trace_id:
        raise ValueError("TraceIdentityMismatch")
    rows = payload.get("observations") or []
    if not isinstance(rows, list):
        raise ValueError("InvalidObservations")
    include_content = get_settings().EASYRAG_OBSERVABILITY_CONTENT_MODE == "redacted"
    observations = []
    for row in rows[:MAX_OBSERVATIONS]:
        if not isinstance(row, dict) or row.get("traceId") != trace_id:
            continue
        if user_id is not None and row.get("userId") != str(user_id):
            continue
        identifier = str(row.get("id") or "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", identifier):
            continue
        safe_meta = display_content(metadata(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}))
        usage = row.get("usageDetails") or row.get("usage") or {}
        cost = row.get("costDetails") or {}
        observations.append({
            "id": identifier, "parent_id": row.get("parentObservationId") if isinstance(row.get("parentObservationId"), str) else None,
            "name": redact(str(row.get("name") or "Observation"), 160),
            "type": str(row.get("type") or "SPAN")[:32],
            "status": safe_meta.get("status") or ("error" if row.get("level") == "ERROR" else "completed" if row.get("endTime") else "running"),
            "started_at": row.get("startTime"), "completed_at": row.get("endTime"),
            "model": redact(str(row.get("model") or ""), 160),
            "input": display_content(_parse_io(row.get("input"))) if include_content else None,
            "output": display_content(_parse_io(row.get("output"))) if include_content else None,
            "metadata": safe_meta,
            "token_usage": {key: _number(usage.get(key)) for key in ("input", "output", "total")} if isinstance(usage, dict) else {},
            "cost_usd": _number(cost.get("total", row.get("totalCost"))) if isinstance(cost, dict) else None,
            "error": redact(str(row.get("statusMessage") or ""), 500) if row.get("level") in {"ERROR", "WARNING"} else "",
            "source": "langfuse",
        })
    return {"state": "available", "source": "langfuse", "trace_id": trace_id,
            "observations": observations, "truncated": len(rows) > MAX_OBSERVATIONS,
            "content_mode": get_settings().EASYRAG_OBSERVABILITY_CONTENT_MODE}


def _parse_io(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value


async def read_observations(trace_id, *, sampled=True, started_at=None, completed_at=None, user_id=None):
    cfg = get_settings()
    empty = {"source": "langfuse", "observations": [], "trace_id": trace_id}
    if not trace_id:
        return {**empty, "state": "not_linked"}
    if not sampled:
        return {**empty, "state": "not_sampled"}
    if not cfg.LANGFUSE_ENABLED:
        return {**empty, "state": "disabled"}
    if not (valid_base_url(cfg.LANGFUSE_BASE_URL) and cfg.LANGFUSE_PUBLIC_KEY and cfg.LANGFUSE_SECRET_KEY):
        return {**empty, "state": "misconfigured"}
    if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
        return {**empty, "state": "invalid_link"}
    def timestamp(value, fallback):
        try:
            parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
        except (ValueError, TypeError, AttributeError):
            return fallback
    now = datetime.now(timezone.utc)
    params = {"traceId": trace_id, "limit": MAX_OBSERVATIONS + 1,
              "fields": "core,basic,time,io,metadata,model,usage",
              "fromStartTime": (timestamp(started_at, now - timedelta(days=30)) - timedelta(minutes=5)).isoformat(),
              "toStartTime": (timestamp(completed_at, now) + timedelta(hours=1)).isoformat()}
    if user_id:
        params["userId"] = str(user_id)
    try:
        async with httpx.AsyncClient(timeout=4, follow_redirects=False) as client:
            async with client.stream("GET", f"{cfg.LANGFUSE_BASE_URL.rstrip('/')}/api/public/v2/observations", params=params,
                                     auth=(cfg.LANGFUSE_PUBLIC_KEY, cfg.LANGFUSE_SECRET_KEY)) as response:
                if response.status_code == 404:
                    return {**empty, "state": "pending_index"}
                if response.status_code in {401, 403}:
                    return {**empty, "state": "misconfigured"}
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_RESPONSE_BYTES:
                        return {**empty, "state": "response_too_large"}
        payload = json.loads(data)
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("InvalidObservationResponse")
        if not rows:
            return {**empty, "state": "pending_index"}
        result = project_observations({"id": trace_id, "observations": rows}, trace_id, user_id=user_id)
        result["truncated"] = result["truncated"] or bool((payload.get("meta") or {}).get("cursor"))
        return result
    except (httpx.HTTPError, ValueError, TypeError):
        return {**empty, "state": "unavailable"}
