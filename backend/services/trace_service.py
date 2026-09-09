"""Persistence and read models for canonical Agent execution traces."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.storage.postgres.models_trace import ExecutionTrace, TraceEvent


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def summarize_trace(events: Iterable[dict]) -> dict:
    """Derive trace status, timing and token totals from canonical events."""
    rows = list(events)
    started = _datetime(rows[0].get("timestamp")) if rows else datetime.now(timezone.utc)
    completed = _datetime(rows[-1].get("timestamp")) if rows else started
    status = "error" if any(e.get("status") == "error" or e.get("type") == "error" for e in rows) else "completed"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in rows:
        candidate = (event.get("metadata") or {}).get("token_usage") or {}
        for key in usage:
            usage[key] = max(usage[key], int(candidate.get(key, 0) or 0))
    duration_ms = max(0.0, (completed - started).total_seconds() * 1000)
    final_meta = rows[-1].get("metadata") or {} if rows else {}
    if final_meta.get("elapsed_ms") is not None:
        duration_ms = float(final_meta["elapsed_ms"])
    return {
        "status": status,
        "started_at": started,
        "completed_at": completed,
        "duration_ms": round(duration_ms, 1),
        **usage,
    }


async def persist_execution_trace(
    session_factory,
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    events: Iterable[dict],
    mode: str,
    model_id: str,
    source_message_id: Optional[int],
    goal: str = "",
) -> str:
    """Persist one completed run. Repeated calls for a trace are idempotent."""
    rows = [dict(event) for event in events if event.get("id") and event.get("trace_id")]
    if not rows:
        return ""
    trace_id = str(rows[0]["trace_id"])
    summary = summarize_trace(rows)
    async with session_factory() as session:
        existing = await session.get(ExecutionTrace, trace_id)
        if existing is not None:
            return trace_id
        trace = ExecutionTrace(
            id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            source_message_id=source_message_id,
            mode=mode,
            model_id=model_id,
            metadata_json=_json({"goal": goal}),
            **summary,
        )
        session.add(trace)
        session.add_all([
            TraceEvent(
                id=str(event["id"]),
                trace_id=trace_id,
                parent_id=event.get("parent_id"),
                sequence=index,
                type=str(event.get("type") or "planning"),
                status=(
                    "completed"
                    if event.get("status") == "running"
                    else str(event.get("status") or "info")
                ),
                timestamp=_datetime(event.get("timestamp")),
                input_json=_json(event.get("input")),
                output_json=_json(event.get("output")),
                metadata_json=_json(event.get("metadata") or {}),
                duration_ms=(event.get("metadata") or {}).get("elapsed_ms"),
            )
            for index, event in enumerate(rows, 1)
        ])
        await session.commit()
    return trace_id


def serialize_event(event: TraceEvent) -> dict:
    return {
        "id": event.id,
        "parent_id": event.parent_id,
        "type": event.type,
        "status": event.status,
        "timestamp": event.timestamp.isoformat(),
        "input": _loads(event.input_json),
        "output": _loads(event.output_json),
        "metadata": _loads(event.metadata_json) or {},
        "sequence": event.sequence,
    }


def serialize_trace(trace: ExecutionTrace, include_events: bool = True) -> dict:
    payload = {
        "id": trace.id,
        "conversation_id": str(trace.conversation_id),
        "source_message_id": trace.source_message_id,
        "mode": trace.mode,
        "model_id": trace.model_id,
        "status": trace.status,
        "token_usage": {
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "total_tokens": trace.total_tokens,
        },
        "duration_ms": trace.duration_ms,
        "started_at": trace.started_at.isoformat(),
        "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
        "metadata": _loads(trace.metadata_json) or {},
    }
    if include_events:
        payload["events"] = [serialize_event(event) for event in trace.events]
    return payload


async def get_execution_trace(session, trace_id: str, user_id: uuid.UUID) -> Optional[dict]:
    row = (
        await session.execute(
            select(ExecutionTrace)
            .options(selectinload(ExecutionTrace.events))
            .where(ExecutionTrace.id == trace_id, ExecutionTrace.user_id == user_id)
        )
    ).scalar_one_or_none()
    return serialize_trace(row) if row else None


async def list_execution_traces(
    session, conversation_id: uuid.UUID, user_id: uuid.UUID, limit: int = 50,
    include_events: bool = True,
) -> list[dict]:
    statement = select(ExecutionTrace)
    if include_events:
        statement = statement.options(selectinload(ExecutionTrace.events))
    rows = (
        await session.execute(
            statement.where(
                ExecutionTrace.conversation_id == conversation_id,
                ExecutionTrace.user_id == user_id,
            )
            .order_by(ExecutionTrace.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return [serialize_trace(row, include_events=include_events) for row in rows]
