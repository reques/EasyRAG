"""Durable execution links and owner-scoped, administrator-only navigation."""
from __future__ import annotations

from datetime import timezone
import secrets
from urllib.parse import quote
import uuid

from app.core.config import get_settings
from app.observability.langfuse_client import valid_base_url


async def apply_migrations(connection):
    from sqlalchemy import text
    for table, columns in {
        "chat_streams": ("local_trace_id VARCHAR(32)", "observability_trace_id VARCHAR(32)",
                         "observability_sampled BOOLEAN", "resumed_from_stream_id UUID"),
        "execution_traces": ("stream_id UUID", "observability_trace_id VARCHAR(32)"),
    }.items():
        for column in columns:
            await connection.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column}"))
    await connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_execution_trace_stream ON execution_traces(stream_id)"))


def admission_fields(previous=None):
    cfg = get_settings()
    ready = bool(cfg.LANGFUSE_ENABLED and cfg.LANGFUSE_PUBLIC_KEY and cfg.LANGFUSE_SECRET_KEY
                 and valid_base_url(cfg.LANGFUSE_BASE_URL))
    return {
        "local_trace_id": uuid.uuid4().hex[:16],
        "observability_trace_id": uuid.uuid4().hex if ready else None,
        "observability_sampled": ready and secrets.randbelow(1000000) < cfg.LANGFUSE_SAMPLE_RATE * 1000000,
        "resumed_from_stream_id": previous.id if previous is not None else None,
    }


async def sync_execution_trace(execution, events=()):
    """Best effort local trace update; read committed business status, never infer it."""
    from app.core.logger import get_logger
    from backend.services import chat_stream_service as streams
    from backend.services.trace_service import persist_execution_trace
    from backend.storage.postgres.models_trace import ExecutionTrace
    try:
        async with streams.get_session() as session:
            run = await streams.owned_run(session, execution.id, execution.user_id)
            if not run.local_trace_id:
                return
            # Only the owner producer (or stale recovery acting on this exact
            # row) can update its trace. This doesn't write business status.
            if run.owner_token != execution.owner_token:
                return
            trace = await session.get(ExecutionTrace, run.local_trace_id)
            if trace is None:
                trace = ExecutionTrace(
                    id=run.local_trace_id, stream_id=run.id,
                    conversation_id=run.conversation_id, user_id=run.user_id,
                    source_message_id=run.source_message_id,
                    observability_trace_id=run.observability_trace_id,
                    mode="unknown", started_at=run.created_at,
                )
                session.add(trace)
            trace.status = run.status
            trace.completed_at = run.finished_at
            if run.finished_at:
                start = run.created_at.replace(tzinfo=timezone.utc) if run.created_at.tzinfo is None else run.created_at
                end = run.finished_at.replace(tzinfo=timezone.utc) if run.finished_at.tzinfo is None else run.finished_at
                trace.duration_ms = max(0, (end - start).total_seconds() * 1000)
            await session.commit()
        if events:
            await persist_execution_trace(
                streams.get_session, conversation_id=run.conversation_id, user_id=run.user_id,
                events=events, mode=getattr(execution, "agent_mode", "unknown"),
                model_id=getattr(execution, "model_id", ""), source_message_id=run.source_message_id,
            )
    except Exception as exc:
        get_logger(__name__).warning("[observability] local trace sync skipped (%s)", type(exc).__name__)


def navigation(run):
    from app.observability.langfuse_client import state
    cfg = get_settings()
    status = state()
    if not run.local_trace_id:
        status = "historical"
    elif status == "initialized" and not run.observability_sampled:
        status = "not_sampled" if run.observability_trace_id else "not_enabled_for_execution"
    url = None
    if (run.observability_sampled and run.observability_trace_id and
            valid_base_url(cfg.EASYRAG_OBSERVABILITY_WEB_BASE_URL) and cfg.EASYRAG_OBSERVABILITY_PROJECT_ID):
        base = cfg.EASYRAG_OBSERVABILITY_WEB_BASE_URL.rstrip("/")
        project = quote(cfg.EASYRAG_OBSERVABILITY_PROJECT_ID, safe="")
        trace = quote(run.observability_trace_id, safe="")
        url = f"{base}/project/{project}/traces/{trace}"
    return {"stream_id": str(run.id), "status": run.status,
            "local_trace_id": run.local_trace_id, "trace_id": run.observability_trace_id,
            "sampled": bool(run.observability_sampled), "export_state": status,
            "url": url, "visibility": "not_confirmed",
            "resumed_from_stream_id": str(run.resumed_from_stream_id) if run.resumed_from_stream_id else None}
