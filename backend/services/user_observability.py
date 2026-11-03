"""Owner-scoped read models for the native observability workspace."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import JSON, String, Text, cast, func, or_, select

from backend.services.observation_reader import display_content
from backend.services.trace_service import serialize_event, serialize_trace
from backend.storage.postgres.models_chat_stream import ChatStream
from backend.storage.postgres.models_conversation import Conversation, Message
from backend.storage.postgres.models_observability import MessageFeedback, ObservabilityScore
from backend.storage.postgres.models_trace import ExecutionTrace, TraceEvent

JSON_TEXT = JSON().with_variant(Text(), "sqlite")


def json_object(raw):
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def public_event(event):
    payload = serialize_event(event)
    for key in ("input", "output", "metadata"):
        payload[key] = display_content(payload[key])
    return payload


def trace_query(user_id, *, window="7d", q="", status="", model="", agent="", conversation_id=None):
    query = select(ExecutionTrace, Conversation.title, Message.content, Message.metadata_json).join(
        Conversation, Conversation.id == ExecutionTrace.conversation_id).outerjoin(
        Message, Message.id == ExecutionTrace.source_message_id).where(
        ExecutionTrace.user_id == user_id, Conversation.user_id == user_id)
    if window != "all":
        query = query.where(ExecutionTrace.started_at >= datetime.now(timezone.utc) - timedelta(days=7 if window == "7d" else 30))
    if q:
        query = query.where(or_(ExecutionTrace.id.contains(q, autoescape=True), Message.content.icontains(q, autoescape=True),
                                Conversation.title.icontains(q, autoescape=True)))
    if status:
        query = query.where(ExecutionTrace.status == status)
    if model:
        query = query.where(ExecutionTrace.model_id == model)
    if agent:
        query = query.where(cast(Message.metadata_json, JSON_TEXT)["agent_id"].as_string() == agent)
    if conversation_id:
        query = query.where(ExecutionTrace.conversation_id == conversation_id)
    return query


def trace_row(row):
    trace, title, question, request_meta = row
    info = serialize_trace(trace, include_events=False)
    # metadata may contain old runtime fields; expose only the selected meter.
    info.pop("metadata", None)
    meta = json_object(request_meta)
    info.update(conversation_title=display_content(title), question=display_content(question, limit=2000),
                agent_id=meta.get("agent_id", ""), agent_name=display_content(meta.get("agent_name", "自动选择")),
                stream_id=str(trace.stream_id) if trace.stream_id else None,
                observability_trace_id=trace.observability_trace_id, cost_usd=None,
                context_usage=display_content(info.get("context_usage")))
    return info


async def owned_trace(session, trace_id, user_id):
    row = (await session.execute(trace_query(user_id, window="all").where(ExecutionTrace.id == trace_id))).first()
    if row is None:
        raise HTTPException(404, "执行记录不存在或无权访问")
    return row


async def list_traces(session, user_id, *, page=1, page_size=20, **filters):
    query = trace_query(user_id, **filters)
    total = (await session.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
    rows = (await session.execute(query.order_by(ExecutionTrace.started_at.desc(), ExecutionTrace.id.desc())
                                  .offset((page - 1) * page_size).limit(page_size))).all()
    return {"items": [trace_row(row) for row in rows], "total": total, "page": page, "page_size": page_size}


async def overview(session, user_id, **filters):
    sub = trace_query(user_id, **filters).subquery()
    counts = dict((await session.execute(select(sub.c.status, func.count()).group_by(sub.c.status))).all())
    totals = (await session.execute(select(func.coalesce(func.sum(sub.c.total_tokens), 0),
                                          func.avg(sub.c.duration_ms).filter(sub.c.completed_at.is_not(None))))).one()
    models = (await session.execute(select(sub.c.model_id, func.count(), func.sum(sub.c.total_tokens))
                                    .group_by(sub.c.model_id).order_by(func.count().desc()).limit(12))).all()
    tools = (await session.execute(select(TraceEvent.metadata_json, TraceEvent.status).where(
        TraceEvent.trace_id.in_(select(sub.c.id)), TraceEvent.type.in_(["tool_call", "file_operation", "code_execution"])
    ).order_by(TraceEvent.timestamp.desc()).limit(10000))).all()
    tool_groups = {}
    for raw, state in tools:
        name = str(json_object(raw).get("tool") or "工具")[:120]
        bucket = tool_groups.setdefault(name, {"name": name, "calls": 0, "errors": 0})
        bucket["calls"] += 1
        bucket["errors"] += int(state == "error")
    # Feedback follows exact persisted execution links, never message proximity.
    message_meta = cast(Message.metadata_json, JSON_TEXT)
    feedback_rows = (await session.execute(select(MessageFeedback.rating).join(Message, Message.id == MessageFeedback.message_id)
        .join(Conversation, Conversation.id == Message.conversation_id).where(
        MessageFeedback.user_id == user_id, Conversation.user_id == user_id,
        or_(message_meta["trace_id"].as_string().in_(select(sub.c.id)),
            message_meta["stream_id"].as_string().in_(select(cast(sub.c.stream_id, String))))))).scalars().all()
    # UUID string casts differ in SQLite; trace_id is the normal durable path.
    terminal = sum(n for s, n in counts.items() if s not in {"running", "cancel_requested"})
    return {"total": sum(counts.values()), "status_counts": counts,
            "success_rate": counts.get("completed", 0) / terminal if terminal else None,
            "total_tokens": int(totals[0]), "avg_duration_ms": totals[1], "cost_usd": None,
            "models": [{"model_id": model, "calls": count, "total_tokens": int(tokens or 0)} for model, count, tokens in models],
            "tools": sorted(tool_groups.values(), key=lambda x: -x["calls"])[:12], "tools_truncated": len(tools) == 10000,
            "feedback": {"like": feedback_rows.count("like"), "dislike": feedback_rows.count("dislike")}}


async def scores_for_trace(session, remote_trace_id, user_id):
    if not remote_trace_id:
        return []
    rows = (await session.execute(select(ObservabilityScore).where(
        ObservabilityScore.user_id == user_id,
        cast(ObservabilityScore.payload_json, JSON_TEXT)["trace_id"].as_string() == remote_trace_id))).scalars().all()
    return [{"id": row.id, "name": json_object(row.payload_json).get("name"),
             "value": json_object(row.payload_json).get("value"),
             "comment": display_content(json_object(row.payload_json).get("comment")),
             "observation_id": json_object(row.payload_json).get("observation_id"),
             "delivery_state": row.state} for row in rows]


async def trace_detail(session, trace_id, user_id):
    row = await owned_trace(session, trace_id, user_id)
    trace = row[0]
    info = trace_row(row)
    events = (await session.execute(select(TraceEvent).where(TraceEvent.trace_id == trace_id)
                                   .order_by(TraceEvent.sequence).limit(501))).scalars().all()
    info["events"] = [public_event(event) for event in events[:500]]
    info["events_truncated"] = len(events) > 500
    conditions = [cast(Message.metadata_json, JSON_TEXT)["trace_id"].as_string() == trace.id]
    if trace.stream_id:
        conditions.append(cast(Message.metadata_json, JSON_TEXT)["stream_id"].as_string() == str(trace.stream_id))
    answer = (await session.execute(select(Message).where(Message.conversation_id == trace.conversation_id,
        Message.role == "assistant", or_(*conditions)).order_by(Message.id.desc()).limit(1))).scalar_one_or_none()
    info["answer"] = display_content(answer.content, limit=12000) if answer else None
    info["answer_message_id"] = answer.id if answer else None
    feedback = (await session.execute(select(MessageFeedback).where(MessageFeedback.user_id == user_id,
        MessageFeedback.message_id == answer.id))).scalar_one_or_none() if answer else None
    info["feedback"] = {"rating": feedback.rating, "reason": display_content(feedback.reason)} if feedback else None
    run = await session.get(ChatStream, trace.stream_id) if trace.stream_id else None
    info["sampled"] = bool(run.observability_sampled) if run else bool(trace.observability_trace_id)
    info["scores"] = await scores_for_trace(session, trace.observability_trace_id, user_id)
    from backend.services.turn_judge import get_judgement
    info["judgement"] = await get_judgement(session, trace.id, user_id)
    return info
