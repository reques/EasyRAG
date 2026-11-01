"""Owner-scoped assistant feedback with exact persisted execution association."""
from __future__ import annotations

import json
import uuid

from fastapi import HTTPException
from sqlalchemy import select

from backend.storage.postgres.models_conversation import Conversation, Message
from backend.storage.postgres.models_chat_stream import ChatStream
from backend.storage.postgres.models_observability import MessageFeedback, ObservabilityScore
from backend.services.score_delivery_service import enqueue_score


async def _owned_message(session, message_id, user_id):
    # Serialize concurrent updates for one message, including initial insert.
    message = (await session.execute(select(Message).join(Conversation).where(
        Message.id == message_id, Conversation.user_id == user_id,
    ).with_for_update(of=Message))).scalar_one_or_none()
    if message is None:
        raise HTTPException(404, "Message not found")
    if message.role != "assistant":
        raise HTTPException(422, "Only assistant messages can be rated")
    return message


async def get_feedback(session, message_id, user_id):
    await _owned_message(session, message_id, user_id)
    row = (await session.execute(select(MessageFeedback).where(
        MessageFeedback.message_id == message_id, MessageFeedback.user_id == user_id,
    ))).scalar_one_or_none()
    if row is None:
        return {"rating": None, "reason": "", "delivery_state": "not_submitted"}
    delivery = await session.get(ObservabilityScore, f"feedback-{row.id}")
    return {"rating": row.rating, "reason": row.reason,
            "delivery_state": delivery.state if delivery else "not_linked"}


async def save_feedback(session, message_id, user_id, rating, reason):
    message = await _owned_message(session, message_id, user_id)
    row = (await session.execute(select(MessageFeedback).where(
        MessageFeedback.message_id == message_id, MessageFeedback.user_id == user_id,
    ))).scalar_one_or_none()
    if row is None:
        row = MessageFeedback(id=uuid.uuid4(), message_id=message_id, user_id=user_id)
        session.add(row)
    row.rating, row.reason = rating, reason.strip()
    try:
        stream_id = uuid.UUID(json.loads(message.metadata_json or "{}").get("stream_id", ""))
    except (ValueError, TypeError, AttributeError):
        stream_id = None
    run = await session.get(ChatStream, stream_id) if stream_id else None
    if (run is not None and run.user_id == user_id and run.conversation_id == message.conversation_id
            and run.observability_sampled and run.observability_trace_id):
        from app.core.config import get_settings
        await enqueue_score(session, score_id=f"feedback-{row.id}", user_id=user_id, payload={
            "trace_id": run.observability_trace_id, "name": "user-feedback",
            "value": 1 if rating == "like" else 0, "data_type": "BOOLEAN", "comment": row.reason,
            "environment": get_settings().LANGFUSE_TRACING_ENVIRONMENT,
        })
    await session.flush()
    return await get_feedback(session, message_id, user_id)
