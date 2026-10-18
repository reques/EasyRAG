"""Durable execution identity and ordered replay, separate from HTTP clients."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.storage.postgres.manager import Base


class ChatStream(Base):
    __tablename__ = "chat_streams"
    __table_args__ = (
        Index(
            "uq_chat_stream_active_conversation", "conversation_id", unique=True,
            postgresql_where=text("status IN ('running', 'cancel_requested')"),
            sqlite_where=text("status IN ('running', 'cancel_requested')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    source_message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id", ondelete="CASCADE"))
    fingerprint: Mapped[str] = mapped_column(String(64))
    owner_token: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    last_sequence: Mapped[int] = mapped_column(Integer, default=0)
    local_trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    observability_trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    observability_sampled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    resumed_from_stream_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ChatStreamEvent(Base):
    __tablename__ = "chat_stream_events"

    stream_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_streams.id", ondelete="CASCADE"), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text)
