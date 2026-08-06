"""对话与消息模型。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.storage.postgres.manager import Base


class Conversation(Base):
    """会话 — 一次完整的对话。"""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 情景记忆：长对话的增量摘要（每 10 轮由 LLM 压缩一次，注入 prompt 替代全部历史）
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 关联
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation",
        order_by="Message.created_at", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.id} title={self.title}>"


class Message(Base):
    """消息 — 对话中的单条消息。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant / system / tool
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 可选的元数据 JSON（来源引用、工具调用等）
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 关联
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<Message {self.id} role={self.role}>"
