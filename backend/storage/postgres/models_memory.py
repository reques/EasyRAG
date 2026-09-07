"""持久记忆模型 — 跨会话语义事实与任务情景。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.storage.postgres.manager import Base


class UserFact(Base):
    """用户级事实（语义记忆）— 跨会话持久的用户偏好/身份/历史结论。

    本期为骨架：规则触发存储（用户说"记住/我喜欢/我是"时提取）+
    prompt 注入。LLM 自动判断提取留后续阶段。
    """

    __tablename__ = "user_facts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    # 溯源：这条事实从哪个会话提取的
    source_conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<UserFact {self.fact[:30]}>"


class EpisodeMemory(Base):
    """一次可复用的任务经历：目标、过程、结果、经验和遗留事项。"""

    __tablename__ = "episode_memories"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "source_message_id", name="uq_episode_user_source_message"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    source_conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    source_message_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str] = mapped_column(String(24), nullable=False, default="success")
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    lessons: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    unfinished: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    steps_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<EpisodeMemory {self.title[:30]}>"
