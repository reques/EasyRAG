"""记忆模型 — 语义记忆（跨会话用户事实）。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Text, func
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
