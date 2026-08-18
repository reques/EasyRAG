"""Per-user custom Agent Skill configuration."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.storage.postgres.manager import Base


class CustomSkillConfig(Base):
    __tablename__ = "custom_skill_configs"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_custom_skill_owner_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    tool_names_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    category: Mapped[str] = mapped_column(String(32), default="自定义", nullable=False)
    icon: Mapped[str] = mapped_column(String(32), default="sparkles", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    @property
    def public_id(self) -> str:
        return f"custom:{self.id}"
