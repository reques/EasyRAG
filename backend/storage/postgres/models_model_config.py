"""Per-user custom LLM endpoint configuration."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.storage.postgres.manager import Base


class CustomModelConfig(Base):
    """A user-owned OpenAI-compatible local or cloud model endpoint."""

    __tablename__ = "custom_model_configs"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_custom_model_owner_name"),
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
    provider_name: Mapped[str] = mapped_column(String(80), nullable=False)
    provider_type: Mapped[str] = mapped_column(String(16), nullable=False)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    model_name: Mapped[str] = mapped_column(String(160), nullable=False)
    api_key_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requires_api_key: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # 是否支持图片（多模态）输入；由用户在添加/编辑模型时勾选，默认 False
    supports_vision: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
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
