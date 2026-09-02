"""Owner-scoped persistence for user-created chat model endpoints."""
from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import BaseRepository
from backend.storage.postgres.models_model_config import CustomModelConfig


class CustomModelConfigRepository(BaseRepository[CustomModelConfig]):
    model = CustomModelConfig

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def list_by_owner(
        self, owner_id: uuid.UUID
    ) -> Sequence[CustomModelConfig]:
        stmt = (
            select(CustomModelConfig)
            .where(
                CustomModelConfig.owner_id == owner_id,
                CustomModelConfig.is_active.is_(True),
            )
            .order_by(CustomModelConfig.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_public_id(
        self, owner_id: uuid.UUID, public_id: str
    ) -> Optional[CustomModelConfig]:
        if not public_id.startswith("custom:"):
            return None
        try:
            record_id = uuid.UUID(public_id.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
        stmt = select(CustomModelConfig).where(
            CustomModelConfig.id == record_id,
            CustomModelConfig.owner_id == owner_id,
            CustomModelConfig.is_active.is_(True),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    async def get_by_name(
        self, owner_id: uuid.UUID, name: str
    ) -> Optional[CustomModelConfig]:
        stmt = select(CustomModelConfig).where(
            CustomModelConfig.owner_id == owner_id,
            CustomModelConfig.name == name,
            CustomModelConfig.is_active.is_(True),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
