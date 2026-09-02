"""Owner-scoped persistence for custom Skill configurations."""
from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import BaseRepository
from backend.storage.postgres.models_skill_config import CustomSkillConfig


class CustomSkillConfigRepository(BaseRepository[CustomSkillConfig]):
    model = CustomSkillConfig

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def list_by_owner(
        self, owner_id: uuid.UUID
    ) -> Sequence[CustomSkillConfig]:
        stmt = (
            select(CustomSkillConfig)
            .where(
                CustomSkillConfig.owner_id == owner_id,
                CustomSkillConfig.is_active.is_(True),
            )
            .order_by(CustomSkillConfig.created_at.asc())
        )
        return (await self.session.execute(stmt)).scalars().all()

    async def get_by_public_id(
        self, owner_id: uuid.UUID, public_id: str
    ) -> Optional[CustomSkillConfig]:
        if not public_id.startswith("custom:"):
            return None
        try:
            record_id = uuid.UUID(public_id.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
        stmt = select(CustomSkillConfig).where(
            CustomSkillConfig.id == record_id,
            CustomSkillConfig.owner_id == owner_id,
            CustomSkillConfig.is_active.is_(True),
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_name(
        self, owner_id: uuid.UUID, name: str
    ) -> Optional[CustomSkillConfig]:
        stmt = select(CustomSkillConfig).where(
            CustomSkillConfig.owner_id == owner_id,
            CustomSkillConfig.name == name,
            CustomSkillConfig.is_active.is_(True),
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
