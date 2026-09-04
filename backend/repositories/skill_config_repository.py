"""个人 Skill 索引表的 owner-scoped 持久化（内容在磁盘，见 skill_config_service）。"""
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

    async def get_by_slug(
        self, owner_id: uuid.UUID, slug: str
    ) -> Optional[CustomSkillConfig]:
        """按 slug 取索引行（owner 隔离）。

        重构前是 ``get_by_public_id`` 解析 ``custom:<uuid>``；现在 slug 就是
        对外标识，不再需要前缀解析与 UUID 转换。
        """
        normalized = (slug or "").strip().lower()
        if not normalized:
            return None
        stmt = select(CustomSkillConfig).where(
            CustomSkillConfig.owner_id == owner_id,
            CustomSkillConfig.slug == normalized,
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
