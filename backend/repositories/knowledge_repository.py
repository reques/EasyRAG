"""知识库仓库。"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import BaseRepository
from backend.storage.postgres.models_knowledge import KnowledgeBase, KnowledgeFile


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    model = KnowledgeBase

    async def get_by_name(self, name: str, owner_id: uuid.UUID) -> Optional[KnowledgeBase]:
        stmt = select(KnowledgeBase).where(
            KnowledgeBase.name == name,
            KnowledgeBase.owner_id == owner_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_owner(
        self, owner_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> Sequence[KnowledgeBase]:
        stmt = (
            select(KnowledgeBase)
            .where(KnowledgeBase.owner_id == owner_id)
            .order_by(KnowledgeBase.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_by_department(
        self, department_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> Sequence[KnowledgeBase]:
        stmt = (
            select(KnowledgeBase)
            .where(KnowledgeBase.department_id == department_id)
            .order_by(KnowledgeBase.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class KnowledgeFileRepository(BaseRepository[KnowledgeFile]):
    model = KnowledgeFile

    async def list_by_kb(
        self, kb_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> Sequence[KnowledgeFile]:
        stmt = (
            select(KnowledgeFile)
            .where(KnowledgeFile.knowledge_base_id == kb_id)
            .order_by(KnowledgeFile.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_filename(
        self, kb_id: uuid.UUID, filename: str
    ) -> Optional[KnowledgeFile]:
        stmt = select(KnowledgeFile).where(
            KnowledgeFile.knowledge_base_id == kb_id,
            KnowledgeFile.filename == filename,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
