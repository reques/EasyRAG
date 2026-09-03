"""知识库仓库。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

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

    async def list_ids_by_owner(self, owner_id: uuid.UUID) -> Sequence[uuid.UUID]:
        """Return the complete authorised retrieval scope without UI pagination."""
        stmt = select(KnowledgeBase.id).where(KnowledgeBase.owner_id == owner_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_catalog_by_owner(
        self, owner_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """Return authorised KB/file metadata without loading document bodies."""
        stmt = (
            select(
                KnowledgeBase.id.label("kb_id"),
                KnowledgeBase.name.label("kb_name"),
                KnowledgeFile.id.label("file_id"),
                KnowledgeFile.filename,
                KnowledgeFile.file_type,
                KnowledgeFile.status,
            )
            .outerjoin(
                KnowledgeFile,
                KnowledgeFile.knowledge_base_id == KnowledgeBase.id,
            )
            .where(KnowledgeBase.owner_id == owner_id)
            .order_by(KnowledgeBase.updated_at.desc(), KnowledgeFile.created_at.desc())
        )
        rows = (await self.session.execute(stmt)).all()

        catalog_by_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            kb_id = str(row.kb_id)
            item = catalog_by_id.setdefault(
                kb_id,
                {"id": kb_id, "name": row.kb_name, "files": []},
            )
            if row.file_id is not None:
                item["files"].append({
                    "id": str(row.file_id),
                    "filename": row.filename,
                    "file_type": row.file_type,
                    "status": row.status,
                })

        return list(catalog_by_id.values())

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

    async def list_by_ids_for_kb(
        self,
        kb_id: uuid.UUID,
        file_ids: Sequence[uuid.UUID],
    ) -> Sequence[KnowledgeFile]:
        """Return only requested files that belong to the given KB."""
        if not file_ids:
            return []
        stmt = select(KnowledgeFile).where(
            KnowledgeFile.knowledge_base_id == kb_id,
            KnowledgeFile.id.in_(file_ids),
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
