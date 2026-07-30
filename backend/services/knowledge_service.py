"""知识库服务 — 知识库 CRUD + 文件管理。"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from backend.repositories.knowledge_repository import (
    KnowledgeBaseRepository,
    KnowledgeFileRepository,
)
from backend.storage.postgres.models_knowledge import KnowledgeBase, KnowledgeFile

logger = get_logger(__name__)


async def create_knowledge_base(
    session: AsyncSession,
    name: str,
    owner_id: uuid.UUID,
    description: Optional[str] = None,
    department_id: Optional[uuid.UUID] = None,
) -> KnowledgeBase:
    repo = KnowledgeBaseRepository(session)
    kb = KnowledgeBase(
        name=name,
        description=description,
        owner_id=owner_id,
        department_id=department_id,
        collection_name=f"kb_{uuid.uuid4().hex[:12]}",
    )
    await repo.add(kb)
    return kb


async def add_file_record(
    session: AsyncSession,
    kb_id: uuid.UUID,
    filename: str,
    file_type: str,
    chunk_count: int = 0,
    char_count: int = 0,
    minio_bucket: Optional[str] = None,
    minio_object: Optional[str] = None,
) -> KnowledgeFile:
    repo = KnowledgeFileRepository(session)
    f = KnowledgeFile(
        knowledge_base_id=kb_id,
        filename=filename,
        file_type=file_type,
        chunk_count=chunk_count,
        char_count=char_count,
        minio_bucket=minio_bucket,
        minio_object=minio_object,
        status="pending",
    )
    await repo.add(f)
    return f


async def list_kb_files(
    session: AsyncSession, kb_id: uuid.UUID
) -> list[KnowledgeFile]:
    repo = KnowledgeFileRepository(session)
    return list(await repo.list_by_kb(kb_id))


async def update_file_status(
    session: AsyncSession, file_id: uuid.UUID, status: str
) -> None:
    repo = KnowledgeFileRepository(session)
    f = await repo.get_by_id(file_id)
    if f:
        f.status = status
        await session.flush()
