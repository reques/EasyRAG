"""知识库路由 — 知识库 CRUD。"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from backend.services.knowledge_service import (
    create_knowledge_base,
    add_file_record,
    list_kb_files,
)
from backend.repositories.knowledge_repository import KnowledgeBaseRepository
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.models_user import User

logger = get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# ── Request / Response ────────────────────────────────────────────────────────

class KBCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field(None, max_length=1024)


class KBResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    collection_name: str
    created_at: str


class FileResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    char_count: int
    status: str
    created_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/bases", response_model=KBResponse, status_code=status.HTTP_201_CREATED)
async def create_kb(
    req: KBCreateRequest,
    current_user: User = Depends(get_current_user),
):
    """创建新知识库。"""
    async with get_session() as session:
        repo = KnowledgeBaseRepository(session)
        existing = await repo.get_by_name(req.name, current_user.id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Knowledge base '{req.name}' already exists",
            )
        kb = await create_knowledge_base(
            session,
            name=req.name,
            owner_id=current_user.id,
            description=req.description,
        )
        await session.commit()
        return KBResponse(
            id=str(kb.id),
            name=kb.name,
            description=kb.description,
            collection_name=kb.collection_name,
            created_at=kb.created_at.isoformat() if kb.created_at else "",
        )


@router.get("/bases", response_model=list[KBResponse])
async def list_kbs(current_user: User = Depends(get_current_user)):
    """列出当前用户的所有知识库。"""
    async with get_session() as session:
        repo = KnowledgeBaseRepository(session)
        kbs = await repo.list_by_owner(current_user.id)
        return [
            KBResponse(
                id=str(kb.id),
                name=kb.name,
                description=kb.description,
                collection_name=kb.collection_name,
                created_at=kb.created_at.isoformat() if kb.created_at else "",
            )
            for kb in kbs
        ]


@router.get("/bases/{kb_id}/files", response_model=list[FileResponse])
async def list_files(
    kb_id: str,
    current_user: User = Depends(get_current_user),
):
    """列出知识库中的所有文件。"""
    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        files = await list_kb_files(session, uuid.UUID(kb_id))
        return [
            FileResponse(
                id=str(f.id),
                filename=f.filename,
                file_type=f.file_type,
                chunk_count=f.chunk_count,
                char_count=f.char_count,
                status=f.status,
                created_at=f.created_at.isoformat() if f.created_at else "",
            )
            for f in files
        ]
