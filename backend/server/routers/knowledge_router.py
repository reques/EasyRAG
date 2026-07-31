"""知识库路由 — 知识库 CRUD + 文件上传索引。"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.services.knowledge_service import (
    create_knowledge_base,
    add_file_record,
    list_kb_files,
    update_file_status,
)
from backend.repositories.knowledge_repository import KnowledgeBaseRepository
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.models_user import User

logger = get_logger(__name__)
cfg = get_settings()
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


class UploadResponse(BaseModel):
    file_id: str
    indexed: int
    message: str
    graph: Optional[dict] = None   # GRAPH_ENABLED 时: {"entities": n, "relations": m}


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ── 阶段 2C: 图谱查询 ─────────────────────────────────────────────────────────

class EntityResponse(BaseModel):
    id: str
    name: str
    entity_type: str
    description: Optional[str]
    source_chunks: Optional[str]


class RelationResponse(BaseModel):
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    description: Optional[str]
    weight: float


class GraphResponse(BaseModel):
    entities: list[EntityResponse]
    relations: list[RelationResponse]


@router.get("/bases/{kb_id}/graph", response_model=GraphResponse)
async def get_kb_graph(
    kb_id: str,
    current_user: User = Depends(get_current_user),
):
    """返回知识库的完整图谱（实体 + 关系），供前端可视化。"""
    from sqlalchemy import select
    from backend.storage.postgres.models_knowledge import KnowledgeEntity, KnowledgeRelation

    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        entities = (await session.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.knowledge_base_id == kb.id)
        )).scalars().all()
        relations = (await session.execute(
            select(KnowledgeRelation).where(KnowledgeRelation.knowledge_base_id == kb.id)
        )).scalars().all()

        return GraphResponse(
            entities=[EntityResponse(
                id=str(e.id), name=e.name, entity_type=e.entity_type,
                description=e.description, source_chunks=e.source_chunks,
            ) for e in entities],
            relations=[RelationResponse(
                id=str(r.id), source_entity=r.source_entity, target_entity=r.target_entity,
                relation_type=r.relation_type, description=r.description, weight=r.weight,
            ) for r in relations],
        )


@router.post("/bases/{kb_id}/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_to_kb(
    kb_id: str,
    file: UploadFile = File(...),
    strategy: str = Form(default="", description="覆盖分块策略: fixed/recursive/markdown/parent_child"),
    current_user: User = Depends(get_current_user),
):
    """上传文件到指定知识库：解析分块 → 向量索引 → 落库文件记录。

    复用旧 /kb/upload 的解析与索引链路，额外在 PostgreSQL 中登记文件，
    使 GET /knowledge/bases/{id}/files 能列出已上传文件。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # 先登记文件记录（pending），索引完成后更新状态
        record = await add_file_record(
            session,
            kb_id=kb.id,
            filename=file.filename,
            file_type=ext.lstrip("."),
            char_count=len(raw),
        )
        await session.commit()

        try:
            from app.rag.chunker import parse_and_chunk
            from app.rag.retriever import get_retriever

            chunks = parse_and_chunk(raw=raw, filename=file.filename, strategy=strategy or None)
            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="File parsed but produced no text chunks.",
                )

            texts = [c[0] for c in chunks]
            metas = [c[1] for c in chunks]
            for m in metas:
                m.setdefault("knowledge_base_id", str(kb.id))

            n = get_retriever().add_documents(texts, metas)

            record.chunk_count = n
            await update_file_status(session, record.id, "completed")

            # 阶段 2C: 图谱抽取（GRAPH_ENABLED 时，失败不阻塞主链路）
            graph_stats = None
            if cfg.GRAPH_ENABLED:
                try:
                    from backend.services.graph_service import extract_graph_from_chunks
                    graph_stats = await extract_graph_from_chunks(
                        session, kb.id, chunks, file.filename,
                    )
                    logger.info("[knowledge/upload] graph: %s", graph_stats)
                except Exception as exc:
                    logger.warning("[knowledge/upload] graph extraction failed: %s", exc)

            await session.commit()

            return UploadResponse(
                file_id=str(record.id),
                indexed=n,
                message=f"Successfully indexed {n} chunks from '{file.filename}'.",
                graph=graph_stats,
            )
        except HTTPException:
            await update_file_status(session, record.id, "failed")
            await session.commit()
            raise
        except Exception as exc:
            logger.error("[knowledge/upload] error: %s", exc)
            await update_file_status(session, record.id, "failed")
            await session.commit()
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
