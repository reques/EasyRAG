"""知识库路由 — 知识库 CRUD + 文件上传索引。"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
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
from backend.repositories.knowledge_repository import (
    KnowledgeBaseRepository,
    KnowledgeFileRepository,
)
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
    progress: int = 0
    error_message: Optional[str] = None


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
                progress=f.progress,
                error_message=f.error_message,
            )
            for f in files
        ]


class UploadResponse(BaseModel):
    file_id: str
    indexed: int
    message: str
    graph: Optional[dict] = None   # GRAPH_ENABLED 时: {"entities": n, "relations": m}
    status: str = "completed"      # 异步模式: 立即返回 "processing"


class FilePreviewResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    content_type: str          # "text" | "image" | "pdf_text"
    text_content: Optional[str] = None
    # 前端根据 content_type 决定展示方式


class FileContentResponse(BaseModel):
    """二进制文件内容（图片等）直接流式返回，不走 JSON。"""


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


@router.post("/bases/{kb_id}/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_to_kb(
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str = Form(default="", description="覆盖分块策略: fixed/recursive/markdown/parent_child"),
    current_user: User = Depends(get_current_user),
):
    """上传文件到指定知识库：立即登记记录 + 存 MinIO，索引放后台任务。

    返回 202 + file_id，前端轮询 GET /bases/{id}/files 按 status/progress
    渲染进度条，直到 status=completed / failed。
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

        # 登记文件记录（pending → 后台任务推进 progress）
        record = await add_file_record(
            session,
            kb_id=kb.id,
            filename=file.filename,
            file_type=ext.lstrip("."),
            char_count=len(raw),
        )
        await session.commit()
        # commit 后立即取出纯值，避免后续异常时访问 expired/回滚 session 上的 ORM 属性
        file_id = record.id
        kb_uuid = kb.id
        filename = file.filename

        # 存入 MinIO 以便后续预览（同步完成，失败不阻塞索引）
        minio_bucket = cfg.MINIO_BUCKET
        minio_object = f"kb/{kb_uuid}/{file_id}/{filename}"
        try:
            from backend.storage.minio.client import get_minio_client
            import io as std_io
            client = get_minio_client()
            client.put_object(
                bucket_name=minio_bucket,
                object_name=minio_object,
                data=std_io.BytesIO(raw),
                length=len(raw),
                content_type=file.content_type or "application/octet-stream",
            )
            # put_object 成功后单独一个短事务更新 minio 字段
            await session.refresh(record)
            record.minio_bucket = minio_bucket
            record.minio_object = minio_object
            await session.commit()
            logger.info("[knowledge/upload] stored in MinIO: %s/%s", minio_bucket, minio_object)
        except Exception as exc:
            # 必须 rollback：否则 session 进入 PendingRollback 状态，
            # 后续任何 ORM 属性访问都会抛 PendingRollbackError → 500
            await session.rollback()
            logger.warning("[knowledge/upload] MinIO store failed (preview unavailable): %s", exc)

    # 后台任务：解析分块 → 向量索引 → 图谱抽取，分阶段更新 progress
    background_tasks.add_task(
        _run_ingestion, file_id, kb_uuid, raw, filename, strategy or None
    )

    return UploadResponse(
        file_id=str(file_id),
        indexed=0,
        message=f"File '{filename}' accepted, indexing in background.",
        status="processing",
    )


async def _run_ingestion(
    file_id: uuid.UUID,
    kb_id: uuid.UUID,
    raw: bytes,
    filename: str,
    strategy: Optional[str],
) -> None:
    """后台索引任务：每阶段独立 session 提交进度，供前端轮询。"""
    from backend.services.knowledge_service import update_file_progress

    try:
        # 阶段 1: 解析分块 (10%)
        async with get_session() as s:
            await update_file_progress(s, file_id, 10, status="processing")

        from app.rag.chunker import parse_and_chunk
        chunks = parse_and_chunk(raw=raw, filename=filename, strategy=strategy)
        if not chunks:
            async with get_session() as s:
                await update_file_progress(
                    s, file_id, 100, status="failed",
                    error_message="文件解析后没有产生文本块",
                )
            return

        # 阶段 2: 存储全文用于预览 (30%)
        async with get_session() as s:
            repo = KnowledgeFileRepository(s)
            f = await repo.get_by_id(file_id)
            if f:
                try:
                    from app.rag.chunker import extract_text as _extract_full
                    f.text_content = _extract_full(raw, filename)
                except Exception as exc:
                    logger.warning("[ingestion] preview text store failed: %s", exc)
                await update_file_progress(s, file_id, 30)

        # 阶段 3: 向量索引 (30% → 80%，embedding 是最耗时阶段)
        from app.rag.retriever import get_retriever
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        for m in metas:
            m.setdefault("knowledge_base_id", str(kb_id))

        n = get_retriever().add_documents(texts, metas)

        async with get_session() as s:
            repo = KnowledgeFileRepository(s)
            f = await repo.get_by_id(file_id)
            if f:
                f.chunk_count = n
            await update_file_progress(s, file_id, 80)

        # 阶段 4: 图谱抽取 (80% → 100%，GRAPH_ENABLED 时)
        if cfg.GRAPH_ENABLED:
            try:
                from backend.services.graph_service import extract_graph_from_chunks
                async with get_session() as s:
                    await extract_graph_from_chunks(s, kb_id, chunks, filename)
                    await s.commit()
            except Exception as exc:
                logger.warning("[ingestion] graph extraction failed: %s", exc)

        async with get_session() as s:
            await update_file_progress(s, file_id, 100, status="completed")
        logger.info("[ingestion] completed: %s (%d chunks)", filename, n)

    except Exception as exc:
        logger.error("[ingestion] failed: %s — %s", filename, exc)
        try:
            async with get_session() as s:
                await update_file_progress(
                    s, file_id, 100, status="failed",
                    error_message=str(exc)[:500],
                )
        except Exception:
            logger.exception("[ingestion] failed to persist error status")


# ── 文件预览端点 ─────────────────────────────────────────────────────────────

@router.get("/bases/{kb_id}/files/{file_id}/preview", response_model=FilePreviewResponse)
async def preview_file(
    kb_id: str,
    file_id: str,
    current_user: User = Depends(get_current_user),
):
    """预览文件内容 — 按文件类型提取文本或返回图片信息。

    content_type 取值:
      - "text"      : txt/md/docx 等文本格式，text_content 为完整文本
      - "pdf_text"  : PDF 提取文本，text_content 为逐页文本
      - "image"     : 图片文件，text_content 为空（前端直接读取 /raw 端点）
    """
    from backend.repositories.knowledge_repository import KnowledgeFileRepository

    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        file_repo = KnowledgeFileRepository(session)
        f = await file_repo.get_by_id(uuid.UUID(file_id))
        if not f or f.knowledge_base_id != kb.id:
            raise HTTPException(status_code=404, detail="File not found")

        file_type = f.file_type.lower()

        # ── 图片：前端应请求 /raw 端点获取二进制，这里仅返回元信息 ──
        if file_type in ("png", "jpg", "jpeg", "bmp", "webp"):
            return FilePreviewResponse(
                file_id=str(f.id),
                filename=f.filename,
                file_type=file_type,
                content_type="image",
            )

        # ── 文本格式：优先读 text_content 列（最快），否则从 MinIO 提取 ──
        text = None
        if f.text_content:
            text = f.text_content
        elif f.minio_bucket and f.minio_object:
            try:
                from backend.storage.minio.client import get_minio_client
                client = get_minio_client()
                response = client.get_object(f.minio_bucket, f.minio_object)
                raw = response.read()
                response.close()
                response.release_conn()
                from app.rag.chunker import extract_text
                text = extract_text(raw, f.filename)
            except Exception as exc:
                logger.warning("[preview] MinIO read failed: %s", exc)

        if text is None:
            raise HTTPException(
                status_code=404,
                detail="此文件暂无可预览内容。旧版本上传的文件请重新上传以启用预览。",
            )

        content_type = "pdf_text" if file_type == "pdf" else "text"

        return FilePreviewResponse(
            file_id=str(f.id),
            filename=f.filename,
            file_type=file_type,
            content_type=content_type,
            text_content=text,
        )


@router.get("/bases/{kb_id}/files/{file_id}/raw")
async def raw_file(
    kb_id: str,
    file_id: str,
    current_user: User = Depends(get_current_user),
):
    """返回图片文件的原始二进制数据（带正确的 Content-Type）。

    非图片文件也支持，但主要用于前端 <img> 直接引用。
    """
    from fastapi.responses import Response
    from backend.repositories.knowledge_repository import KnowledgeFileRepository

    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        file_repo = KnowledgeFileRepository(session)
        f = await file_repo.get_by_id(uuid.UUID(file_id))
        if not f or f.knowledge_base_id != kb.id:
            raise HTTPException(status_code=404, detail="File not found")

        if not f.minio_bucket or not f.minio_object:
            raise HTTPException(status_code=404, detail="File content not available")

        try:
            from backend.storage.minio.client import get_minio_client
            client = get_minio_client()
            resp = client.get_object(f.minio_bucket, f.minio_object)
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            resp.close()
            resp.release_conn()
        except Exception as exc:
            logger.error("[raw] MinIO read failed: %s", exc)
            raise HTTPException(status_code=404, detail="File not found in storage")

        _mime_map = {
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "bmp": "image/bmp", "webp": "image/webp", "txt": "text/plain; charset=utf-8",
            "md": "text/plain; charset=utf-8", "pdf": "application/pdf",
        }
        mime = _mime_map.get(f.file_type.lower(), content_type)

        # PDF 需要 inline 以便 iframe 渲染，不强制 download
        from urllib.parse import quote
        headers = {}
        if f.file_type.lower() != "pdf":
            # 非 PDF 显式设置 attachment 防止意外导航（但前端用 blob，这里仅作兜底）
            safe_name = quote(f.filename.encode("utf-8"))
            headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

        return Response(content=raw, media_type=mime, headers=headers)


# ── 文件删除端点 ─────────────────────────────────────────────────────────────

@router.delete("/bases/{kb_id}/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    kb_id: str,
    file_id: str,
    current_user: User = Depends(get_current_user),
):
    """删除知识库文件：向量索引 + MinIO 对象 + PostgreSQL 记录。

    注意：向量删除按 metadata.source 匹配文件名，若同名文件多次上传会全部清除。
    """
    from backend.repositories.knowledge_repository import KnowledgeFileRepository

    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        file_repo = KnowledgeFileRepository(session)
        f = await file_repo.get_by_id(uuid.UUID(file_id))
        if not f or f.knowledge_base_id != kb.id:
            raise HTTPException(status_code=404, detail="File not found")

        filename = f.filename

        # 1. 删除向量索引（按 source 文件名匹配）
        try:
            from app.rag.retriever import get_retriever
            n = get_retriever().delete_documents_by_source(filename)
            logger.info("[knowledge/delete] removed %d vector chunks for '%s'", n, filename)
        except NotImplementedError:
            logger.warning("[knowledge/delete] vector backend does not support per-file delete")
        except Exception as exc:
            logger.error("[knowledge/delete] vector delete failed: %s", exc)
            # 不阻塞——继续删 MinIO 和 DB

        # 2. 删除 MinIO 对象
        if f.minio_bucket and f.minio_object:
            try:
                from backend.storage.minio.client import get_minio_client
                client = get_minio_client()
                client.remove_object(f.minio_bucket, f.minio_object)
                logger.info("[knowledge/delete] removed MinIO object: %s/%s", f.minio_bucket, f.minio_object)
            except Exception as exc:
                logger.warning("[knowledge/delete] MinIO delete failed: %s", exc)

        # 3. 删除 PostgreSQL 记录
        await file_repo.delete(f)
        await session.commit()

        logger.info("[knowledge/delete] file '%s' deleted from kb '%s'", filename, kb.name)
        return None
