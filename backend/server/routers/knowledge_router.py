"""知识库路由 — 知识库 CRUD + 文件上传索引。"""

from __future__ import annotations

import asyncio
from time import perf_counter
import uuid
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.parsers import UnsupportedDocumentError, get_parser_router
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
    parser_name: Optional[str] = None
    parser_version: Optional[str] = None
    parser_task_id: Optional[str] = None
    parser_backend: Optional[str] = None
    parse_method: Optional[str] = None
    parser_warnings: Optional[str] = None
    processing_stage: Optional[str] = None
    progress_message: Optional[str] = None
    progress_current: int = 0
    progress_total: int = 0


class RetrievalTestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class RetrievalHitResponse(BaseModel):
    rank: int
    chunk_id: str
    content: str
    score: float
    source: Optional[str] = None
    file_id: Optional[str] = None
    chunk_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None
    parser_name: Optional[str] = None
    retrieval_path: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestResponse(BaseModel):
    query: str
    knowledge_base_id: str
    top_k: int
    score_threshold: float
    elapsed_ms: int
    total: int
    results: list[RetrievalHitResponse]


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
                parser_name=f.parser_name,
                parser_version=f.parser_version,
                parser_task_id=f.parser_task_id,
                parser_backend=f.parser_backend,
                parse_method=f.parse_method,
                parser_warnings=f.parser_warnings,
                processing_stage=f.processing_stage,
                progress_message=f.progress_message,
                progress_current=f.progress_current,
                progress_total=f.progress_total,
            )
            for f in files
        ]


@router.post(
    "/bases/{kb_id}/retrieval/test",
    response_model=RetrievalTestResponse,
)
async def test_retrieval(
    kb_id: uuid.UUID,
    req: RetrievalTestRequest,
    current_user: User = Depends(get_current_user),
):
    """在单个已授权知识库内执行一次轻量级向量检索测试。"""
    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(kb_id)
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

    from app.rag.retriever import get_document_chunk_id, get_retriever

    started_at = perf_counter()
    try:
        docs = await asyncio.to_thread(
            get_retriever().retrieve,
            req.query,
            top_k=req.top_k,
            knowledge_base_ids=[str(kb_id)],
            # The frontend uses 0 to mean "no filtering". Cosine/IP scores can
            # be negative, so pass the true lower bound instead of zero.
            score_threshold=(
                -1.0 if req.score_threshold == 0 else req.score_threshold
            ),
        )
    except Exception as exc:
        logger.exception(
            "[knowledge/retrieval-test] failed for kb=%s: %s", kb_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service unavailable",
        ) from exc

    results: list[RetrievalHitResponse] = []
    for doc in docs:
        metadata = dict(doc.get("metadata") or {})
        score = float(metadata.pop("score", 0.0))
        # parent_text may duplicate the complete result content and make the
        # debugging response unexpectedly large.
        metadata.pop("parent_text", None)
        results.append(
            RetrievalHitResponse(
                rank=len(results) + 1,
                chunk_id=get_document_chunk_id(
                    kb_id,
                    str(doc.get("content") or ""),
                    metadata,
                ),
                content=str(doc.get("content") or ""),
                score=score,
                source=metadata.get("source"),
                file_id=_optional_string(metadata.get("file_id")),
                chunk_index=_optional_int(metadata.get("chunk_index")),
                page_start=_optional_int(metadata.get("page_start")),
                page_end=_optional_int(metadata.get("page_end")),
                section_path=_optional_string(metadata.get("section_path")),
                parser_name=_optional_string(metadata.get("parser_name")),
                retrieval_path=_optional_string(metadata.get("retrieval_path")),
                metadata=metadata,
            )
        )

    elapsed_ms = max(0, round((perf_counter() - started_at) * 1000))
    return RetrievalTestResponse(
        query=req.query,
        knowledge_base_id=str(kb_id),
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        elapsed_ms=elapsed_ms,
        total=len(results),
        results=results,
    )


def _optional_string(value: Any) -> Optional[str]:
    return None if value is None or value == "" else str(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


ALLOWED_EXTENSIONS = get_parser_router().supported_extensions


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


# ── GraphRAG 阶段 5: Neo4j 图谱管理 ────────────────────────────────────────

class GraphConfigResponse(BaseModel):
    graph_enabled: bool
    neo4j_uri: str
    neo4j_connected: bool
    extractors: list[str]
    entity_collection: str


class GraphBuildResponse(BaseModel):
    run_id: str
    status: str


class GraphStatusResponse(BaseModel):
    run: Optional[dict] = None
    neo4j: dict = {}
    indexed: int = 0
    pg_entities: int = 0
    pg_relations: int = 0


class GraphSearchResponse(BaseModel):
    entities: list[dict]
    nodes: list[dict]
    edges: list[dict]


async def _require_owned_kb(kb_id: str, current_user: User):
    """加载并校验知识库归属，返回 KnowledgeBase 或抛 404。"""
    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(uuid.UUID(kb_id))
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        return kb


@router.get("/bases/{kb_id}/graph/config", response_model=GraphConfigResponse)
async def get_graph_config(
    kb_id: str,
    current_user: User = Depends(get_current_user),
):
    """图谱配置：开关、Neo4j 连接状态、可用抽取器。"""
    await _require_owned_kb(kb_id, current_user)
    from backend.storage.neo4j.client import get_neo4j_client

    try:
        connected = get_neo4j_client().available
    except Exception:
        connected = False
    return GraphConfigResponse(
        graph_enabled=cfg.GRAPH_ENABLED,
        neo4j_uri=cfg.NEO4J_URI,
        neo4j_connected=connected,
        extractors=["llm"],
        entity_collection=cfg.GRAPH_ENTITY_COLLECTION,
    )


@router.post(
    "/bases/{kb_id}/graph/build",
    response_model=GraphBuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def build_kb_graph(
    kb_id: str,
    background_tasks: BackgroundTasks,
    extractor: str = Form(default="llm", description="抽取器: llm"),
    current_user: User = Depends(get_current_user),
):
    """从已入库 chunks 触发图谱构建（后台任务）。返回 run_id 供轮询状态。"""
    kb = await _require_owned_kb(kb_id, current_user)
    from backend.services.graph_build_service import create_build_run, run_build

    try:
        run_id = await create_build_run(kb.id, extractor)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(run_build, run_id)
    return GraphBuildResponse(run_id=str(run_id), status="pending")


@router.get("/bases/{kb_id}/graph/status", response_model=GraphStatusResponse)
async def get_graph_status(
    kb_id: str,
    current_user: User = Depends(get_current_user),
):
    """图谱构建状态与统计：最新 run + Neo4j/PG/Milvus 三方计数。"""
    kb = await _require_owned_kb(kb_id, current_user)
    from sqlalchemy import func, select
    from backend.services.graph_build_service import latest_build_run
    from backend.storage.postgres.models_knowledge import (
        KnowledgeEntity,
        KnowledgeRelation,
    )

    run = await latest_build_run(kb.id)
    run_dict = None
    if run:
        run_dict = {
            "id": str(run.id),
            "status": run.status,
            "extractor": run.extractor,
            "total_chunks": run.total_chunks,
            "processed_chunks": run.processed_chunks,
            "entities_found": run.entities_found,
            "relations_found": run.relations_found,
            "entities_indexed": run.entities_indexed,
            "relations_indexed": run.relations_indexed,
            "error_message": run.error_message,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }

    neo4j_stats: dict = {}
    try:
        from backend.storage.neo4j.client import get_neo4j_client

        client = get_neo4j_client()
        if client.available:
            neo4j_stats = client.count_stats(str(kb.id))
    except Exception as exc:
        logger.warning("[graph] neo4j stats failed: %s", exc)

    indexed = 0
    try:
        from app.rag.graph_vector_index import get_graph_vector_index

        indexed = get_graph_vector_index().count(str(kb.id))
    except Exception as exc:
        logger.warning("[graph] milvus graph index stats failed: %s", exc)

    async with get_session() as session:
        pg_entities = (await session.execute(
            select(func.count()).select_from(KnowledgeEntity).where(
                KnowledgeEntity.knowledge_base_id == kb.id
            )
        )).scalar_one()
        pg_relations = (await session.execute(
            select(func.count()).select_from(KnowledgeRelation).where(
                KnowledgeRelation.knowledge_base_id == kb.id
            )
        )).scalar_one()

    return GraphStatusResponse(
        run=run_dict,
        neo4j=neo4j_stats,
        indexed=indexed,
        pg_entities=pg_entities,
        pg_relations=pg_relations,
    )


@router.get("/bases/{kb_id}/graph/search", response_model=GraphSearchResponse)
async def search_kb_graph(
    kb_id: str,
    q: str = Query(default="", max_length=100, description="实体名关键词"),
    depth: int = Query(default=1, ge=1, le=3, description="子图扩展深度"),
    current_user: User = Depends(get_current_user),
):
    """子图搜索：实体名模糊匹配 → 以首个命中实体为中心扩展子图。"""
    kb = await _require_owned_kb(kb_id, current_user)
    from backend.storage.neo4j.client import Neo4jUnavailableError, get_neo4j_client

    try:
        client = get_neo4j_client()
        if not client.available:
            raise HTTPException(status_code=503, detail="Neo4j 未连接，请先启动 neo4j 服务")
    except Neo4jUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j 未连接: {exc}")

    entities = client.search_entities(str(kb.id), q, limit=10) if q else []
    nodes: list = []
    edges: list = []
    if entities:
        subgraph = client.get_subgraph(
            str(kb.id), entities[0]["name"], depth=depth, max_nodes=60
        )
        nodes, edges = subgraph["nodes"], subgraph["edges"]
    return GraphSearchResponse(entities=entities, nodes=nodes, edges=edges)


@router.delete("/bases/{kb_id}/graph", status_code=status.HTTP_204_NO_CONTENT)
async def reset_kb_graph(
    kb_id: str,
    current_user: User = Depends(get_current_user),
):
    """重置图谱：清空 Neo4j 子图 + Milvus 语义索引 + PG 本体 + 内存缓存。"""
    kb = await _require_owned_kb(kb_id, current_user)
    from backend.services.graph_build_service import reset_kb_graph as _reset

    await _reset(kb.id)


@router.post("/bases/{kb_id}/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_to_kb(
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str = Form(default="", description="覆盖分块策略: fixed/recursive/markdown/parent_child"),
    parser: str = Form(default="auto", description="解析器: auto/mineru/local"),
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

    try:
        selected_parser = get_parser_router().select_parser(
            file.filename,
            preferred_parser=parser,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

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
        record.parser_name = selected_parser.parser_name
        if selected_parser.parser_name == "mineru":
            record.parser_backend = cfg.MINERU_BACKEND
        record.processing_stage = "queued"
        record.progress_message = "文件已接收，等待开始处理"
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
        _run_ingestion,
        file_id,
        kb_uuid,
        raw,
        filename,
        strategy or None,
        file.content_type,
        parser,
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
    content_type: Optional[str] = None,
    preferred_parser: str = "auto",
) -> None:
    """后台索引任务：每阶段独立 session 提交进度，供前端轮询。"""
    from backend.services.knowledge_service import update_file_progress

    try:
        # 阶段 1: 先记录计划使用的解析器，让前端在耗时解析期间可见。
        parser_router = get_parser_router()
        selected_parser = parser_router.select_parser(
            filename,
            preferred_parser=preferred_parser,
        )
        async with get_session() as s:
            repo = KnowledgeFileRepository(s)
            f = await repo.get_by_id(file_id)
            if f:
                f.parser_name = selected_parser.parser_name
                if selected_parser.parser_name == "mineru":
                    f.parser_backend = cfg.MINERU_BACKEND
            parser_label = (
                "MinerU" if selected_parser.parser_name == "mineru" else "本地解析器"
            )
            await update_file_progress(
                s,
                file_id,
                10,
                status="processing",
                stage="parsing",
                message=f"{parser_label} 正在解析文档",
                current=0,
                total=0,
            )
        logger.info(
            "[ingestion] parser selected: %s -> %s (requested=%s)",
            filename,
            selected_parser.parser_name,
            preferred_parser,
        )

        from app.rag.chunker import chunk_parsed_document

        parsed_document = await parser_router.parse(
            raw,
            filename,
            content_type=content_type,
            preferred_parser=preferred_parser,
        )
        chunks = chunk_parsed_document(parsed_document, strategy=strategy)
        if not chunks:
            async with get_session() as s:
                await update_file_progress(
                    s, file_id, 100, status="failed",
                    error_message="文件解析后没有产生文本块",
                    stage="failed",
                    message="解析完成，但没有生成可索引的内容",
                )
            return

        # 阶段 2: 存储全文用于预览 (30%)
        async with get_session() as s:
            repo = KnowledgeFileRepository(s)
            f = await repo.get_by_id(file_id)
            if f:
                f.text_content = parsed_document.text
                f.char_count = len(parsed_document.text)
                provenance = parsed_document.provenance
                f.parser_name = provenance.parser_name
                f.parser_version = provenance.parser_version
                f.parser_task_id = provenance.task_id
                f.parser_backend = provenance.backend
                f.parse_method = provenance.parse_method
                f.parser_warnings = (
                    "\n".join(parsed_document.warnings)
                    if parsed_document.warnings
                    else None
                )
                await update_file_progress(
                    s,
                    file_id,
                    30,
                    stage="chunking",
                    message=f"解析完成，共生成 {len(chunks)} 个内容块",
                    current=len(chunks),
                    total=len(chunks),
                )
        logger.info(
            "[ingestion] parsed: %s with %s %s (task=%s)",
            filename,
            parsed_document.provenance.parser_name,
            parsed_document.provenance.parser_version or "",
            parsed_document.provenance.task_id or "-",
        )

        # 阶段 3: 向量索引 (30% → 80%，embedding 是最耗时阶段)
        from app.rag.retriever import get_retriever
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        for m in metas:
            m["knowledge_base_id"] = str(kb_id)
            m["file_id"] = str(file_id)

        retriever = get_retriever()
        n = 0
        processed = 0
        batch_size = 16
        total_chunks = len(texts)
        async with get_session() as s:
            await update_file_progress(
                s,
                file_id,
                30,
                stage="indexing",
                message=f"正在生成向量并写入索引 0/{total_chunks}",
                current=0,
                total=total_chunks,
            )

        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            added = await asyncio.to_thread(
                retriever.add_documents,
                texts[start:end],
                metas[start:end],
            )
            n += added
            processed = end
            vector_progress = min(80, 30 + int(50 * processed / total_chunks))
            async with get_session() as s:
                repo = KnowledgeFileRepository(s)
                f = await repo.get_by_id(file_id)
                if f:
                    f.chunk_count = n
                await update_file_progress(
                    s,
                    file_id,
                    vector_progress,
                    stage="indexing",
                    message=f"正在生成向量并写入索引 {processed}/{total_chunks}",
                    current=processed,
                    total=total_chunks,
                )

        # 阶段 4: 图谱抽取 (80% → 100%，GRAPH_ENABLED 时)
        if cfg.GRAPH_ENABLED:
            try:
                from backend.services.graph_service import extract_graph_from_chunks

                async def report_graph_progress(
                    current: int,
                    total: int,
                    message: str,
                ) -> None:
                    graph_progress = min(
                        99,
                        80 + int(19 * current / max(total, 1)),
                    )
                    async with get_session() as progress_session:
                        await update_file_progress(
                            progress_session,
                            file_id,
                            graph_progress,
                            stage="graph",
                            message=message,
                            current=current,
                            total=total,
                        )

                async with get_session() as s:
                    await extract_graph_from_chunks(
                        s,
                        kb_id,
                        chunks,
                        filename,
                        progress_callback=report_graph_progress,
                    )
                    await s.commit()
            except Exception as exc:
                logger.warning("[ingestion] graph extraction failed: %s", exc)

        async with get_session() as s:
            await update_file_progress(
                s,
                file_id,
                100,
                status="completed",
                stage="completed",
                message="文档解析和索引已完成",
                current=n,
                total=n,
            )
        logger.info("[ingestion] completed: %s (%d chunks)", filename, n)

    except Exception as exc:
        logger.error("[ingestion] failed: %s — %s", filename, exc)
        try:
            async with get_session() as s:
                await update_file_progress(
                    s, file_id, 100, status="failed",
                    error_message=str(exc)[:500],
                    stage="failed",
                    message="处理失败",
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
