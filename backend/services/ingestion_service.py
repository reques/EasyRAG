"""文件索引服务 — Redis Stream worker 的消费处理逻辑。

原为 knowledge_router 内 FastAPI BackgroundTasks 的 _run_ingestion（进程内任务，
重启即丢、无并发闸门）；2026-08-27 抽出为独立服务，由 backend.worker 的
ingestion_worker 进程消费队列调用。进度上报机制不变（DB progress + 前端轮询）。
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.parsers import get_parser_router
from backend.repositories.knowledge_repository import KnowledgeFileRepository
from backend.services.knowledge_service import update_file_progress
from backend.storage.postgres.manager import get_session

logger = get_logger(__name__)
cfg = get_settings()


async def fetch_raw_from_minio(minio_bucket: str, minio_object: str) -> Optional[bytes]:
    """从 MinIO 拉取文件原始字节（消费端替代上传请求内存里的 raw）。

    失败返回 None（调用方负责标记 failed，避免任务卡死）。
    """
    try:
        from backend.storage.minio.client import get_minio_client

        client = get_minio_client()
        resp = await asyncio.to_thread(client.get_object, minio_bucket, minio_object)
        try:
            data = await asyncio.to_thread(resp.read)
        finally:
            resp.close()
            resp.release_conn()
        return data
    except Exception as exc:
        logger.error("[ingestion] minio fetch failed %s/%s: %s", minio_bucket, minio_object, exc)
        return None


async def run_ingestion(
    file_id: uuid.UUID,
    kb_id: uuid.UUID,
    raw: bytes,
    filename: str,
    strategy: Optional[str],
    content_type: Optional[str] = None,
    preferred_parser: str = "auto",
) -> None:
    """单文件索引任务：解析分块 → 向量索引 → 图谱抽取，分阶段更新 progress。

    与 worker 解耦：不关心消息从哪来（Redis Stream / 回退后台任务均可调用）。
    """
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
        batch_size = 64
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
