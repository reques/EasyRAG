"""阶段 1 新 FastAPI 入口 — 整合新旧路由。

启动::

    uvicorn backend.server.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logger import get_logger

cfg = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理。"""
    logger.info("Starting %s v%s", cfg.APP_NAME, cfg.APP_VERSION)

    # 初始化数据库表（开发用，生产用 Alembic 迁移）
    try:
        from backend.storage.postgres.manager import init_db, get_engine
        await init_db()
        logger.info("[lifespan] database tables ensured")

        # 增量列迁移（开发阶段快速迭代；生产用 Alembic）
        from sqlalchemy import text
        engine = await get_engine()
        async with engine.begin() as conn:
            incremental_columns = (
                "text_content TEXT",
                "parser_name VARCHAR(32)",
                "parser_version VARCHAR(64)",
                "parser_task_id VARCHAR(128)",
                "parser_backend VARCHAR(64)",
                "parse_method VARCHAR(32)",
                "parser_warnings TEXT",
                "processing_stage VARCHAR(32)",
                "progress_message VARCHAR(512)",
                "progress_current INTEGER NOT NULL DEFAULT 0",
                "progress_total INTEGER NOT NULL DEFAULT 0",
            )
            for column in incremental_columns:
                await conn.execute(text(
                    f"ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS {column}"
                ))
            # 情景记忆可靠性：conversations.last_summarized_message_id（摘要折叠断点）
            await conn.execute(text(
                "ALTER TABLE conversations "
                "ADD COLUMN IF NOT EXISTS last_summarized_message_id INTEGER"
            ))
            # 图谱实体/关系按文件命名空间隔离（2026-08-25）：
            # 实体身份 = (kb_id, source_file, name)，同名不同文件各自独立成节点。
            await conn.execute(text(
                "ALTER TABLE knowledge_entities "
                "ADD COLUMN IF NOT EXISTS source_file VARCHAR(512)"
            ))
            await conn.execute(text(
                "ALTER TABLE knowledge_relations "
                "ADD COLUMN IF NOT EXISTS source_file VARCHAR(512)"
            ))
            # 老实体回填：source_chunks 格式为 "<filename>#<chunk_index>"，取 # 前部分。
            # 老关系无来源信息，保持 NULL（图谱展示时按 name 唯一匹配兜底）。
            await conn.execute(text(
                "UPDATE knowledge_entities SET source_file = split_part(source_chunks, '#', 1) "
                "WHERE source_file IS NULL AND source_chunks IS NOT NULL "
                "AND source_chunks <> ''"
            ))
            # 老库同 (kb_id, source_file, name) 重复行清理：保留 source_chunks 最长的行
            # （信息最全），删除其余——防止同一文件内同名实体多行残留导致前端节点 id 冲突。
            await conn.execute(text(
                """
                DELETE FROM knowledge_entities a
                USING knowledge_entities b
                WHERE a.id <> b.id
                  AND a.knowledge_base_id = b.knowledge_base_id
                  AND a.source_file IS NOT DISTINCT FROM b.source_file
                  AND a.name = b.name
                  AND (length(coalesce(a.source_chunks, '')) <
                       length(coalesce(b.source_chunks, '')))
                """
            ))
            await conn.execute(text(
                """
                DELETE FROM knowledge_relations a
                USING knowledge_relations b
                WHERE a.id <> b.id
                  AND a.knowledge_base_id = b.knowledge_base_id
                  AND a.source_file IS NOT DISTINCT FROM b.source_file
                  AND a.source_entity = b.source_entity
                  AND a.target_entity = b.target_entity
                  AND a.relation_type = b.relation_type
                """
            ))
        logger.info("[lifespan] incremental migrations applied")
    except Exception as exc:
        logger.warning("[lifespan] database init skipped: %s", exc)

    # 确保 MinIO bucket
    try:
        from backend.storage.minio.client import get_minio_client, ensure_bucket
        ensure_bucket()
        logger.info("[lifespan] minio bucket ensured")
    except Exception as exc:
        logger.warning("[lifespan] minio init skipped: %s", exc)

    # 清理图谱构建的孤儿运行记录（上次进程被强杀/重启时遗留的 running 状态）
    try:
        from backend.services.graph_build_service import mark_interrupted_runs

        cleaned = await mark_interrupted_runs()
        if cleaned:
            logger.info("[lifespan] marked %d interrupted graph build run(s) as failed", cleaned)
    except Exception as exc:
        logger.warning("[lifespan] graph build run cleanup skipped: %s", exc)

    # MCP 外部工具服务：启动所有 enabled 的 server（失败不阻塞应用启动）
    try:
        from app.tools.mcp.manager import get_mcp_manager
        mcp_results = get_mcp_manager().start_all(wait=False)
        logger.info("[lifespan] MCP servers started: %s", mcp_results)
    except Exception as exc:
        logger.warning("[lifespan] MCP init skipped: %s", exc)

    # DeepAgents 可发现性（2026-08-21, S2）：启动时打印执行路径与 SubAgent 名册
    logger.info("[lifespan] AGENT_MODE=%s (auto=智能路由 | single | multi | deepagents)", cfg.AGENT_MODE)
    if cfg.AGENT_MODE == "deepagents":
        try:
            from app.agents.deep.subagents import get_subagents
            roster = ", ".join(s.name for s in get_subagents())
            logger.info("[lifespan] deepagents subagents: %s", roster)
        except Exception as exc:
            logger.warning("[lifespan] deepagents roster unavailable: %s", exc)

    # 文件索引 worker（内嵌进程，2026-08-27）：消费 Redis Stream 处理上传任务。
    # 与 API 同进程运行（uvicorn 重启即 worker 重启；消息在 Redis 中持久化不丢，
    # 重启后 XAUTOCLAIM 认领上次未完成的任务）。失败不阻塞应用启动。
    ingestion_worker_task: Optional[asyncio.Task] = None
    try:
        from backend.worker.ingestion_worker import start_worker

        ingestion_worker_task = start_worker(cfg.INGESTION_CONCURRENCY)
        logger.info("[lifespan] ingestion worker started (concurrency=%d)", cfg.INGESTION_CONCURRENCY)
    except Exception as exc:
        logger.warning("[lifespan] ingestion worker start skipped: %s", exc)

    yield

    # 停止文件索引 worker（先停消费再关 Redis；存量任务排空后返回）
    if ingestion_worker_task is not None:
        try:
            from backend.worker.ingestion_worker import stop_worker

            stop_worker()
            await asyncio.wait_for(ingestion_worker_task, timeout=60)
            logger.info("[lifespan] ingestion worker stopped")
        except asyncio.TimeoutError:
            logger.warning("[lifespan] ingestion worker drain timed out, cancelling")
            ingestion_worker_task.cancel()
        except Exception as exc:
            logger.warning("[lifespan] ingestion worker stop skipped: %s", exc)

    # 清理资源
    try:
        from app.tools.mcp.manager import get_mcp_manager
        get_mcp_manager().stop_all()
        logger.info("[lifespan] MCP servers stopped")
    except Exception:
        pass

    # 清理资源
    try:
        from backend.storage.redis.manager import close_redis
        await close_redis()
    except Exception:
        pass

    logger.info("Shutting down %s", cfg.APP_NAME)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    application = FastAPI(
        title=cfg.APP_NAME,
        version=cfg.APP_VERSION,
        description="企业级多智能体知识库平台 — 阶段 1",
        lifespan=lifespan,
    )

    # CORS
    origins = [o.strip() for o in cfg.CORS_ORIGINS.split(",")]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # 阶段 1 新路由
    # =========================================================================
    from backend.server.routers.auth_router import router as auth_router
    from backend.server.routers.chat_router import router as chat_router
    from backend.server.routers.knowledge_router import router as kb_router
    from backend.server.routers.evaluation_router import router as eval_router
    from backend.server.routers.mcp_router import router as mcp_router

    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(kb_router, prefix="/api/v1")
    application.include_router(eval_router, prefix="/api/v1")
    application.include_router(mcp_router, prefix="/api/v1")

    # =========================================================================
    # 旧路由（保持兼容）— app/api/routes.py + app/api/kb_routes.py
    # =========================================================================
    from app.api.routes import router as legacy_router
    from app.api.kb_routes import router as legacy_kb_router

    application.include_router(legacy_router, prefix="/api/v1")
    application.include_router(legacy_kb_router, prefix="/api/v1")

    return application


app = create_app()
