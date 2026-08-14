"""阶段 1 新 FastAPI 入口 — 整合新旧路由。

启动::

    uvicorn backend.server.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

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

    # MCP 外部工具服务：启动所有 enabled 的 server（失败不阻塞应用启动）
    try:
        from app.tools.mcp.manager import get_mcp_manager
        mcp_results = get_mcp_manager().start_all(wait=False)
        logger.info("[lifespan] MCP servers started: %s", mcp_results)
    except Exception as exc:
        logger.warning("[lifespan] MCP init skipped: %s", exc)

    yield

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
