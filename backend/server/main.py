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
        from backend.storage.postgres.manager import init_db
        await init_db()
        logger.info("[lifespan] database tables ensured")
    except Exception as exc:
        logger.warning("[lifespan] database init skipped: %s", exc)

    # 确保 MinIO bucket
    try:
        from backend.storage.minio.client import get_minio_client, ensure_bucket
        ensure_bucket()
        logger.info("[lifespan] minio bucket ensured")
    except Exception as exc:
        logger.warning("[lifespan] minio init skipped: %s", exc)

    yield

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

    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(kb_router, prefix="/api/v1")
    application.include_router(eval_router, prefix="/api/v1")

    # =========================================================================
    # 旧路由（保持兼容）— app/api/routes.py + app/api/kb_routes.py
    # =========================================================================
    from app.api.routes import router as legacy_router
    from app.api.kb_routes import router as legacy_kb_router

    application.include_router(legacy_router, prefix="/api/v1")
    application.include_router(legacy_kb_router, prefix="/api/v1")

    return application


app = create_app()
