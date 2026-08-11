"""PostgreSQL 数据库管理器 — 异步 SQLAlchemy 引擎 + 会话工厂。

用法::

    from backend.storage.postgres.manager import get_session, init_db
    await init_db()
    async with get_session() as session:
        ...
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

DATABASE_URL = (
    f"postgresql+asyncpg://{cfg.POSTGRES_USER}:{cfg.POSTGRES_PASSWORD}"
    f"@{cfg.POSTGRES_HOST}:{cfg.POSTGRES_PORT}/{cfg.POSTGRES_DB}"
)

_engine = create_async_engine(
    DATABASE_URL,
    pool_size=cfg.POSTGRES_POOL_SIZE,
    max_overflow=cfg.POSTGRES_MAX_OVERFLOW,
    echo=cfg.DEBUG,
)

_async_session_factory = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""
    pass


async def init_db() -> None:
    """创建所有表（开发/测试用；生产应使用 Alembic 迁移）。"""
    from backend.storage.postgres.models_user import User, Department  # noqa: F401
    from backend.storage.postgres.models_conversation import Conversation, Message  # noqa: F401
    from backend.storage.postgres.models_knowledge import KnowledgeBase, KnowledgeFile  # noqa: F401
    from backend.storage.postgres.models_memory import UserFact  # noqa: F401
    from backend.storage.postgres.models_model_config import CustomModelConfig  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("[postgres] tables created")


def get_session() -> AsyncSession:
    """获取一个异步数据库会话（调用方负责关闭）。"""
    return _async_session_factory()


async def get_engine():
    """返回异步引擎（供 Alembic 等使用）。"""
    return _engine
