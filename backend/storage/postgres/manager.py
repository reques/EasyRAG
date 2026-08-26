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
from sqlalchemy import text

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


async def _migrate_legacy_evaluation_runs(conn) -> None:
    """存量库兼容：evaluation_runs.dataset_id 是后加的列。

    create_all 只创建缺失的表，不会给已存在的表补列；
    这里对新旧库都做幂等检查，只有列缺失时才执行 ALTER。
    """
    row = await conn.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'evaluation_runs' "
        "AND column_name = 'dataset_id'"
    ))
    if row.scalar():
        return
    await conn.execute(text(
        "ALTER TABLE evaluation_runs ADD COLUMN dataset_id UUID"
    ))
    await conn.execute(text(
        "CREATE INDEX ix_evaluation_runs_dataset_id "
        "ON evaluation_runs (dataset_id)"
    ))
    await conn.execute(text(
        "ALTER TABLE evaluation_runs ADD CONSTRAINT fk_evaluation_runs_dataset_id "
        "FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets(id) ON DELETE SET NULL"
    ))
    logger.info("[postgres] migrated evaluation_runs.dataset_id")


async def _migrate_custom_model_supports_vision(conn) -> None:
    """Add custom_model_configs.supports_vision to pre-existing databases."""
    row = await conn.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'custom_model_configs' "
        "AND column_name = 'supports_vision'"
    ))
    if row.scalar():
        return
    await conn.execute(text(
        "ALTER TABLE custom_model_configs "
        "ADD COLUMN supports_vision BOOLEAN NOT NULL DEFAULT FALSE"
    ))
    logger.info("[postgres] migrated custom_model_configs.supports_vision")


async def _migrate_messages_image(conn) -> None:
    """Add messages.image for persisted chat images."""
    row = await conn.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'messages' "
        "AND column_name = 'image'"
    ))
    if row.scalar():
        return
    await conn.execute(text(
        "ALTER TABLE messages ADD COLUMN image TEXT"
    ))
    logger.info("[postgres] migrated messages.image")


async def init_db() -> None:
    """创建所有表（开发/测试用；生产应使用 Alembic 迁移）。"""
    from backend.storage.postgres.models_user import User, Department  # noqa: F401
    from backend.storage.postgres.models_conversation import Conversation, Message  # noqa: F401
    from backend.storage.postgres.models_knowledge import KnowledgeBase, KnowledgeFile  # noqa: F401
    from backend.storage.postgres.models_memory import UserFact  # noqa: F401
    from backend.storage.postgres.models_model_config import CustomModelConfig  # noqa: F401
    from backend.storage.postgres.models_skill_config import CustomSkillConfig  # noqa: F401
    from backend.storage.postgres.models_agent_run import Run, Task, AgentRun  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_legacy_evaluation_runs(conn)
        await _migrate_custom_model_supports_vision(conn)
        await _migrate_messages_image(conn)
    logger.info("[postgres] tables created")


def get_session() -> AsyncSession:
    """获取一个异步数据库会话（调用方负责关闭）。"""
    return _async_session_factory()


async def get_engine():
    """返回异步引擎（供 Alembic 等使用）。"""
    return _engine
