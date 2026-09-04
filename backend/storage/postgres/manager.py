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


async def _migrate_skill_config_to_files(conn) -> None:
    """Skill 内容从 DB 列迁到磁盘 SKILL.md（2026-09-04 Skill 重构）。

    ``custom_skill_configs`` 降级为索引表：``instructions`` 与
    ``tool_names_json`` 导出到 ``<SKILLS_PERSONAL_DIR>/<owner_id>/<slug>/SKILL.md``
    后 DROP，新增 ``slug`` 与 ``source_type``。

    顺序是关键：**先导出全部行，全部成功后才删列**。任一行失败则保留旧列并
    记 ERROR，下次启动重试 —— 先删后写会在中途失败时丢数据。

    幂等：``slug`` 列已存在即视为迁移完成直接返回。
    """
    row = await conn.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'custom_skill_configs' AND column_name = 'slug'"
    ))
    if row.scalar():
        return

    has_instructions = (await conn.execute(text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'custom_skill_configs' AND column_name = 'instructions'"
    ))).scalar()

    await conn.execute(text(
        "ALTER TABLE custom_skill_configs ADD COLUMN slug VARCHAR(128)"
    ))
    await conn.execute(text(
        "ALTER TABLE custom_skill_configs "
        "ADD COLUMN source_type VARCHAR(16) NOT NULL DEFAULT 'personal'"
    ))
    # 存量 name 多为中文，slugify 会得到空串 —— 用 id 前缀保证唯一且路径安全
    await conn.execute(text(
        "UPDATE custom_skill_configs "
        "SET slug = 'skill-' || substr(replace(id::text, '-', ''), 1, 8) "
        "WHERE slug IS NULL"
    ))
    await conn.execute(text(
        "ALTER TABLE custom_skill_configs ALTER COLUMN slug SET NOT NULL"
    ))
    await conn.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_custom_skill_owner_slug "
        "ON custom_skill_configs (owner_id, slug)"
    ))

    if not has_instructions:
        # 新库：没有旧列可导出，建表时就是新结构
        logger.info("[postgres] custom_skill_configs: added slug/source_type (fresh)")
        return

    rows = (await conn.execute(text(
        "SELECT id, owner_id, slug, name, description, instructions, "
        "tool_names_json, category, icon FROM custom_skill_configs"
    ))).mappings().all()

    exported, failed = 0, 0
    for record in rows:
        try:
            _export_skill_row_to_disk(record)
            exported += 1
        except Exception as exc:
            failed += 1
            logger.error(
                "[postgres] skill export failed for %s (slug=%s): %s",
                record["id"], record["slug"], exc,
            )

    if failed:
        logger.error(
            "[postgres] %d/%d skills failed to export; keeping legacy columns "
            "(instructions/tool_names_json) for retry on next startup",
            failed, len(rows),
        )
        return

    await conn.execute(text(
        "ALTER TABLE custom_skill_configs "
        "DROP COLUMN instructions, DROP COLUMN tool_names_json"
    ))
    logger.info(
        "[postgres] migrated custom_skill_configs to files: %d exported", exported
    )


def _export_skill_row_to_disk(record) -> None:
    """把一行存量 Skill 写成 SKILL.md（迁移用；不校验工具名是否仍已注册）。

    刻意不做工具名校验：存量 Skill 可能引用已下线的 MCP 工具，校验失败会让
    整次迁移卡住。坏的工具依赖在运行时被门控自然忽略（不在注册表 = 不可用）。
    """
    import json

    from app.skills.loader import SKILL_FILENAME, render_skill_markdown
    from app.skills.registry import personal_dir

    try:
        tool_names = json.loads(record["tool_names_json"] or "[]")
        if not isinstance(tool_names, list):
            tool_names = []
    except (TypeError, ValueError):
        logger.warning(
            "[postgres] skill %s has corrupt tool_names_json; exporting with none",
            record["id"],
        )
        tool_names = []

    directory = personal_dir(str(record["owner_id"]), record["slug"])
    directory.mkdir(parents=True, exist_ok=True)
    text_content = render_skill_markdown(
        name=record["name"],
        description=record["description"] or "",
        body=record["instructions"] or "",
        slug=record["slug"],
        tool_dependencies=[str(t) for t in tool_names],
        category=record["category"] or "自定义",
        icon=record["icon"] or "sparkles",
    )
    (directory / SKILL_FILENAME).write_text(text_content, encoding="utf-8")


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
        await _migrate_skill_config_to_files(conn)
    logger.info("[postgres] tables created")


def get_session() -> AsyncSession:
    """获取一个异步数据库会话（调用方负责关闭）。"""
    return _async_session_factory()


async def get_engine():
    """返回异步引擎（供 Alembic 等使用）。"""
    return _engine
