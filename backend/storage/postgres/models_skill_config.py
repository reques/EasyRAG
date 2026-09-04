"""个人 Skill 的索引表（2026-09-04 重构：文件为真相，本表降级为索引）。

内容（工作指令 + 工具依赖）存磁盘 ``SKILL.md``，见
``backend/services/skill_config_service.py``。本表只保留：

- ``slug``：目录名，与 owner 联合唯一 —— 列表查询与命名冲突检测的依据；
- 展示元数据（name / description / category / icon）：列表接口不必为了拿
  名字去读全部磁盘文件；
- ``source_type`` / ``is_active``：来源与启停（``shared`` 留作扩展点）。

移除的列（``instructions`` / ``tool_names_json``）由 ``manager.py`` 的
``_migrate_skill_config_to_files`` 在导出到磁盘后 DROP。
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.storage.postgres.manager import Base


class CustomSkillConfig(Base):
    __tablename__ = "custom_skill_configs"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_custom_skill_owner_name"),
        UniqueConstraint("owner_id", "slug", name="uq_custom_skill_owner_slug"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 磁盘目录名：<SKILLS_PERSONAL_DIR>/<owner_id>/<slug>/SKILL.md
    slug: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    category: Mapped[str] = mapped_column(String(32), default="自定义", nullable=False)
    icon: Mapped[str] = mapped_column(String(32), default="sparkles", nullable=False)
    # personal（本期唯一取值）| shared（扩展点：共享范围与 read/manage scope）
    source_type: Mapped[str] = mapped_column(
        String(16), default="personal", nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    @property
    def public_id(self) -> str:
        """对外标识 = slug（重构前是 ``custom:<uuid>``）。

        改成 slug 是为了让"前端选中的 id"、"磁盘目录名"、"read_skill 的参数"
        三者一致 —— 旧的 ``custom:<uuid>`` 形态需要在三处之间来回映射。
        """
        return self.slug
