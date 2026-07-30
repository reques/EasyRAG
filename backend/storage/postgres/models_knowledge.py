"""知识库与文件模型。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.storage.postgres.manager import Base


class KnowledgeBase(Base):
    """知识库 — 一组文档的集合。"""

    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 向量库类型和集合名
    vector_store_type: Mapped[str] = mapped_column(
        String(32), default="milvus", nullable=False
    )
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)

    # 所属用户/部门
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    department_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 关联
    files: Mapped[list["KnowledgeFile"]] = relationship(
        "KnowledgeFile", back_populates="knowledge_base",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase {self.name}>"


class KnowledgeFile(Base):
    """知识库文件 — 已上传并索引的文档。"""

    __tablename__ = "knowledge_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)  # txt, md, pdf, docx

    # MinIO 对象路径
    minio_bucket: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    minio_object: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # 索引统计
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 处理状态
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False
    )  # pending / processing / completed / failed

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 关联
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="files"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeFile {self.filename}>"
