"""知识库与文件模型。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
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

    # 提取的纯文本内容（预览兜底；MinIO 不可用时直接读此列）
    text_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 文档解析来源。保留实际生效的解析器信息，方便界面展示与问题追踪。
    parser_name: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    parser_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    parser_task_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    parser_backend: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    parse_method: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    parser_warnings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 索引统计
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 处理状态
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False
    )  # pending / processing / completed / failed
    # 处理进度 0-100（后台索引任务推进时更新，供前端进度条轮询）
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_stage: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    progress_message: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 失败原因（status=failed 时记录，便于前端展示）
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # 关联
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="files"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeFile {self.filename}>"


class KnowledgeEntity(Base):
    """知识图谱实体（阶段 2C）— 从文档 chunk 中抽取。"""

    __tablename__ = "knowledge_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="concept", nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 来源 chunk（逗号分隔的 chunk 标识，用于溯源）
    source_chunks: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KnowledgeRelation(Base):
    """知识图谱关系（阶段 2C）— 实体间的有向边。"""

    __tablename__ = "knowledge_relations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    source_entity: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    target_entity: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GraphBuildRun(Base):
    """图谱构建运行记录（GraphRAG 阶段 5）— 从已入库 chunks 构建 Neo4j 图谱的任务状态。

    status: pending / running / completed / failed
    """

    __tablename__ = "graph_build_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    extractor: Mapped[str] = mapped_column(String(64), default="llm", nullable=False)

    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    entities_found: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    relations_found: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    entities_indexed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    relations_indexed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class GraphExtractionCache(Base):
    """逐 chunk 图谱抽取缓存；模型或 prompt 变化会生成新的 cache_key。"""

    __tablename__ = "graph_extraction_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    extractor: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class EvaluationRun(Base):
    """检索评估运行（阶段 2D）— 一次命名评估的聚合指标 + 逐条明细。

    metrics_json 结构:
        {"metrics_version": "local-v1", "k": 10,
         "hit_rate_at_k": 0.8, "mrr_at_k": 0.65,
         "recall_at_k": 0.8, "precision_at_k": 0.08,
         "ndcg_at_k": 0.71, "avg_score": 0.72,
         "details": [{"question":..., "expected_file_id":...,
                      "expected_chunk_id":..., "reference_answer":...,
                      "file_hit_rank":1|None, "chunk_hit_rank":1|None,
                      "top_score":...}]}
    """

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    knowledge_base_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True, index=True
    )
    top_k: Mapped[int] = mapped_column(Integer, default=4, nullable=False)
    query_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    hit_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mrr: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
