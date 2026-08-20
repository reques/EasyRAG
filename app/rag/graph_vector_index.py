"""Milvus 图谱语义索引（GraphRAG 阶段 5）。

为知识库中**唯一实体 / 唯一三元组**建立语义向量索引（独立 collection，
与 chunk collection 分离），检索时召回图谱元素再经 Neo4j 展开映射回
chunk 引用。

条目键（id 字段）：
- 实体:  ``e:{kb_id}:{name}``
- 三元组: ``t:{kb_id}:{source}|{relation}|{target}``

collection schema: id(VARCHAR pk), kind(VARCHAR), kb_id(VARCHAR),
text(VARCHAR), vector(FLOAT_VECTOR, COSINE)。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()


def entity_key(kb_id: str, name: str) -> str:
    return f"e:{kb_id}:{name}"


def triple_key(kb_id: str, source: str, relation: str, target: str) -> str:
    return f"t:{kb_id}:{source}|{relation}|{target}"


class GraphVectorIndex:
    """唯一实体/三元组语义索引（pymilvus 直连，懒加载）。"""

    def __init__(self, collection_name: str | None = None):
        self._name = collection_name or cfg.GRAPH_ENTITY_COLLECTION
        self._col = None
        self._dim: Optional[int] = None
        self._lock = threading.Lock()

    # ── 连接与 schema ────────────────────────────────────────────────────

    def _ensure_collection(self):
        if self._col is not None:
            return self._col
        with self._lock:
            if self._col is not None:
                return self._col
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )

            connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
            if not utility.has_collection(self._name):
                fields = [
                    FieldSchema("id", DataType.VARCHAR, max_length=512, is_primary=True),
                    FieldSchema("kind", DataType.VARCHAR, max_length=16),   # entity / triple
                    FieldSchema("kb_id", DataType.VARCHAR, max_length=64),
                    FieldSchema("text", DataType.VARCHAR, max_length=2048),
                    FieldSchema("source", DataType.VARCHAR, max_length=512),
                    FieldSchema("target", DataType.VARCHAR, max_length=512),
                    FieldSchema("relation", DataType.VARCHAR, max_length=256),
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=self._dimension()),
                ]
                schema = CollectionSchema(fields, description="Graph entity/triple semantic index")
                col = Collection(self._name, schema)
                col.create_index(
                    "vector",
                    {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
                )
                logger.info("[graph_vector_index] collection '%s' created (dim=%d)",
                            self._name, self._dimension())
            else:
                col = Collection(self._name)
                existing_names = {f.name for f in col.schema.fields}
                if not {"source", "target", "relation"}.issubset(existing_names):
                    # 旧 schema 无结构化字段 → 重建（图谱索引可整体重建）
                    from pymilvus import utility as _utility
                    _utility.drop_collection(self._name)
                    col = self._ensure_collection()
                    return col
            col.load()
            self._col = col
            return col

    def _dimension(self) -> int:
        """从现有 embedder 探测向量维度（懒加载一次）。"""
        if self._dim is None:
            from app.rag.embeddings import get_embedder

            probe = get_embedder().embed_query("维度探测")
            self._dim = len(probe)
        return self._dim

    # ── 写入 ─────────────────────────────────────────────────────────────

    def upsert(self, items: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
        """批量 upsert。items 每条含 id/kind/kb_id/text/source/target/relation。"""
        if not items:
            return
        col = self._ensure_collection()
        col.upsert([
            [it["id"] for it in items],
            [it["kind"] for it in items],
            [it["kb_id"] for it in items],
            [it["text"] for it in items],
            [it.get("source", "") for it in items],
            [it.get("target", "") for it in items],
            [it.get("relation", "") for it in items],
            vectors,
        ])
        col.flush()

    # ── 检索 ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        kb_ids: List[str],
        top_k: int = 5,
        kinds: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """语义检索实体/三元组，返回 [{id, kind, kb_id, text, score}]。"""
        if not kb_ids:
            return []
        try:
            col = self._ensure_collection()
        except Exception as exc:
            logger.warning("[graph_vector_index] search skipped: %s", exc)
            return []
        expr = "kb_id in [" + ", ".join(f'"{k}"' for k in kb_ids) + "]"
        if kinds:
            expr += " and kind in [" + ", ".join(f'"{k}"' for k in kinds) + "]"
        try:
            hits = col.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=expr,
                output_fields=["kind", "kb_id", "text", "source", "target", "relation"],
            )
        except Exception as exc:
            logger.warning("[graph_vector_index] search failed: %s", exc)
            return []
        out = []
        for h in hits[0]:
            out.append({
                "id": h.id,
                "kind": h.entity.get("kind", ""),
                "kb_id": h.entity.get("kb_id", ""),
                "text": h.entity.get("text", ""),
                "source": h.entity.get("source", ""),
                "target": h.entity.get("target", ""),
                "relation": h.entity.get("relation", ""),
                "score": float(h.score),
            })
        return out

    # ── 统计 / 清理 ──────────────────────────────────────────────────────

    def count(self, kb_id: str) -> int:
        try:
            col = self._ensure_collection()
            if not kb_id:
                return col.num_entities
            res = col.query(expr=f'kb_id == "{kb_id}"', output_fields=["count(*)"])
            return int(res[0]["count(*)"]) if res else 0
        except Exception as exc:
            logger.warning("[graph_vector_index] count failed: %s", exc)
            return 0

    def delete_by_kb(self, kb_id: str) -> None:
        """删除某知识库的全部图谱索引条目（重置用）。"""
        try:
            col = self._ensure_collection()
            col.delete(expr=f'kb_id == "{kb_id}"')
            col.flush()
        except Exception as exc:
            logger.warning("[graph_vector_index] delete_by_kb failed: %s", exc)


_graph_vector_index: Optional[GraphVectorIndex] = None
_index_lock = threading.Lock()


def get_graph_vector_index() -> GraphVectorIndex:
    global _graph_vector_index
    if _graph_vector_index is None:
        with _index_lock:
            if _graph_vector_index is None:
                _graph_vector_index = GraphVectorIndex()
    return _graph_vector_index
