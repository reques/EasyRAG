"""图谱召回器（GraphRAG 阶段 5）。

检索流程：
1. query 向量化 → Milvus 图谱语义索引（graph_entity_index）召回实体/三元组；
2. 实体命中 → Neo4j 1 跳子图展开，收集关系上的 chunk 引用；
3. 三元组命中 → Neo4j 按 (source, relation, target) 精确取 chunk 引用；
4. 按 chunk 被命中的加权次数聚合排序，输出候选 chunk_id 列表 + 图谱上下文。

所有 Neo4j/Milvus 异常降级为空结果（检索主链路不受影响）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()


class GraphRetriever:
    """图谱召回：实体/三元组 → chunk 引用映射。"""

    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k or cfg.GRAPH_ENTITY_TOP_K

    def retrieve(
        self,
        query: str,
        knowledge_base_ids: Sequence[str],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """图谱召回，返回 {"chunk_ids": [...], "entities": [...], "triples": [...]}。"""
        kb_ids = [str(i) for i in (knowledge_base_ids or [])]
        if not kb_ids:
            return {"chunk_ids": [], "entities": [], "triples": []}
        k = top_k or self.top_k

        # 1) query 向量化
        try:
            from app.rag.embeddings import get_embedder

            query_vector = get_embedder().embed_query(query)
        except Exception as exc:
            logger.warning("[graph_retriever] embed failed: %s", exc)
            return {"chunk_ids": [], "entities": [], "triples": []}

        # 2) Milvus 图谱索引召回
        from app.rag.graph_vector_index import get_graph_vector_index

        try:
            hits = get_graph_vector_index().search(query_vector, kb_ids, top_k=k)
        except Exception as exc:
            logger.warning("[graph_retriever] index search failed: %s", exc)
            return {"chunk_ids": [], "entities": [], "triples": []}
        if not hits:
            return {"chunk_ids": [], "entities": [], "triples": []}

        # 3) Neo4j 展开取 chunk 引用
        from backend.storage.neo4j.client import Neo4jUnavailableError, get_neo4j_client

        try:
            neo4j = get_neo4j_client()
            if not neo4j.available:
                return {"chunk_ids": [], "entities": [], "triples": []}
        except Neo4jUnavailableError as exc:
            logger.warning("[graph_retriever] neo4j unavailable: %s", exc)
            return {"chunk_ids": [], "entities": [], "triples": []}

        chunk_counter: Dict[str, float] = defaultdict(float)
        entities: List[Dict[str, Any]] = []
        triples: List[Dict[str, Any]] = []

        for hit in hits:
            hit_kb = hit.get("kb_id", "")
            score = hit.get("score", 0.0)
            if hit["kind"] == "entity":
                name = hit.get("text", "")
                entities.append({"name": name, "score": round(score, 4)})
                try:
                    # 实体 1 跳子图 → 收集所有关系上的 chunk 引用
                    subgraph = neo4j.get_subgraph(hit_kb, name, depth=1, max_nodes=40)
                    for edge in subgraph.get("edges", []):
                        for cid in edge.get("chunk_ids", []):
                            chunk_counter[cid] += score
                except Neo4jUnavailableError:
                    continue
            else:  # triple
                triples.append({
                    "source": hit.get("source", ""),
                    "target": hit.get("target", ""),
                    "relation": hit.get("relation", ""),
                    "score": round(score, 4),
                })
                try:
                    refs = neo4j.get_relation_chunk_refs(
                        hit_kb,
                        hit.get("source", ""),
                        hit.get("target", ""),
                        hit.get("relation", ""),
                    )
                    for cid in refs:
                        chunk_counter[cid] += score
                except Neo4jUnavailableError:
                    continue

        # 4) 聚合排序
        ranked = sorted(chunk_counter.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "chunk_ids": [cid for cid, _ in ranked],
            "entities": entities,
            "triples": triples,
        }


_graph_retriever: Optional[GraphRetriever] = None


def get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever
