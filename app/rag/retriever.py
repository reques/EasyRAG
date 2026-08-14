"""Vector-store retriever with pluggable backends.

Backends
--------
* ``memory``  – in-process numpy store (default, no external deps)
* ``milvus``  – Milvus / Zilliz Cloud via pymilvus
* ``chroma``  – ChromaDB (persistent local store)

The active backend is selected by ``Settings.VECTOR_STORE_TYPE``.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logger import get_logger
from app.rag.embeddings import get_embedder

logger = get_logger(__name__)
cfg = get_settings()

DocList = List[Dict[str, Any]]  # [{"content": str, "metadata": dict}]


def get_document_chunk_id(
    knowledge_base_id: uuid.UUID | str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a stable public ID for a retrieved chunk.

    Older Milvus collections only persist source and content, so the ID must
    remain reproducible even when richer chunk metadata is unavailable.
    """
    meta = metadata or {}
    explicit_id = meta.get("chunk_id")
    if explicit_id:
        return str(explicit_id)
    payload = "\x1f".join((
        str(uuid.UUID(str(knowledge_base_id))),
        str(meta.get("source") or ""),
        str(meta.get("chunk_index") if meta.get("chunk_index") is not None else ""),
        str(content or ""),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_knowledge_base_ids(
    knowledge_base_ids: Optional[Sequence[str]],
) -> List[str]:
    """Validate and canonicalise an authorised knowledge-base scope.

    Retrieval is fail-closed: callers that do not provide at least one valid
    knowledge-base id receive no documents.  UUID validation also prevents a
    caller-controlled value from being interpolated into a Milvus expression.
    """
    if not knowledge_base_ids:
        return []
    if isinstance(knowledge_base_ids, str):
        knowledge_base_ids = [knowledge_base_ids]

    normalised: List[str] = []
    seen = set()
    for value in knowledge_base_ids:
        canonical = str(uuid.UUID(str(value)))
        if canonical not in seen:
            seen.add(canonical)
            normalised.append(canonical)
    return normalised


def _unwrap_parent(docs: DocList) -> DocList:
    """parent_child 策略：检索命中的 child 块替换为 parent 上下文块返回。

    命中的 metadata 含 parent_text 时，用它替换 content（parent 才是完整上下文），
    相同 parent 去重，避免一个父块被多个 child 命中后重复占上下文窗口。
    """
    if not any(d["metadata"].get("parent_text") for d in docs):
        return docs
    seen: set = set()
    out: DocList = []
    for d in docs:
        parent = d["metadata"].get("parent_text")
        if not parent:
            out.append(d)
            continue
        key = hash(parent)
        if key in seen:
            continue
        seen.add(key)
        out.append({"content": parent, "metadata": d["metadata"]})
    return out


# ── Base ─────────────────────────────────────────────────────────────────────

class FileInfo(dict):
    """Dict subclass representing per-file statistics in the knowledge base.

    Keys: source (str), chunk_count (int), char_count (int)
    """


class BaseRetriever:
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> DocList:
        raise NotImplementedError

    def delete_collection(self) -> None:
        raise NotImplementedError

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all chunks belonging to a source file. Returns count deleted."""
        raise NotImplementedError

    def list_documents(self) -> List[FileInfo]:
        """Return per-file statistics: [{source, chunk_count, char_count}]."""
        raise NotImplementedError


# ── In-memory backend ─────────────────────────────────────────────────────────

class MemoryRetriever(BaseRetriever):
    """Simple numpy cosine-similarity store – no external dependencies."""

    def __init__(self):
        self._texts: List[str] = []
        self._metas: List[Dict] = []
        self._vecs: List[List[float]] = []
        logger.info("[MemoryRetriever] initialised")

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        if not texts:
            return 0
        metas = metadatas or [{} for _ in texts]
        embedder = get_embedder()
        vecs = embedder.embed_texts(texts)
        self._texts.extend(texts)
        self._metas.extend(metas)
        self._vecs.extend(vecs)
        logger.info("[MemoryRetriever] added %d docs, total=%d", len(texts), len(self._texts))
        return len(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> DocList:
        allowed_ids = set(normalize_knowledge_base_ids(knowledge_base_ids))
        if not self._vecs or not allowed_ids:
            return []
        import numpy as np
        embedder = get_embedder()
        q_vec = np.array(embedder.embed_query(query), dtype=float)
        allowed_indices = [
            idx for idx, meta in enumerate(self._metas)
            if str(meta.get("knowledge_base_id", "")) in allowed_ids
        ]
        if not allowed_indices:
            return []
        mat = np.array([self._vecs[idx] for idx in allowed_indices], dtype=float)
        # cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        scores = (mat / norms) @ (q_vec / (np.linalg.norm(q_vec) + 1e-9))
        top_idx = np.argsort(scores)[::-1][:top_k]
        threshold = (
            cfg.RAG_SCORE_THRESHOLD
            if score_threshold is None
            else score_threshold
        )
        results: DocList = []
        for local_idx in top_idx:
            idx = allowed_indices[int(local_idx)]
            score = float(scores[local_idx])
            if score < threshold:
                continue
            results.append({
                "content": self._texts[idx],
                "metadata": {**self._metas[idx], "score": score},
            })
        logger.info("[MemoryRetriever] query returned %d docs", len(results))
        return _unwrap_parent(results)

    def delete_collection(self) -> None:
        self._texts.clear()
        self._metas.clear()
        self._vecs.clear()
        logger.info("[MemoryRetriever] collection cleared")

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all chunks with metadata.source == source."""
        keep = [(t, m, v) for t, m, v in zip(self._texts, self._metas, self._vecs)
                if m.get("source") != source]
        deleted = len(self._texts) - len(keep)
        self._texts = [k[0] for k in keep]
        self._metas = [k[1] for k in keep]
        self._vecs = [k[2] for k in keep]
        if deleted:
            logger.info("[MemoryRetriever] deleted %d docs for source '%s'", deleted, source)
        return deleted

    def list_documents(self) -> List["FileInfo"]:
        from collections import defaultdict
        stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"chunk_count": 0, "char_count": 0})
        for text, meta in zip(self._texts, self._metas):
            src = meta.get("source", "(unknown)")
            stats[src]["chunk_count"] += 1
            stats[src]["char_count"] += len(text)
        return [
            FileInfo(source=src, chunk_count=v["chunk_count"], char_count=v["char_count"])
            for src, v in stats.items()
        ]


# ── Milvus backend ────────────────────────────────────────────────────────────

class MilvusRetriever(BaseRetriever):
    """Milvus / Zilliz Cloud retriever via pymilvus."""

    _METRIC = "IP"  # inner-product (use normalised vectors → cosine)

    def __init__(self):
        try:
            from pymilvus import (
                connections, Collection, CollectionSchema,
                FieldSchema, DataType, utility,
            )
        except ImportError as exc:
            raise VectorStoreError(
                f"pymilvus import failed (installed but API changed, or wrong "
                f"Python env): {exc}"
            ) from exc

        connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
        logger.info("[MilvusRetriever] connected to %s:%s", cfg.MILVUS_HOST, cfg.MILVUS_PORT)

        col_name = cfg.MILVUS_COLLECTION
        desired_fields = {"id", "content", "source", "knowledge_base_id", "vector"}
        if utility.has_collection(col_name):
            existing = Collection(col_name)
            existing_names = {f.name for f in existing.schema.fields}
            if not desired_fields.issubset(existing_names):
                # schema 不含 knowledge_base_id（旧版 4 字段）→ 重建 collection。
                # 旧向量无 kb_id 无法回填,由调用方从 PostgreSQL text_content 重建索引。
                logger.warning(
                    "[MilvusRetriever] collection '%s' schema outdated (%s), dropping to rebuild with knowledge_base_id",
                    col_name, sorted(existing_names),
                )
                utility.drop_collection(col_name)
        if not utility.has_collection(col_name):
            fields = [
                FieldSchema(name="id",                dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="content",           dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source",            dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="knowledge_base_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="vector",            dtype=DataType.FLOAT_VECTOR, dim=cfg.EMBEDDING_DIMENSION),
            ]
            schema = CollectionSchema(fields, description="RAG document store")
            Collection(name=col_name, schema=schema)
            logger.info("[MilvusRetriever] collection '%s' created (with knowledge_base_id)", col_name)

        self._col = Collection(col_name)
        # Ensure index exists
        if not self._col.indexes:
            self._col.create_index(
                field_name="vector",
                index_params={"metric_type": self._METRIC, "index_type": "IVF_FLAT", "params": {"nlist": 128}},
            )
        self._col.load()
        logger.info("[MilvusRetriever] collection '%s' loaded", col_name)

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        if not texts:
            return 0
        metas = metadatas or [{} for _ in texts]
        embedder = get_embedder()
        vecs = embedder.embed_texts(texts)

        import numpy as np
        # Normalise for cosine via IP
        arr = np.array(vecs, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = (arr / norms).tolist()

        ids = [str(uuid.uuid4())[:63] for _ in texts]
        sources = [m.get("source", "") for m in metas]
        kb_ids = [str(m.get("knowledge_base_id", "") or "") for m in metas]
        entities = [ids, texts, sources, kb_ids, normed]
        self._col.insert(entities)
        self._col.flush()
        logger.info("[MilvusRetriever] inserted %d docs", len(texts))
        return len(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> DocList:
        allowed_ids = normalize_knowledge_base_ids(knowledge_base_ids)
        if not allowed_ids:
            return []
        import numpy as np
        embedder = get_embedder()
        q = np.array(embedder.embed_query(query), dtype=float)
        q = q / (np.linalg.norm(q) + 1e-9)
        scope_expr = "knowledge_base_id in [" + ", ".join(
            f'"{kb_id}"' for kb_id in allowed_ids
        ) + "]"
        results = self._col.search(
            data=[q.tolist()],
            anns_field="vector",
            param={"metric_type": self._METRIC, "params": {"nprobe": 16}},
            limit=top_k,
            expr=scope_expr,
            output_fields=["content", "source", "knowledge_base_id"],
        )
        threshold = (
            cfg.RAG_SCORE_THRESHOLD
            if score_threshold is None
            else score_threshold
        )
        docs: DocList = []
        for hit in results[0]:
            hit_kb_id = hit.entity.get("knowledge_base_id", "") or ""
            if hit_kb_id not in allowed_ids:
                logger.warning("[MilvusRetriever] discarded out-of-scope hit")
                continue
            score = float(hit.score)
            if score < threshold:
                continue
            docs.append({
                "content": hit.entity.get("content", ""),
                "metadata": {
                    "source": hit.entity.get("source", ""),
                    "knowledge_base_id": hit_kb_id,
                    "score": score,
                },
            })
        logger.info("[MilvusRetriever] query returned %d docs", len(docs))
        return _unwrap_parent(docs)

    def delete_collection(self) -> None:
        from pymilvus import utility
        utility.drop_collection(cfg.MILVUS_COLLECTION)
        logger.info("[MilvusRetriever] collection '%s' dropped", cfg.MILVUS_COLLECTION)

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all chunks with metadata.source == source."""
        if not self._col:
            return 0
        # Escape single quotes in source for Milvus expr
        safe = source.replace("'", "''")
        expr = f"source == '{safe}'"
        # Count first
        res = self._col.query(expr=expr, output_fields=["id"])
        count = len(res)
        if count == 0:
            return 0
        self._col.delete(expr)
        self._col.flush()
        logger.info("[MilvusRetriever] deleted %d docs for source '%s'", count, source)
        return count

    def list_documents(self) -> List["FileInfo"]:
        """Return per-file statistics by scanning all stored entities."""
        from collections import defaultdict
        try:
            total = self._col.num_entities
            if total == 0:
                return []
            # Query in pages to avoid fetching too many at once
            stats: dict = defaultdict(lambda: {"chunk_count": 0, "char_count": 0})
            page_size = 1000
            offset = 0
            while offset < total:
                res = self._col.query(
                    expr="id != ''",
                    output_fields=["content", "source"],
                    offset=offset,
                    limit=page_size,
                )
                for row in res:
                    src = row.get("source") or "(unknown)"
                    text = row.get("content") or ""
                    stats[src]["chunk_count"] += 1
                    stats[src]["char_count"] += len(text)
                offset += page_size
                if len(res) < page_size:
                    break
            return [
                FileInfo(source=src, chunk_count=v["chunk_count"], char_count=v["char_count"])
                for src, v in stats.items()
            ]
        except Exception as exc:
            logger.error("[MilvusRetriever] list_documents failed: %s", exc)
            return []


# ── Chroma backend ────────────────────────────────────────────────────────────

class ChromaRetriever(BaseRetriever):
    """ChromaDB persistent retriever."""

    def __init__(self):
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc
        client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        self._col = client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("[ChromaRetriever] collection '%s' ready", cfg.CHROMA_COLLECTION)

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        if not texts:
            return 0
        metas = metadatas or [{} for _ in texts]
        embedder = get_embedder()
        vecs = embedder.embed_texts(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        self._col.add(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)
        logger.info("[ChromaRetriever] added %d docs", len(texts))
        return len(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> DocList:
        allowed_ids = normalize_knowledge_base_ids(knowledge_base_ids)
        if not allowed_ids:
            return []
        embedder = get_embedder()
        q_vec = embedder.embed_query(query)
        where = (
            {"knowledge_base_id": allowed_ids[0]}
            if len(allowed_ids) == 1
            else {"knowledge_base_id": {"$in": allowed_ids}}
        )
        res = self._col.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        threshold = (
            cfg.RAG_SCORE_THRESHOLD
            if score_threshold is None
            else score_threshold
        )
        docs: DocList = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            if str(meta.get("knowledge_base_id", "")) not in allowed_ids:
                logger.warning("[ChromaRetriever] discarded out-of-scope hit")
                continue
            score = 1.0 - float(dist)  # chroma returns L2 distance with cosine space
            if score < threshold:
                continue
            docs.append({"content": text, "metadata": {**meta, "score": score}})
        logger.info("[ChromaRetriever] query returned %d docs", len(docs))
        return _unwrap_parent(docs)

    def delete_collection(self) -> None:
        import chromadb
        client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        client.delete_collection(cfg.CHROMA_COLLECTION)
        logger.info("[ChromaRetriever] collection '%s' deleted", cfg.CHROMA_COLLECTION)

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all chunks with metadata.source == source."""
        res = self._col.get(where={"source": source})
        ids = res.get("ids", [])
        if not ids:
            return 0
        self._col.delete(ids=ids)
        logger.info("[ChromaRetriever] deleted %d docs for source '%s'", len(ids), source)
        return len(ids)


# ── Singleton factory ─────────────────────────────────────────────────────────

_retriever: Optional[BaseRetriever] = None


def get_retriever() -> BaseRetriever:
    """Return the process-level retriever singleton based on config."""
    global _retriever
    if _retriever is None:
        vtype = cfg.VECTOR_STORE_TYPE
        logger.info("[retriever] building backend: %s", vtype)
        if vtype == "memory":
            _retriever = MemoryRetriever()
        elif vtype == "milvus":
            _retriever = MilvusRetriever()
        elif vtype == "chroma":
            _retriever = ChromaRetriever()
        else:
            raise VectorStoreError(f"Unknown VECTOR_STORE_TYPE: {vtype}")
    return _retriever
