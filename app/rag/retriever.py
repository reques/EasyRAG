"""Vector-store retriever with pluggable backends.

Backends
--------
* ``memory``  – in-process numpy store (default, no external deps)
* ``milvus``  – Milvus / Zilliz Cloud via pymilvus
* ``chroma``  – ChromaDB (persistent local store)

The active backend is selected by ``Settings.VECTOR_STORE_TYPE``.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logger import get_logger
from app.rag.embeddings import get_embedder

logger = get_logger(__name__)
cfg = get_settings()

DocList = List[Dict[str, Any]]  # [{"content": str, "metadata": dict}]


# ── Base ─────────────────────────────────────────────────────────────────────

class FileInfo(dict):
    """Dict subclass representing per-file statistics in the knowledge base.

    Keys: source (str), chunk_count (int), char_count (int)
    """


class BaseRetriever:
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 4) -> DocList:
        raise NotImplementedError

    def delete_collection(self) -> None:
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

    def retrieve(self, query: str, top_k: int = 4) -> DocList:
        if not self._vecs:
            return []
        import numpy as np
        embedder = get_embedder()
        q_vec = np.array(embedder.embed_query(query), dtype=float)
        mat = np.array(self._vecs, dtype=float)
        # cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        scores = (mat / norms) @ (q_vec / (np.linalg.norm(q_vec) + 1e-9))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: DocList = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < cfg.RAG_SCORE_THRESHOLD:
                continue
            results.append({
                "content": self._texts[idx],
                "metadata": {**self._metas[idx], "score": score},
            })
        logger.info("[MemoryRetriever] query returned %d docs", len(results))
        return results

    def delete_collection(self) -> None:
        self._texts.clear()
        self._metas.clear()
        self._vecs.clear()
        logger.info("[MemoryRetriever] collection cleared")

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
                "pymilvus is not installed. Run: pip install pymilvus"
            ) from exc

        connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
        logger.info("[MilvusRetriever] connected to %s:%s", cfg.MILVUS_HOST, cfg.MILVUS_PORT)

        col_name = cfg.MILVUS_COLLECTION
        if not utility.has_collection(col_name):
            fields = [
                FieldSchema(name="id",      dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source",  dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="vector",  dtype=DataType.FLOAT_VECTOR, dim=cfg.EMBEDDING_DIMENSION),
            ]
            schema = CollectionSchema(fields, description="RAG document store")
            Collection(name=col_name, schema=schema)
            logger.info("[MilvusRetriever] collection '%s' created", col_name)

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
        entities = [ids, texts, sources, normed]
        self._col.insert(entities)
        self._col.flush()
        logger.info("[MilvusRetriever] inserted %d docs", len(texts))
        return len(texts)

    def retrieve(self, query: str, top_k: int = 4) -> DocList:
        import numpy as np
        embedder = get_embedder()
        q = np.array(embedder.embed_query(query), dtype=float)
        q = q / (np.linalg.norm(q) + 1e-9)
        results = self._col.search(
            data=[q.tolist()],
            anns_field="vector",
            param={"metric_type": self._METRIC, "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["content", "source"],
        )
        docs: DocList = []
        for hit in results[0]:
            score = float(hit.score)
            if score < cfg.RAG_SCORE_THRESHOLD:
                continue
            docs.append({
                "content": hit.entity.get("content", ""),
                "metadata": {"source": hit.entity.get("source", ""), "score": score},
            })
        logger.info("[MilvusRetriever] query returned %d docs", len(docs))
        return docs

    def delete_collection(self) -> None:
        from pymilvus import utility
        utility.drop_collection(cfg.MILVUS_COLLECTION)
        logger.info("[MilvusRetriever] collection '%s' dropped", cfg.MILVUS_COLLECTION)

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

    def retrieve(self, query: str, top_k: int = 4) -> DocList:
        embedder = get_embedder()
        q_vec = embedder.embed_query(query)
        res = self._col.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs: DocList = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = 1.0 - float(dist)  # chroma returns L2 distance with cosine space
            if score < cfg.RAG_SCORE_THRESHOLD:
                continue
            docs.append({"content": text, "metadata": {**meta, "score": score}})
        logger.info("[ChromaRetriever] query returned %d docs", len(docs))
        return docs

    def delete_collection(self) -> None:
        import chromadb
        client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        client.delete_collection(cfg.CHROMA_COLLECTION)
        logger.info("[ChromaRetriever] collection '%s' deleted", cfg.CHROMA_COLLECTION)


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
