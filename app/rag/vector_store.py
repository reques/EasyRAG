"""Vector store abstraction.

Supports:
  memory  - in-process cosine similarity (default, no server needed)
  milvus  - Milvus server (production)
  chroma  - ChromaDB (lightweight persistent option)
"""
from __future__ import annotations
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()


class VectorStore:
    def add(self, texts, vectors, metadatas=None): raise NotImplementedError
    def search(self, query_vector, top_k=4): raise NotImplementedError
    def count(self): raise NotImplementedError


class MemoryVectorStore(VectorStore):
    """Pure-Python in-memory cosine similarity store."""
    def __init__(self):
        self._texts: List[str] = []
        self._vectors: List[List[float]] = []
        self._metadatas: List[Dict[str, Any]] = []

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    def add(self, texts, vectors, metadatas=None):
        metas = metadatas or [{} for _ in texts]
        self._texts.extend(texts)
        self._vectors.extend(vectors)
        self._metadatas.extend(metas)
        logger.debug("[MemoryVectorStore] total=%d", len(self._texts))

    def search(self, query_vector, top_k=4):
        if not self._vectors:
            return []
        scores = [self._cosine(query_vector, v) for v in self._vectors]
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._texts[i], scores[i], self._metadatas[i]) for i in idx]

    def count(self):
        return len(self._texts)


class MilvusVectorStore(VectorStore):
    """Milvus-backed vector store with local persistence."""
    
    def __init__(self):
        try:
            import pymilvus  # noqa
        except ImportError:
            raise VectorStoreError("pymilvus not installed. Run: pip install pymilvus")
        
        from pymilvus import connections
        
        self._name = cfg.MILVUS_COLLECTION
        self._dim = cfg.EMBEDDING_DIMENSION
        self._persist_dir = Path(cfg.MILVUS_DATA_DIR)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._persist_dir / "metadata.jsonl"
        
        # Connect to Milvus
        try:
            connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
            logger.info("[MilvusVectorStore] connected to %s:%d", cfg.MILVUS_HOST, cfg.MILVUS_PORT)
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Milvus: {e}")
        
        self._col = self._ensure_collection()
        self._load_metadata_cache()
        logger.info("[MilvusVectorStore] initialized col=%s, persist_dir=%s", self._name, self._persist_dir)

    def _ensure_collection(self):
        """Create or load existing collection."""
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
        
        if utility.has_collection(self._name):
            logger.info("[MilvusVectorStore] loading existing collection: %s", self._name)
            col = Collection(self._name)
            col.load()
            return col
        
        logger.info("[MilvusVectorStore] creating new collection: %s", self._name)
        fields = [
            FieldSchema("id",       DataType.INT64,        is_primary=True, auto_id=True),
            FieldSchema("text",     DataType.VARCHAR,      max_length=8192),
            FieldSchema("source",   DataType.VARCHAR,      max_length=512),
            FieldSchema("chunk_id", DataType.VARCHAR,      max_length=256),
            FieldSchema("vector",   DataType.FLOAT_VECTOR, dim=self._dim),
        ]
        schema = CollectionSchema(fields, description="RAG document vectors")
        col = Collection(self._name, schema)
        
        # Create index for faster search
        col.create_index(
            "vector",
            {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        col.load()
        return col

    def _load_metadata_cache(self):
        """Load metadata cache from local file."""
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            chunk_id = data.get("chunk_id")
                            if chunk_id:
                                self._metadata_cache[chunk_id] = data
                logger.info("[MilvusVectorStore] loaded %d metadata entries from cache", 
                           len(self._metadata_cache))
            except Exception as e:
                logger.warning("[MilvusVectorStore] failed to load metadata cache: %s", e)

    def _save_metadata(self, chunk_id: str, metadata: Dict[str, Any]):
        """Save metadata to local file."""
        try:
            metadata["chunk_id"] = chunk_id
            self._metadata_cache[chunk_id] = metadata
            
            with open(self._metadata_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error("[MilvusVectorStore] failed to save metadata: %s", e)

    def add(self, texts: List[str], vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add texts and vectors to Milvus."""
        if not texts:
            return
        
        metas = metadatas or [{} for _ in texts]
        chunk_ids = []
        sources = []
        
        # Generate chunk IDs and prepare data
        for text, meta in zip(texts, metas):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            sources.append(meta.get("source", ""))
            
            # Save metadata locally
            self._save_metadata(chunk_id, meta)
        
        try:
            # Insert into Milvus
            self._col.insert([texts, sources, chunk_ids, vectors])
            self._col.flush()
            logger.info("[MilvusVectorStore] added %d documents", len(texts))
        except Exception as e:
            logger.error("[MilvusVectorStore] failed to insert data: %s", e)
            raise VectorStoreError(f"Failed to insert into Milvus: {e}")

    def search(self, query_vector: List[float], top_k: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        try:
            hits = self._col.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                output_fields=["text", "source", "chunk_id"],
            )
            
            results = []
            for h in hits[0]:
                text = h.entity.get("text", "")
                score = float(h.score)
                chunk_id = h.entity.get("chunk_id", "")
                source = h.entity.get("source", "")
                
                # Retrieve full metadata from cache
                metadata = self._metadata_cache.get(chunk_id, {"source": source})
                results.append((text, score, metadata))
            
            return results
        except Exception as e:
            logger.error("[MilvusVectorStore] search failed: %s", e)
            return []

    def count(self) -> int:
        """Get total number of documents."""
        try:
            return self._col.num_entities
        except Exception as e:
            logger.error("[MilvusVectorStore] count failed: %s", e)
            return 0


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store."""
    def __init__(self):
        try:
            import chromadb  # noqa
        except ImportError:
            raise VectorStoreError("chromadb not installed. Run: pip install chromadb")
        import chromadb
        self._client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        self._col = self._client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        logger.info("[ChromaVectorStore] col=%s", cfg.CHROMA_COLLECTION)

    def add(self, texts, vectors, metadatas=None):
        metas = metadatas or [{} for _ in texts]
        self._col.add(ids=[str(uuid.uuid4()) for _ in texts],
                      documents=texts, embeddings=vectors, metadatas=metas)

    def search(self, query_vector, top_k=4):
        res = self._col.query(query_embeddings=[query_vector], n_results=top_k,
                              include=["documents", "distances", "metadatas"])
        docs  = res.get("documents",  [[]])[0]
        dists = res.get("distances",  [[]])[0]
        metas = res.get("metadatas",  [[]])[0]
        return [(d, 1.0 - dist, m or {}) for d, dist, m in zip(docs, dists, metas)]

    def count(self):
        return self._col.count()


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        t = cfg.VECTOR_STORE_TYPE
        logger.info("[vector_store] init backend=%s", t)
        if t == "milvus":
            _store = MilvusVectorStore()
        elif t == "chroma":
            _store = ChromaVectorStore()
        else:
            _store = MemoryVectorStore()
    return _store
