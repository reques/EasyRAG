"""Embedding model abstraction.

Supports backends controlled by ``Settings.EMBEDDING_TYPE``:

* ``local``              – SentenceTransformers (BGE-M3 by default, runs on CPU/GPU)
* ``openai_compatible``  – any OpenAI-compatible /v1/embeddings HTTP endpoint
* ``ollama``             – local Ollama server (e.g. bge-m3) via /api/embed
"""
from __future__ import annotations

import math
import re
import threading
from typing import List, Optional

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()


class BaseEmbedder:
    """Abstract base for embedding models."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class LocalEmbedder(BaseEmbedder):
    """SentenceTransformers-based local embedder (e.g. BGE-M3)."""

    def __init__(self, model_path: Optional[str] = None):
        path = model_path or cfg.EMBEDDING_MODEL_PATH
        logger.info("[LocalEmbedder] loading model from %s", path)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(path)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load local embedding model: {exc}") from exc
        # SentenceTransformer 实例非线程安全：并发 reindex/ingestion 时多线程
        # 同时 encode 会竞争，用锁串行化（CPU 推理串行反而更高效、更稳）。
        self._lock = threading.Lock()
        logger.info("[LocalEmbedder] model loaded, dim=%d", cfg.EMBEDDING_DIMENSION)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            with self._lock:
                vecs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [v.tolist() for v in vecs]
        except Exception as exc:
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Embedder that calls an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from openai import OpenAI
        self._client = OpenAI(
            base_url=base_url or cfg.EMBEDDING_API_BASE or cfg.LLM_BASE_URL,
            api_key=api_key or cfg.EMBEDDING_API_KEY or cfg.LLM_API_KEY,
        )
        self._model = model or cfg.EMBEDDING_MODEL_NAME
        logger.info("[OpenAICompatibleEmbedder] model=%s", self._model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self._client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            raise EmbeddingError(f"OpenAI-compatible embedding failed: {exc}") from exc


class _NonFiniteEmbeddingError(EmbeddingError):
    """A numeric inference failure for which a CPU retry is appropriate."""


_NONFINITE_RESPONSE = re.compile(
    r"unsupported value:\s*(?:nan|[+-]?inf(?:inity)?)\b|non[- ]finite", re.IGNORECASE
)


class OllamaEmbedder(BaseEmbedder):
    """Embedder that calls a local Ollama server's /api/embed endpoint.

    Works with any Ollama embedding model (bge-m3, nomic-embed-text, mxbai-embed-large...).
    Uses the modern batch API (POST /api/embed, {"model":..., "input":[...]}).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        import requests
        self._requests = requests
        self._base_url = (base_url or cfg.OLLAMA_BASE_URL).rstrip("/")
        self._model = model or cfg.OLLAMA_EMBED_MODEL
        self._timeout = timeout or cfg.OLLAMA_TIMEOUT
        logger.info("[OllamaEmbedder] base=%s model=%s", self._base_url, self._model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 分批请求：Ollama /api/embed 对单次批量大小有限制，
        # 大文件数百个 chunk 一次性 POST 会 400。每批 32 条是安全值。
        BATCH = 32
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            all_vectors.extend(self._embed_batch(batch))
        return all_vectors

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            try:
                return self._request_embeddings(texts)
            except _NonFiniteEmbeddingError:
                logger.warning(
                    "[OllamaEmbedder] non-finite inference result: model=%s batch_size=%d; retrying once on CPU",
                    self._model, len(texts),
                )
                try:
                    vectors = self._request_embeddings(texts, cpu=True)
                except Exception as exc:
                    raise EmbeddingError(
                        f"Ollama CPU retry failed for model '{self._model}': {exc}"
                    ) from exc
                logger.info("[OllamaEmbedder] CPU retry succeeded: model=%s batch_size=%d", self._model, len(texts))
                return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama embedding failed for model '{self._model}': {exc}"
            ) from exc

    def _request_embeddings(self, texts: List[str], *, cpu: bool = False) -> List[List[float]]:
        payload = {"model": self._model, "input": texts}
        if cpu:
            # These options apply only to this retry. Unload afterwards so the
            # next default request can load its normal GPU runner again.
            payload.update(options={"num_gpu": 0}, keep_alive=0)
        resp = self._requests.post(
            f"{self._base_url}/api/embed", json=payload, timeout=self._timeout,
        )
        if resp.status_code >= 400:
            try:
                error = resp.json().get("error", "")
            except (ValueError, AttributeError):
                error = ""
            if isinstance(error, str) and _NONFINITE_RESPONSE.search(error):
                raise _NonFiniteEmbeddingError("Ollama returned a non-finite inference error")
        resp.raise_for_status()
        vectors = resp.json().get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an invalid embedding count")
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("Ollama returned an empty or malformed embedding")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise EmbeddingError("Ollama returned inconsistent embedding dimensions")
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingError("Ollama returned a non-numeric embedding value")
                if not math.isfinite(value):
                    raise _NonFiniteEmbeddingError("Ollama returned a non-finite embedding value")
        return vectors


# ── Singleton factory ─────────────────────────────────────────────────────────

_embedder: Optional[BaseEmbedder] = None


def get_embedder() -> BaseEmbedder:
    """Return the process-level embedder singleton."""
    global _embedder
    if _embedder is None:
        if cfg.EMBEDDING_TYPE == "local":
            _embedder = LocalEmbedder()
        elif cfg.EMBEDDING_TYPE == "openai_compatible":
            _embedder = OpenAICompatibleEmbedder()
        elif cfg.EMBEDDING_TYPE == "ollama":
            _embedder = OllamaEmbedder()
        else:
            raise EmbeddingError(f"Unknown EMBEDDING_TYPE: {cfg.EMBEDDING_TYPE}")
    return _embedder
