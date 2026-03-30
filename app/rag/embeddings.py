"""Embedding model abstraction.

Supports two backends controlled by ``Settings.EMBEDDING_TYPE``:

* ``local``            – SentenceTransformers (BGE-M3 by default, runs on CPU/GPU)
* ``openai_compatible`` – any OpenAI-compatible /v1/embeddings HTTP endpoint
"""
from __future__ import annotations

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
        logger.info("[LocalEmbedder] model loaded, dim=%d", cfg.EMBEDDING_DIMENSION)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
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
        else:
            raise EmbeddingError(f"Unknown EMBEDDING_TYPE: {cfg.EMBEDDING_TYPE}")
    return _embedder
