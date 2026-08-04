"""Embedding model abstraction.

Supports backends controlled by ``Settings.EMBEDDING_TYPE``:

* ``local``              – SentenceTransformers (BGE-M3 by default, runs on CPU/GPU)
* ``openai_compatible``  – any OpenAI-compatible /v1/embeddings HTTP endpoint
* ``ollama``             – local Ollama server (e.g. bge-m3) via /api/embed
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
            resp = self._requests.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            vectors = data.get("embeddings")
            if not vectors or len(vectors) != len(texts):
                raise EmbeddingError(
                    f"Ollama returned {len(vectors or [])} vectors for {len(texts)} inputs"
                )
            return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama embedding failed (is `ollama serve` running and model "
                f"'{self._model}' pulled?): {exc}"
            ) from exc


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
