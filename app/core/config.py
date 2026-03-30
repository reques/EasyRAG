"""Centralised configuration via pydantic-settings.

All values can be overridden by environment variables or a .env file.
Call `get_settings()` everywhere you need config (returns a cached singleton).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "All-in-RAG Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Server ───────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "*"

    # ── API ──────────────────────────────────────────────────────────────
    API_PREFIX: str = "/api/v1"

    # ── LLM ──────────────────────────────────────────────────────────────
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = Field(default="sk-placeholder", alias="DEEPSEEK_API_KEY")
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60          # seconds
    LLM_MAX_RETRIES: int = 2

    # ── Embedding ────────────────────────────────────────────────────────
    # EMBEDDING_TYPE: "local" uses SentenceTransformers; "openai_compatible" uses HTTP API
    EMBEDDING_TYPE: Literal["local", "openai_compatible"] = "local"
    EMBEDDING_MODEL_PATH: str = "./models/bge-m3"
    EMBEDDING_API_BASE: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_NAME: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024

    # ── Vector store ─────────────────────────────────────────────────────
    VECTOR_STORE_TYPE: Literal["memory", "milvus", "chroma"] = "milvus"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "rag_docs"
    MILVUS_DATA_DIR: str = "./milvus_data"   # local dir for persisted metadata
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "rag_docs"

    # ── RAG ──────────────────────────────────────────────────────────────
    RETRIEVER_TOP_K: int = 4
    RAG_SCORE_THRESHOLD: float = 0.0
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Agent / LangGraph ────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 20   # LangGraph recursion_limit
    MAX_PLAN_STEPS: int = 5          # max sub-tasks per plan
    SESSION_TTL: int = 3600          # seconds to keep session state

    # ── Answer quality ───────────────────────────────────────────────────
    ANSWER_VALIDATION_ENABLED: bool = True
    ANSWER_MIN_LENGTH: int = 20      # chars below which answer is "too short"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
