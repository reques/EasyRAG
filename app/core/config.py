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

    # ── PostgreSQL (阶段 1) ──────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "easyrag"
    POSTGRES_PASSWORD: str = "easyrag_secret"
    POSTGRES_DB: str = "easyrag"
    POSTGRES_POOL_SIZE: int = 10
    POSTGRES_MAX_OVERFLOW: int = 5

    # ── Redis (阶段 1) ──────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # ── MinIO (阶段 1) ──────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9091"
    MINIO_ACCESS_KEY: str = "easyrag_admin"
    MINIO_SECRET_KEY: str = "easyrag_minio_secret"
    MINIO_BUCKET: str = "easyrag-files"
    MINIO_SECURE: bool = False

    # ── JWT (阶段 1) ────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-this-to-a-random-secret-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
