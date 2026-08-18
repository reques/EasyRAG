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
    APP_VERSION: str = "0.3.1"
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
    LLM_API_KEY: str = Field(default="", alias="DEEPSEEK_API_KEY")
    LLM_MODEL: str = "deepseek-v4-flash"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 8192
    LLM_TIMEOUT: int = 60          # seconds
    LLM_MAX_RETRIES: int = 2

    # ── Chat model catalog ──────────────────────────────────────────────────
    # The browser only receives the public IDs below. Provider endpoints,
    # concrete model names and API keys always remain server-side.
    LLM_DEFAULT_MODEL_ID: str = "deepseek-v4-flash"

    MINIMAX_BASE_URL: str = "https://api.minimaxi.com/v1"
    MINIMAX_API_KEY: str = ""
    MINIMAX_MODEL: str = "MiniMax-M2.7"
    MINIMAX_TEMPERATURE: float = 0.1

    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_TEMPERATURE: float = 0.0

    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_API_KEY: str = ""
    QWEN_MODEL: str = "qwen3.6-flash"
    QWEN_TEMPERATURE: float = 0.0

    GLM_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"
    ZHIPUAI_API_KEY: str = ""
    GLM_MODEL: str = "glm-5.2"
    # GLM's OpenAI-compatible endpoint requires temperature > 0.
    GLM_TEMPERATURE: float = 0.6

    # ── LLM 快速模型（分级接口，阶段 1）─────────────────────────────────────
    # 辅助任务（标题生成/意图识别/记忆提取等）可用更快/更便宜的模型。
    # 未配置时 fast tier 回退到主模型，不影响现有行为。
    LLM_FAST_BASE_URL: Optional[str] = None   # 默认回退 LLM_BASE_URL
    LLM_FAST_API_KEY: Optional[str] = None    # 默认回退 LLM_API_KEY
    LLM_FAST_MODEL: Optional[str] = None      # 默认回退 LLM_MODEL

    # ── Embedding ────────────────────────────────────────────────────────
    # EMBEDDING_TYPE: "local"=SentenceTransformers | "openai_compatible"=HTTP API | "ollama"=本地Ollama
    EMBEDDING_TYPE: Literal["local", "openai_compatible", "ollama"] = "local"
    EMBEDDING_MODEL_PATH: str = "./models/bge-m3"
    EMBEDDING_API_BASE: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_MODEL_NAME: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024

    # ── Ollama (本地嵌入) ─────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "bge-m3:latest"
    OLLAMA_TIMEOUT: int = 60

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
    # Optional Ragas evaluation. Keep disabled so the API process never needs
    # to import Ragas unless explicitly configured.
    RAGAS_ENABLED: bool = False
    RAGAS_EXECUTION_MODE: Literal["process", "in_process"] = "process"
    RAGAS_PYTHON_EXECUTABLE: str = ""
    RAGAS_METRICS: str = "id_context_precision,id_context_recall"
    RAGAS_TIMEOUT: float = 300.0
    RAGAS_LLM_BASE_URL: str = ""
    RAGAS_LLM_API_KEY: str = ""
    RAGAS_LLM_MODEL: str = ""
    # 阶段 2C: 知识图谱
    GRAPH_ENABLED: bool = False            # 上传时是否抽取实体/关系（慢，需 LLM 调用）
    GRAPH_MAX_CHUNKS_PER_FILE: int = 30    # 单文件最多送入抽取的 chunk 数（成本控制）
    GRAPH_QUERY_TOP_ENTITIES: int = 3      # 检索增强时最多展开的实体数
    # 阶段 5: Neo4j GraphRAG 图谱
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "easyrag_neo4j_secret"
    GRAPH_ENTITY_COLLECTION: str = "graph_entity_index"   # Milvus 中唯一实体/三元组的语义索引 collection
    GRAPH_BUILD_BATCH_SIZE: int = 10       # 构建时 embedding 的批大小
    GRAPH_RRF_K: int = 60                  # RRF 融合常数（reciprocal rank fusion）
    GRAPH_ENTITY_TOP_K: int = 5            # 图谱召回时最多命中的实体/三元组数
    GRAPH_EXTRACT_CONCURRENCY: int = 4     # 构建时 LLM 抽取并发数（注意 API 限流，过大易 429）
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    # 阶段 2A: 分块策略
    #   fixed        固定窗口滑窗（原行为）
    #   recursive    递归分隔符切分（段落→句子→词，尽量在语义边界断开）
    #   markdown     Markdown 结构感知（按标题层级聚合，代码块不拆）
    #   parent_child 父子分块（小块索引用于检索，返回所属大块作为上下文）
    CHUNK_STRATEGY: Literal["fixed", "recursive", "markdown", "parent_child"] = "recursive"
    PARENT_CHUNK_SIZE: int = 1500   # parent_child 策略下父块（上下文块）大小

    # Document parsing / MinerU
    # Keep MinerU behind a feature flag until the ingestion pipeline is wired to it.
    MINERU_ENABLED: bool = False
    MINERU_API_URL: str = "http://127.0.0.1:18000"
    MINERU_BACKEND: str = "pipeline"
    MINERU_LANG: str = "ch"
    MINERU_CONNECT_TIMEOUT: float = 10.0
    MINERU_REQUEST_TIMEOUT: float = 60.0
    MINERU_RESULT_DOWNLOAD_TIMEOUT: float = 600.0
    MINERU_TASK_TIMEOUT: float = 3600.0
    MINERU_POLL_INTERVAL: float = 2.0
    MINERU_FALLBACK_TO_LOCAL: bool = True

    # ── 增强检索 (阶段 3) ──────────────────────────────────────────────────
    # 是否启用增强检索（查询分解 × 四路并行 × 图谱融合重排 × 知识块聚类 × 迭代补充）
    ENHANCED_RETRIEVAL_ENABLED: bool = False
    # 迭代缺口检测：发现检索不足时自动补充检索（最多2轮）
    ENHANCED_ITERATIVE_GAP_FILLING: bool = True
    ENHANCED_MAX_GAP_ROUNDS: int = 2
    # 融合重排序权重 (α=向量相似度, β=图谱距离, γ=跨路共识, δ=时效性)
    ENHANCED_FUSION_ALPHA: float = 0.35
    ENHANCED_FUSION_BETA: float = 0.25
    ENHANCED_FUSION_GAMMA: float = 0.25
    ENHANCED_FUSION_DELTA: float = 0.15
    # 每条路径的最大候选数
    ENHANCED_TOP_K_PER_PATH: int = 6
    ENHANCED_FINAL_TOP_K: int = 8

    # ── Reranker (交叉编码器精排) ────────────────────────────────────────
    RERANKER_TYPE: Literal["disabled", "local", "openai_compatible"] = "disabled"
    RERANKER_MODEL_PATH: str = "./models/bge-reranker-v2-m3"
    RERANKER_API_BASE: Optional[str] = None
    RERANKER_API_KEY: Optional[str] = None
    RERANKER_MODEL_NAME: str = "bge-reranker-v2-m3"
    RERANKER_TOP_K: int = 5     # 精排后保留条数
    RERANKER_MAX_LENGTH: int = 512  # cross-encoder 最大输入长度

    # ── Agent / LangGraph ────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 20   # LangGraph recursion_limit
    MAX_PLAN_STEPS: int = 5          # max sub-tasks per plan
    SESSION_TTL: int = 3600          # seconds to keep session state
    AGENT_MODE: Literal["auto", "single", "multi"] = "auto"  # 多智能体开关: auto=智能路由

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

    # Encrypts user-created model API keys stored in PostgreSQL. When empty,
    # local development falls back to JWT_SECRET_KEY; production should set a
    # separate, stable high-entropy value so credential rotation is explicit.
    MODEL_CONFIG_ENCRYPTION_KEY: str = ""

    # ── Tavily Web Search ────────────────────────────────────────────────
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "basic"   # "basic" | "advanced"
    TAVILY_INCLUDE_ANSWER: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
