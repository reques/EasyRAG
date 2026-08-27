"""Centralised configuration via pydantic-settings.

All values can be overridden by environment variables or a .env file.
Call `get_settings()` everywhere you need config (returns a cached singleton).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 2026-08-21 修复：本地直连不走代理。
# 用户环境设置了 HTTP_PROXY/HTTPS_PROXY（如 Clash 127.0.0.1:7897），grpcio
# 会经代理连 Milvus（localhost:19530）导致连接失败（TCP 通但 gRPC 握手超时）。
# 这里把本机回环地址加入 no_proxy：Milvus/Postgres/Redis 等本地服务直连，
# 外部 API 请求仍按需走 HTTP_PROXY（科学上网不受影响）。
_NO_PROXY_HOSTS = ("127.0.0.1", "localhost", "::1")
_no_proxy_cur = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
if not any(h in _no_proxy_cur for h in _NO_PROXY_HOSTS):
    os.environ["no_proxy"] = ",".join(
        x for x in (_no_proxy_cur, *_NO_PROXY_HOSTS) if x
    )


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
    LLM_TIMEOUT: int = 30          # seconds（检索路径串行多次调用，短超时避免累加超前端 120s）
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
    GRAPH_LLM_CONCURRENCY: int = 6          # 图谱抽取并发调用 LLM 的并发数（串行 30 次太慢，并发提速）
    GRAPH_QUERY_TOP_ENTITIES: int = 3      # 检索增强时最多展开的实体数
    # 文件索引消息队列（Redis Stream，2026-08-27）
    INGESTION_CONCURRENCY: int = 3          # 索引 worker 同时处理的最大文件数（全局闸门）
    INGESTION_PENDING_CLAIM_MS: int = 1800000  # pending 消息认领超时（毫秒，默认 30 分钟；
    #   必须大于单文件处理时长，否则处理中的消息会被 XAUTOCLAIM 误认领重跑）
    INGESTION_LOCK_TTL: int = 1800          # 单文件处理锁 TTL（秒，默认 30 分钟）
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    # 阶段 2A: 分块策略
    #   fixed        固定窗口滑窗（原行为）
    #   recursive    递归分隔符切分（段落→句子→词，尽量在语义边界断开）
    #   markdown     Markdown 结构感知（按标题层级聚合，代码块不拆）
    #   parent_child 父子分块（小块索引用于检索，返回所属大块作为上下文）
    CHUNK_STRATEGY: Literal["fixed", "recursive", "markdown", "parent_child", "legal"] = "recursive"
    PARENT_CHUNK_SIZE: int = 1500   # parent_child 策略下父块（上下文块）大小

    # Document parsing / MinerU
    # Keep MinerU behind a feature flag until the ingestion pipeline is wired to it.
    MINERU_ENABLED: bool = True
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
    # 查询分解缓存 TTL（秒）：同一 query 在窗口内返回完全相同的子问题划分。
    # LLM 在 temperature=0 下仍会波动（代理端点采样/路由），缓存是唯一能保证
    # 「同一问题结果一致」的手段。默认 3600（1 小时），调大可更稳定，调小更灵敏
    ENHANCED_DECOMPOSITION_CACHE_TTL: int = 3600
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
    # 执行路径: auto=智能路由(单 Agent/多智能体按规则分流) | single=仅单 Agent |
    # multi=deepagents 的兼容别名(已废弃) | deepagents=DeepAgents 主 Agent+SubAgent
    AGENT_MODE: Literal["auto", "single", "multi", "deepagents"] = "auto"

    # ── DeepAgents (AGENT_MODE=deepagents) ────────────────────────────────
    # 外部 SubAgent 配置文件（JSON/YAML，见 subagents.load_subagents 的格式；
    # 为空使用内置默认 research-agent / coding-agent）
    DEEP_SUBAGENTS_FILE: str = ""
    # 主 Agent 与 task 委派 SubAgent 的 LangGraph recursion_limit
    DEEP_MAIN_RECURSION_LIMIT: int = 20
    DEEP_SUBAGENT_RECURSION_LIMIT: int = 20
    # 阶段 2：委派时按任务描述 discover() 动态收窄子智能体工具集（默认关闭；
    # 只能收窄不能放大——仍需通过配置白名单与请求级权限裁决）
    DEEP_DYNAMIC_TOOLS: bool = False

    # ── Answer quality ───────────────────────────────────────────────────
    ANSWER_VALIDATION_ENABLED: bool = True
    ANSWER_MIN_LENGTH: int = 20      # chars below which answer is "too short"

    # ── 快速路径（简单问题零成本分流）──────────────────────────────────
    # 简单常识/问候/计算/时间问题先用规则预判直接回答，跳过 LLM 意图分类、
    # 检索与工具调用（避免"吃坏肚子怎么办"这类问题被误判成联网搜索）。
    # 关闭后回退到旧的纯 LLM 意图分类行为。
    FAST_INTENT_ENABLED: bool = True

    # ── 上下文管理 ──────────────────────────────────────────────────────
    # 会话摘要长期生成失败（LLM 不可用）时，注入 LLM 的原始消息上限：
    # 超出部分取真实尾部并记日志，避免长会话把上下文窗口撑爆。
    # 正常路径（有摘要）不受此限制：摘要承载远期上下文 + 最近 20 条。
    HISTORY_CONTEXT_MAX_MESSAGES: int = 100

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
