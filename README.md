# EasyRAG

> 企业知识库智能问答平台 — 多策略 RAG + Agent 工具调用 + 知识图谱 + 多智能体编排，开箱即用的全栈应用。

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Vue-3.5-42b883?logo=vue.js&logoColor=white)](https://vuejs.org/)
[![Milvus](https://img.shields.io/badge/Milvus-vector_DB-00d4aa)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 技术栈

| 层 | 选型 |
|----|------|
| 前端 | Vue 3.5 · Vite 6 · Pinia · Axios · lucide 图标 |
| 后端 | FastAPI（async）· SQLAlchemy 2.0 async · LangGraph 工作流 |
| Agent | LangGraph（意图分流 / ReAct 循环 / 校验重试）· 多智能体编排（Orchestrator + Worker + Blackboard） |
| 存储 | PostgreSQL（pgvector 镜像，业务数据 + 图谱 + Skill 配置）· Redis · MinIO |
| 向量 | Milvus 2.5（etcd + MinIO 依赖）· BGE-M3 embedding（Ollama 本地 / API） |
| LLM | DeepSeek / MiniMax / Qwen(DashScope) / GLM / 任意 OpenAI 兼容 API |
| 文档解析 | 本地解析器 + 旁路部署 MinerU Pipeline API（Docker） |
| 评估 | 本地确定性指标（HitRate / MRR / avg_score）+ 可选 Ragas（独立 venv） |
| 部署 | Docker Compose 一键编排 |

---

## 快速开始

### 前置依赖

- Python 3.11+ / Node.js 18+ / Docker Desktop
- Ollama（本地 embedding，可选——也可配置远程 embedding API）

### 1. 启动基础设施

```bash
git clone https://github.com/reques/EasyRAG.git
cd EasyRAG
docker compose up -d        # etcd + milvus + minio-s3 + postgres + redis + minio
```

### 2. 配置环境

```bash
cp .env.template .env       # 按需修改 LLM / embedding / 各服务连接
```

关键配置项见 `.env.template` 注释：`DEEPSEEK_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL`（默认生成模型）、`EMBEDDING_TYPE`（ollama/api）、`TAVILY_API_KEY`（联网搜索）、`GRAPH_ENABLED`（图谱抽取）、`JWT_SECRET_KEY`（生产务必修改）。

### 3. 初始化数据库

```bash
pip install -r requirements.txt
python -c "import asyncio; from backend.storage.postgres.manager import init_db; asyncio.run(init_db())"
```

### 4. 启动后端

```bash
uvicorn backend.server.main:app --host 0.0.0.0 --port 8001 --reload
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173, /api 代理到 :8000
```

打开 http://localhost:5173 注册账号，创建知识库，上传文档，开始对话。对话页可切换 MiniMax / DeepSeek / Qwen / GLM 模型、添加自定义 OpenAI 兼容模型、选择或创建 Skill。

> 可选：MinerU 文档解析服务旁路部署见 [deploy/mineru/README.md](deploy/mineru/README.md)；Ragas 评估环境见 [docs/ragas-evaluator.md](docs/ragas-evaluator.md)。

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 项目结构与逻辑（模块详解、核心设计、API 概览） |
| [docs/ARCHITECTURE_DETAILED.md](docs/ARCHITECTURE_DETAILED.md) | 整体架构深度详解（请求生命周期、LangGraph 工作流、增强检索流水线、图谱子系统） |
| [PROGRESS.md](PROGRESS.md) | 逐次迭代的演进记录 |
| [docs/plans/](docs/plans/) · [docs/specs/](docs/specs/) | 设计稿与规格说明 |
| [deploy/mineru/README.md](deploy/mineru/README.md) | MinerU 解析服务部署与运维 |
| [docs/ragas-evaluator.md](docs/ragas-evaluator.md) | 可选 Ragas 评估部署 |

---

## License

MIT — 见 [LICENSE](LICENSE)
