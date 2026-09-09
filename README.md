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
| Agent | LangGraph（意图分流 / ReAct 循环 / 校验重试）· DeepAgents 统一多智能体（主 Agent + SubAgent + DAG 委派 + 结构化黑板） |
| 存储 | PostgreSQL（pgvector 镜像，业务数据 + 图谱 + Skill 配置）· Redis · MinIO |
| 向量 | Milvus 2.5（etcd + MinIO 依赖）· BGE-M3 embedding（Ollama 本地 / API） |
| LLM | DeepSeek / MiniMax / Qwen(DashScope) / GLM / 任意 OpenAI 兼容 API |
| 文档解析 | 本地解析器 + 旁路部署 MinerU Pipeline API（Docker） |
| 评估 | 本地确定性指标（HitRate / MRR / avg_score）+ 可选 Ragas（独立 venv） |
| 部署 | Docker Compose 一键编排 |

---

## 快速开始

### 前置依赖

- Docker Desktop（或安装了 Docker Compose 插件的 Docker Engine）

### 1. 配置

```bash
git clone https://github.com/reques/EasyRAG.git
cd EasyRAG
cp .env.template .env
```

在 `.env` 中至少配置一个可用的对话模型 API Key，例如
`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`MINIMAX_API_KEY` 或
`ZHIPUAI_API_KEY`。

### 2. 一键启动

```bash
docker compose up --build -d
```

该命令会统一启动：

- Vue 前端（Nginx）
- FastAPI 后端和内嵌文件索引 Worker
- PostgreSQL、Redis、应用 MinIO
- Milvus、etcd 和 Milvus 专用 MinIO
- Neo4j
- Ollama，并在首次启动时自动拉取 `bge-m3` 嵌入模型

服务健康后打开 [http://localhost:5173](http://localhost:5173)。后端接口和
Swagger 分别位于 `http://localhost:8000/api/v1` 和
[http://localhost:8000/docs](http://localhost:8000/docs)。

首次启动需要下载 Docker 镜像、Python 依赖和嵌入模型，耗时较长。查看状态和日志：

```bash
docker compose ps
docker compose logs -f backend frontend
```

停止服务：

```bash
docker compose down
```

业务数据保存在项目的 `volumes/` 目录和 `easyrag-ollama-data` Docker
volume 中，普通的 `docker compose down` 不会删除。

需要 NVIDIA GPU 版 MinerU 时，在 `.env` 中设置
`DOCKER_MINERU_ENABLED=true`，然后执行：

```bash
docker compose --profile mineru up --build -d
```

端口可通过 `.env` 中的 `FRONTEND_PORT`、`BACKEND_PORT`、
`POSTGRES_EXPOSE_PORT` 等变量修改。完整选项见 `.env.template`。

> 本地源码开发仍可分别运行 Uvicorn 和 Vite。MinerU 详细说明见
> [deploy/mineru/README.md](deploy/mineru/README.md)；Ragas 评估环境见
> [docs/ragas-evaluator.md](docs/ragas-evaluator.md)。

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
