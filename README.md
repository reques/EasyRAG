# EasyRAG

> 企业知识库智能问答平台 — 多策略 RAG + Agent 工具调用 + 知识图谱，开箱即用的全栈应用。

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Vue-3.5-42b883?logo=vue.js&logoColor=white)](https://vuejs.org/)
[![Milvus](https://img.shields.io/badge/Milvus-vector_DB-00d4aa)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

EasyRAG 是一个面向真实业务场景的检索增强生成（RAG）平台。区别于"跑通 demo 即止"的玩具项目，它具备生产级应用的完整要素：**多用户认证、文档管理、流式对话、知识图谱、检索评估**，全部开箱即用。

---

## 功能总览

| 模块 | 能力 |
|------|------|
| **智能对话** | SSE 流式输出逐 token 渲染 · 意图识别自动分流（闲聊/计算/联网搜索/知识库问答）· 引用来源可点击跳转到原文档 |
| **知识库管理** | 多知识库隔离 · txt/md/pdf/docx/图片上传 · 扫描件 OCR 识别 · 文件预览（PDF 原生渲染/图片/文本）· 两阶段上传进度条（传输 + 索引） |
| **检索增强** | Milvus 向量检索 · 多策略分块（fixed/recursive/markdown/parent-child）· 知识图谱实体关系抽取与图谱查询注入 |
| **Agent 工具** | web_search（Tavily）· calculator · datetime · text_tool，意图识别驱动自动调用 |
| **评估体系** | 命名评估运行（HitRate / MRR / avg_score）· 逐条命中明细，支撑检索策略调优 |
| **用户体系** | JWT 注册登录 · 会话历史持久化 · 知识库按用户隔离 |

---

## 技术栈

| 层 | 选型 |
|----|------|
| 前端 | Vue 3.5 · Vite 6 · Pinia · Axios · lucide 图标 |
| 后端 | FastAPI（async）· SQLAlchemy 2.0 async · LangGraph 工作流 |
| 存储 | PostgreSQL（pgvector 镜像，业务数据 + 图谱）· Redis · MinIO（文件对象存储） |
| 向量 | Milvus 2.x（etcd + MinIO 依赖）· BGE-M3 embedding（Ollama 本地 / API） |
| LLM | DeepSeek / OpenAI 兼容 API（可配置 base_url） |
| 部署 | Docker Compose 6 服务一键编排 |

---

## 快速开始

### 前置依赖

- Python 3.11+
- Node.js 18+
- Docker Desktop
- Ollama（本地 embedding，可选——也可配置远程 embedding API）

### 1. 启动基础设施

```bash
git clone https://github.com/reques/EasyRAG.git
cd EasyRAG
docker compose up -d        # etcd + milvus + minio + postgres + redis + easyrag-minio
```

### 2. 配置环境

```bash
cp .env.template .env       # 按需修改 LLM / embedding / 各服务连接
```

关键配置项：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 默认/共享网关生成模型 |
| `LLM_DEFAULT_MODEL_ID` | 对话页默认模型，默认 `deepseek-v4-flash` |
| `MINIMAX_*` / `DEEPSEEK_*` / `QWEN_*` / `GLM_*` | 对话页四个可切换模型的地址、模型名与温度 |
| `MINIMAX_API_KEY` / `DASHSCOPE_API_KEY` / `ZHIPUAI_API_KEY` | 各供应商密钥；使用同一 `LLM_BASE_URL` 网关时可留空并复用默认密钥 |
| `MODEL_CONFIG_ENCRYPTION_KEY` | 加密数据库中的自定义模型 API Key；生产环境建议设置独立随机值 |
| `EMBEDDING_TYPE` | `ollama`（本地）或 `api`（远程） |
| `OLLAMA_EMBED_MODEL` | 默认 `bge-m3:latest` |
| `TAVILY_API_KEY` | 联网搜索（可选，不配则 web_search 走兜底） |
| `GRAPH_ENABLED` | 知识图谱抽取开关 |
| `JWT_SECRET_KEY` | 生产环境务必修改 |

### 3. 初始化数据库

```bash
pip install -r requirements.txt
pip install -r requirements-stage1.txt
python -c "import asyncio; from backend.storage.postgres.manager import init_db; asyncio.run(init_db())"
```

### 4. 启动后端

```bash
uvicorn backend.server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173, /api 代理到 :8000
```

打开 http://localhost:5173 注册账号，创建知识库，上传文档，开始对话。对话输入框左下角可在
MiniMax-M2.7、DeepSeek-V4-Flash、Qwen-3.6-Flash 和 GLM-5.2 之间切换；未配置密钥的直连模型
会显示为“未配置”且不可选择。模型菜单中的“添加自定义模型”可新增 Ollama、LM Studio 等
本地 OpenAI 兼容服务，或其他云端 OpenAI 兼容接口；这些动态配置按用户保存在 PostgreSQL，
API Key 加密保存，不写入 `.env`、也不会返回前端。修改内置模型 `.env` 后需要重启后端。

---

## 项目结构

```
EasyRAG/
├── app/                        # Agent 核心（LangGraph 工作流）
│   ├── graph/                  #   状态图：意图识别→检索/工具→生成→校验
│   ├── llm/                    #   LLM 客户端（同步/流式/JSON）
│   ├── rag/                    #   分块 / embedding / 向量检索（Milvus/Chroma/Memory 三后端）
│   ├── tools/                  #   工具注册表：web_search, calculator, datetime, text_tool
│   └── services/agent_service.py
├── backend/                    # 业务后端（分层架构）
│   ├── server/routers/         #   /auth /chat /knowledge /evaluation
│   ├── services/               #   业务逻辑（对话/知识库/图谱/评估/认证）
│   ├── repositories/           #   数据访问层
│   └── storage/                #   postgres / redis / minio 客户端
├── frontend/                   # Vue 3 SPA
│   └── src/views/              #   Chat / Knowledge / Login / Register / Layout
├── scripts/                    # 验证脚本（消息持久化 / SSE / Milvus 迁移）
├── docker-compose.yml          # 6 服务编排
├── ARCHITECTURE.md             # 完整架构文档（目录树 + 分层说明）
└── PROGRESS.md                 # 逐次迭代的演进记录
```

---

## API 概览

| 端点 | 说明 |
|------|------|
| `POST /api/v1/auth/register` `/login` | 注册 / 登录（JWT） |
| `POST /api/v1/chat/send` | 同步对话（完整 LangGraph 工作流） |
| `POST /api/v1/chat/stream` | SSE 流式对话（意图识别 → 检索/工具 → 逐 token 生成） |
| `GET /api/v1/chat/conversations` | 会话列表 / 历史 |
| `POST /api/v1/knowledge/bases` | 创建知识库 |
| `POST /api/v1/knowledge/bases/{id}/upload` | 上传文件（202 异步，进度轮询） |
| `GET /api/v1/knowledge/bases/{id}/files` | 文件列表（含索引进度/状态） |
| `GET .../files/{fid}/preview` `/raw` | 文件预览 / 原始二进制 |
| `GET /api/v1/knowledge/bases/{id}/graph` | 知识图谱（实体 + 关系） |
| `POST /api/v1/evaluation/runs` | 创建检索评估运行 |

完整交互式文档：启动后端后访问 http://localhost:8000/docs

---

## 核心设计

**意图识别分流**：每条消息先经意图分类（chitchat / tool_use / knowledge_qa / complex_task），闲聊直接对话、计算走 calculator、实时信息走 web_search、知识问题走向量检索——避免"不管什么 query 都硬检索"的粗暴实现。

**异步索引进度**：文件上传立即返回 202，解析/embedding/图谱抽取放后台任务，按阶段更新 progress（10%→30%→80%→100%），前端轮询渲染进度条。大文件可关弹窗后台继续。

**executor 线程连接池隔离**：FastAPI `run_in_executor` 的 worker 线程内 DB 查询一律走随用随建的独立 engine，杜绝异步连接池跨事件循环污染（这个坑的完整排查记录在 PROGRESS.md）。

**引用可溯源**：检索结果携带 `knowledge_base_id + file_id`，对话页引用点击直接跳转知识库对应文档的预览弹窗。

---

## 演进路线

- [x] 阶段 1：后端架构化（FastAPI 分层 + JWT + PostgreSQL/Redis/MinIO）
- [x] 阶段 2：知识库增强（Milvus / 多策略分块 / OCR / 知识图谱 / 评估）
- [x] 阶段 3：前端产品化（Vue 3 SPA / SSE 流式 / 进度条 / 引用跳转）
- [ ] 阶段 4：多 Agent 体系（MCP / Skill 系统 / 多租户）

详细迭代记录见 [PROGRESS.md](PROGRESS.md)，架构细节见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## License

MIT — 见 [LICENSE](LICENSE)

如果这个项目对你有帮助，欢迎 Star ⭐ 或提交 Issue / PR。
