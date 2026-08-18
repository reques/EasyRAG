# EasyRAG 项目架构与设计

> 最后更新：2026-08-18 | 快速上手见 [README](../README.md)，演进记录见 [PROGRESS.md](../PROGRESS.md)

---

## 1. 项目定位

EasyRAG 是一个面向真实业务场景的企业知识库智能问答平台：**多策略 RAG + Agent 工具调用 + 知识图谱 + 多智能体编排**，开箱即用的全栈应用（Vue 3 + FastAPI + LangGraph + Milvus）。

与"跑通 demo 即止"的玩具项目的区别在于它具备生产级要素：多用户 JWT 认证、文档管理、SSE 流式对话、知识图谱、检索评估（确定性指标 + 可选 Ragas）、可配置技能（Skill）系统、MCP 外部工具接入、旁路部署的 MinerU 文档解析服务。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Vue 3 SPA)                         │
│   ChatView · KnowledgeView · Login/Register · 状态栏/任务面板         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ REST / SSE (/api/v1, Vite 代理 :8000)
┌───────────────────────────────▼─────────────────────────────────────┐
│                    Backend (FastAPI async)                           │
│   routers: auth / chat / knowledge / evaluation / mcp                │
│   services: chat · knowledge · graph · evaluation · model_config     │
│             skill_config · ragas · agent_run                         │
│   repositories: 数据访问层                                            │
└───────┬────────────────┬────────────────┬────────────────┬───────────┘
        │                │                │                │
┌───────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼──────────┐
│   Agent 内核   │ │   RAG 管线    │ │  工具系统     │ │  外部服务        │
│ app/agents     │ │ app/rag      │ │ app/tools    │ │                 │
│ app/graph      │ │ (chunker/    │ │ registry     │ │ Postgres(pgvector│
│ (LangGraph)    │ │  embedding/  │ │ + MCP 桥接   │ │  +图谱+业务)     │
│ Orchestrator   │ │  retriever/  │ │ app/skills   │ │ Milvus(向量)     │
│ + Workers      │ │  bm25/rerank/│ │              │ │ Redis · MinIO    │
│                │ │  ocr/parsers)│ │              │ │ Ollama(embedding)│
└────────────────┘ └──────────────┘ └──────────────┘ │ MinerU(旁路解析) │
                                                     └──────────────────┘
```

---

## 3. 技术栈

| 层 | 选型 |
|----|------|
| 前端 | Vue 3.5 · Vite 6 · Pinia · Axios · lucide 图标 |
| 后端 | FastAPI（async）· SQLAlchemy 2.0 async · LangGraph 工作流 |
| Agent | LangGraph StateGraph（意图分流 / ReAct 循环 / 校验重试）· 多智能体编排（Orchestrator + Worker + Blackboard） |
| 存储 | PostgreSQL（pgvector 镜像，业务数据 + 图谱 + Skill 配置）· Redis · MinIO（文件对象存储） |
| 向量 | Milvus 2.5（etcd + MinIO 依赖）· BGE-M3 embedding（Ollama 本地 / API） |
| LLM | DeepSeek / MiniMax / Qwen(DashScope) / GLM / 任意 OpenAI 兼容 API（可配置 base_url，支持自定义模型） |
| 文档解析 | 本地解析器 + 旁路部署 MinerU Pipeline API（Docker，见 `deploy/mineru`） |
| 评估 | 本地确定性指标（HitRate / MRR / avg_score）+ 可选 Ragas（独立 venv worker） |
| 部署 | Docker Compose 7 服务一键编排 |

---

## 4. 目录结构

```
EasyRAG/
├── app/                          # Agent 内核（与业务后端解耦的纯逻辑层）
│   ├── agents/                   #   多智能体编排
│   │   ├── orchestrator.py       #     任务拆解 → Worker 派发 → 结果汇总
│   │   ├── blackboard.py         #     共享黑板（跨 worker 状态）
│   │   └── workers/              #     base.py + rag/code/legal 三个专家 Worker
│   ├── graph/                    #   LangGraph 工作流
│   │   ├── state.py              #     AgentState 定义
│   │   ├── nodes.py              #     各节点实现（意图/规划/检索/工具/生成/校验/兜底）
│   │   ├── router.py             #     节点间路由函数
│   │   └── workflow.py           #     StateGraph 装配与编译
│   ├── llm/                      #   LLM 客户端（同步/流式/JSON 模式，模型分级接口）
│   ├── rag/                      #   RAG 管线
│   │   ├── chunker.py            #     多策略分块（fixed/recursive/markdown/parent-child）
│   │   ├── embeddings.py         #     BGE-M3 embedding（ollama/api）
│   │   ├── vector_store.py       #     向量库三后端（Milvus/Chroma/Memory）
│   │   ├── retriever.py / enhanced_retriever.py / bm25.py / reranker.py
│   │   ├── graph_cache.py        #     知识图谱缓存
│   │   ├── ocr.py                #     扫描件 OCR
│   │   └── parsers/              #     local_parser / mineru_parser / router（按文件类型分流）
│   ├── tools/                    #   工具系统
│   │   ├── registry.py           #     线程安全工具注册表（RLock）
│   │   ├── web_search_tool.py / calculator.py / datetime_tool.py / text_tool.py
│   │   └── mcp/                  #     MCP 客户端桥接（config/manager/demo_server）
│   ├── skills/                   #   Skill 系统（catalog 内置 + context 注入）
│   ├── memory/                   #   分层记忆管理
│   ├── prompts/                  #   Prompt 模板（意图/规划/ReAct/生成/校验）
│   ├── services/                 #   agent_service（编排入口）、knowledge_catalog
│   ├── core/                     #   config / exceptions / logger
│   └── api/                      #   [遗留] 旧版 KB 路由（/api/v1/kb/*）
├── backend/                      # 业务后端（分层架构）
│   ├── server/
│   │   ├── main.py               #   FastAPI 装配 + lifespan（建表/增量列迁移/种子）
│   │   ├── routers/              #   auth / chat / knowledge / evaluation / mcp
│   │   └── seed.py
│   ├── services/                 #   chat / knowledge / graph / evaluation / model_config
│   │                             #   skill_config / ragas_evaluator / ragas_worker / agent_run
│   ├── repositories/             #   数据访问层（skill_config 等）
│   └── storage/                  #   postgres（models_*.py）/ redis / minio 客户端
├── frontend/                     # Vue 3 SPA
│   └── src/
│       ├── views/                #   ChatView / KnowledgeView / Login / Register / Layout
│       ├── stores/               #   Pinia（auth / chat）
│       ├── api/                  #   Axios + JWT 拦截器
│       └── router/ / styles/
├── deploy/mineru/                # MinerU 独立解析服务旁路部署（compose + Dockerfile + smoke-test）
├── verify/                       # 人工验证脚本（multi-agent / blackboard / ragas / auto-route …）
├── scripts/                      # 迁移/验证脚本（migrate_milvus_kb_id、verify_*.py）
├── tests/                        # pytest（检索隔离/进度/解析器/skill 配置/MinerU 客户端…）
├── docs/                         # 文档（本架构文档、plans/ 设计稿、specs/ 规格、ragas-evaluator）
├── docker-compose.yml            # etcd + milvus + minio-s3 + postgres + redis + minio 编排
├── requirements.txt / requirements-stage1.txt / requirements-ragas.txt
├── .env.template                 # 完整配置模板
└── PROGRESS.md                   # 逐次迭代的演进记录
```

---

## 5. 核心模块

### 5.1 LangGraph 工作流（app/graph）

单条消息的处理主流程，`workflow.py` 装配成 StateGraph：

```
intent_recognition
    |--(use_react: complex_task / 低置信度)--> agent_reasoning <─┐
    |                                              │  │          │
    |                              final_answer ───┘  └─ tool ───┘ (ReAct 循环)
    |                                              │
    |                                              ▼
    |                                       answer_validation
    |--(tool_use)-------> tool_selection --> tool_execution --> answer_generation
    |--(knowledge_qa)---> knowledge_retrieval --> answer_generation
    |--(chitchat)-------> answer_generation
answer_generation --> answer_validation --> END
                                        --> answer_generation (1 次重试)
any error --> fallback_handler --> END
```

- **意图识别**（`intent_recognition`）：chitchat / tool_use / knowledge_qa / complex_task 四类分流，避免"不管什么 query 都硬检索"
- **ReAct 循环**（`agent_reasoning`）：复杂任务/低置信度进入推理-工具循环，工具名白名单取自工具注册表
- **校验重试**：`answer_validation` 检查回答质量，不合格最多重生成 1 次
- **兜底**：任何节点异常 → `fallback_handler`，不把错误裸抛给用户

### 5.2 多智能体编排（app/agents）

借鉴 subagent-driven-development 的"任务简报（brief）"思想：

- **Orchestrator**：LLM 把用户查询拆解为结构化 `TaskBrief`，按 `worker_hint` 路由到专家 Worker，用线程池并行执行，汇总各 `WorkerReport` 生成最终回答
- **Workers**：`rag_worker`（知识库问答）、`code_worker`（代码）、`legal_worker`（法律），继承 `BaseWorker`
- **Blackboard**：每次 run 创建的黑板对象，跨 Worker 共享上下文
- 前端"状态栏/任务面板"展示 run 的任务进度（done/total）与状态

### 5.3 RAG 管线（app/rag）

- **分块**：fixed / recursive / markdown 结构感知 / parent-child 四策略（`CHUNK_STRATEGY` 配置）
- **向量化**：BGE-M3，`EMBEDDING_TYPE=ollama`（本地）或 `api`（远程）
- **检索**：Milvus 主后端（Chroma/Memory 供测试），配 BM25 混合 + reranker 重排；命中 child chunk 时 `_unwrap_parent` 回填父块上下文
- **知识图谱**：实体关系抽取注入检索结果（`GRAPH_ENABLED` 开关），`graph_cache` 缓存
- **OCR / 解析**：扫描件 OCR；`parsers/router.py` 按类型分流到本地解析器或 MinerU API（`deploy/mineru` 旁路部署，见其 README）

### 5.4 工具系统（app/tools）

- **注册表**：`ToolRegistry` 用 RLock 保证线程安全（MCP 停止时反注册与 LangGraph 遍历可并发），`invoke()` 在锁外执行工具函数避免阻塞
- **内置工具**：web_search（Tavily，可兜底）、calculator、datetime、text_tool
- **MCP 接入**：`app/tools/mcp/` 常驻事件循环线程 + 同步桥接，支持 stdio / HTTP 双传输；`mcp_router.py` 提供服务器启停 API；AI 调用需把 `mcp_<server>_<tool>` 全名加进 Worker 的 tool_names 白名单（详见 easyrag-frontend-workflow skill 的 references/mcp-integration.md）

### 5.5 Skill 系统（app/skills + backend skill_config）

- **内置 Skill**：知识库研究、联网研究、数据分析、专业写作、法律分析（`catalog.py` 的 `SkillProfile`）
- **自定义 Skill**：用户可创建/复制/编辑（名称、用途、详细指令 + 最小权限工具白名单），存 PostgreSQL（`models_skill_config`），API Key 等敏感配置加密
- **注入机制**：`context.py` 把选中 Skill 的指令作为本次请求的系统上下文注入；工具白名单在后端执行层强制校验——取消工具权限不是前端展示变化
- 同一条消息最多组合 3 个 Skill；历史消息保留 Skill 名称快照

### 5.6 LLM 层（app/llm）

- 统一客户端：同步 / 流式（SSE）/ JSON 模式
- 模型分级接口 + 按用户自定义模型配置（`model_config` 服务，PostgreSQL 存储，加密 API Key）
- 对话页可切换 MiniMax / DeepSeek / Qwen / GLM，或添加任意 OpenAI 兼容服务（Ollama、LM Studio…）

### 5.7 业务后端（backend）

分层：routers（HTTP 契约）→ services（业务逻辑）→ repositories（数据访问）→ storage（Postgres/Redis/MinIO 客户端）。要点：

- **executor 线程连接池隔离**：FastAPI `run_in_executor` 的 worker 线程内 DB 查询一律走随用随建的独立 engine，杜绝异步连接池跨事件循环污染
- **lifespan 自举**：启动时 `init_db()` 建表 + 增量列迁移（开发用，生产应换 Alembic）+ 种子数据
- **异步任务**：文件上传返回 202，解析/embedding/图谱抽取放后台任务，阶段化更新进度

### 5.8 前端（frontend）

- ChatView：SSE 流式逐 token 渲染、引用来源可点击跳转原文档、模型切换、Skill 选择标签、任务状态栏（可收起/展开）
- KnowledgeView：多知识库隔离、两阶段上传进度条（传输 + 索引）、文件预览
- 全局设计 token 体系（style.css，main 青色系 + gray 色阶）

---

## 6. 关键设计决策

1. **意图识别分流**：先分类再干活，避免无差别检索（`route_after_intent`）
2. **ReAct 子图**：复杂任务走推理-工具循环，普通任务走直通管道，兼顾质量与延迟
3. **多智能体并行**：任务拆解后 Worker 线程池并行执行，黑板上下文共享
4. **SSE 流式**：`/chat/stream` 边生成边推送，前端逐 token 渲染
5. **异步索引**：202 + 阶段进度轮询（10%→30%→80%→100%），大文件可关弹窗后台继续
6. **Skill 执行层强制**：工具白名单在后端校验，指令注入系统上下文，自定义 Skill 无需改代码/重启
7. **MCP 桥接**：常驻线程事件循环隔离 async 生命周期，同步桥接暴露给工具执行层
8. **引用可溯源**：检索结果携带 `knowledge_base_id + file_id`，点击跳转文档预览
9. **评估体系**：本地确定性指标（HitRate/MRR/avg_score）+ 可选 Ragas 独立 venv worker（避免升级主服务依赖，见 docs/ragas-evaluator.md）
10. **MinerU 旁路部署**：解析服务 Docker 独立运行，不污染主 Python 环境（见 deploy/mineru/README.md）

---

## 7. API 概览

统一前缀 `/api/v1`（完整交互文档：启动后端后访问 http://localhost:8000/docs）

### 认证 auth
| 端点 | 说明 |
|------|------|
| `POST /auth/register` / `POST /auth/login` | 注册 / 登录（JWT） |

### 对话 chat
| 端点 | 说明 |
|------|------|
| `POST /chat/send` | 同步对话（完整 LangGraph 工作流） |
| `POST /chat/stream` | SSE 流式对话 |
| `GET /chat/conversations` | 会话列表 |
| `GET /chat/conversations/{id}/history` | 会话历史 |
| `POST /chat/conversations/{id}/summarize` | 会话摘要 |
| `GET /chat/conversations/{id}/runs` | 会话的 agent run（任务面板数据） |
| `GET/POST/PUT/DELETE /chat/skills` | Skill 目录 / 自定义 Skill CRUD |
| `GET/POST/DELETE /chat/models` | 自定义模型配置 CRUD |
| `GET /chat/tools` | 可用工具列表 |

### 知识库 knowledge
| 端点 | 说明 |
|------|------|
| `GET/POST /knowledge/bases` | 知识库列表 / 创建 |
| `POST /knowledge/bases/{id}/upload` | 上传文件（202 异步，进度轮询） |
| `GET /knowledge/bases/{id}/files` | 文件列表（含索引进度/状态） |
| `GET .../files/{fid}/preview` / `raw` | 文件预览 / 原始二进制 |
| `DELETE .../files/{fid}` | 删除文件 |
| `GET /knowledge/bases/{id}/graph` | 知识图谱（实体 + 关系） |

### 评估 evaluation
| 端点 | 说明 |
|------|------|
| `POST /evaluation/runs` | 创建检索评估运行（命名运行） |
| `GET /evaluation/runs` / `GET /evaluation/runs/{id}` | 评估运行列表 / 明细 |

### MCP mcp
| 端点 | 说明 |
|------|------|
| `GET /mcp/servers` | 已配置的 MCP 服务器 |
| `GET /mcp/servers/{name}/tools` | 服务器工具列表 |
| `POST /mcp/servers/{name}/start` / `stop` | 启停 MCP 服务器 |

### 遗留 legacy（app/api）
| 端点 | 说明 |
|------|------|
| `POST /kb/upload` · `POST /kb/ingest_texts` · `POST /kb/search` · `POST /kb/ask` · `DELETE /kb/collection` · `GET /kb/health` | 旧版 KB 直连 API |

---

## 8. 功能总览

| 模块 | 能力 |
|------|------|
| **智能对话** | SSE 流式逐 token 渲染 · 意图识别自动分流（闲聊/计算/联网搜索/知识库问答）· 引用来源可点击跳转 · 多模型切换 + 自定义模型 |
| **多智能体** | 任务拆解 → 专家 Worker 并行执行 → 汇总；侧边任务状态栏展示进度 |
| **知识库管理** | 多知识库隔离 · txt/md/pdf/docx/图片上传 · 扫描件 OCR · 本地/MinerU 双解析 · 文件预览 · 两阶段上传进度条 |
| **检索增强** | Milvus 向量检索 · 多策略分块 · BM25 混合 + 重排 · 知识图谱实体关系抽取与查询注入 |
| **Agent 工具** | web_search（Tavily）· calculator · datetime · text_tool · MCP 外部工具 |
| **Skill 系统** | 内置 + 自定义 Skill，指令注入 + 工具白名单执行层强制 |
| **评估体系** | 命名评估运行（HitRate / MRR / avg_score）· 逐条命中明细 · 可选 Ragas |
| **用户体系** | JWT 注册登录 · 会话历史持久化 · 知识库按用户隔离 · 自定义模型/Skill 按用户存储 |

---

## 9. 演进路线

- [x] 阶段 1：后端架构化（FastAPI 分层 + JWT + PostgreSQL/Redis/MinIO）
- [x] 阶段 2：知识库增强（Milvus / 多策略分块 / OCR / 知识图谱 / 评估）
- [x] 阶段 3：Agent 体系（ReAct 循环 + 工具插件化 + 多智能体编排 + Skill + MCP）
- [x] 阶段 4：产品化（Vue 3 SPA / SSE 流式 / 任务状态栏 / 自定义模型与 Skill）
- [ ] 阶段 5：多租户 / 管理后台 / 生产化（Alembic 迁移、网关鉴权）

详细迭代记录见 [PROGRESS.md](../PROGRESS.md)

---

## 10. 相关文档

| 文档 | 内容 |
|------|------|
| [README](../README.md) | 项目简介、技术栈、快速开始 |
| [PROGRESS.md](../PROGRESS.md) | 逐次迭代的演进记录与路线图 |
| [docs/plans/](../plans/) | 设计稿（ReAct 内核、Skill 配置） |
| [docs/specs/](../specs/) | 规格说明 |
| [docs/ragas-evaluator.md](../ragas-evaluator.md) | 可选 Ragas 评估的独立环境部署 |
| [deploy/mineru/README.md](../../deploy/mineru/README.md) | MinerU 独立解析服务部署与运维 |
