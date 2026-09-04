# EasyRAG 项目架构与设计

> 最后更新：2026-08-31（上传管线迁 Redis Stream 队列、structured 分块、图谱命名空间隔离） | 快速上手见 [README](../README.md)，演进记录见 [PROGRESS.md](../PROGRESS.md)

---

## 1. 项目定位

EasyRAG 是一个面向真实业务场景的企业知识库智能问答平台：**多策略 RAG + Agent 工具调用 + 知识图谱 + 多智能体编排**，开箱即用的全栈应用（Vue 3 + FastAPI + LangGraph + Milvus）。

与"跑通 demo 即止"的玩具项目的区别在于它具备生产级要素：多用户 JWT 认证、文档管理、SSE 流式对话、知识图谱、检索评估（确定性指标 + 可选 Ragas）、可配置技能（Skill）系统、MCP 外部工具接入、旁路部署的 MinerU 文档解析服务。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Vue 3 SPA)                         │
│   ChatView · KnowledgeView · Login/Register · 状态栏/任务面板 · 工作进度面板        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ REST / SSE (/api/v1, Vite 代理 :8001)
┌───────────────────────────────▼─────────────────────────────────────┐
│                    Backend (FastAPI async)                           │
│   routers: auth / chat / knowledge / evaluation / mcp                │
│   services: chat · knowledge · graph · evaluation · model_config     │
│             skill_config(索引) · ragas · agent_run                   │
│   repositories: 数据访问层                                            │
└───────┬────────────────┬────────────────┬────────────────┬───────────┘
        │                │                │                │
┌───────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼──────────┐
│   Agent 内核   │ │   RAG 管线    │ │  工具系统     │ │  外部服务        │
│ app/agents     │ │ app/rag      │ │ app/tools    │ │                 │
│ app/graph      │ │ (chunker/    │ │ registry     │ │ Postgres(pgvector│
│ (LangGraph)    │ │  embedding/  │ │ + MCP 桥接   │ │  +图谱+业务)     │
│ DeepAgents     │ │  retriever/  │ │ skills/*.md  │ │ Milvus(向量)     │
│ 委派协同        │ │  bm25/rerank/│ │              │ │ Redis · MinIO    │
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
| Agent | LangGraph StateGraph（意图分流 / ReAct 循环 / 校验重试）· DeepAgents 统一多智能体（主 Agent + SubAgent + DAG 委派 + 结构化黑板） |
| 存储 | PostgreSQL（pgvector 镜像，业务数据 + 图谱 + Skill 索引）· Redis · MinIO（文件对象存储）· 本地文件系统（Skill 定义 `SKILL.md`） |
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
│   ├── agents/                   #   多智能体层（DeepAgents 统一实现）
│   │   ├── events.py             #     统一事件流（请求级 trace + emit + 事件汇聚）
│   │   ├── progress.py           #     进度投影器（SSE 进度摘要）
│   │   └── deep/                 #     主 Agent / task、spawn_tasks 委派 / 结构化黑板 / 子智能体名册
│   ├── graph/                    #   LangGraph 工作流
│   │   ├── state.py              #     AgentState 定义
│   │   ├── nodes.py              #     各节点实现（意图/规划/检索/工具/生成/校验/兜底）
│   │   ├── router.py             #     节点间路由函数
│   │   └── workflow.py           #     StateGraph 装配与编译
│   ├── llm/                      #   LLM 客户端（同步/流式/JSON 模式，模型分级接口）
│   ├── rag/                      #   RAG 管线
│   │   ├── chunker.py            #     多策略分块（fixed/recursive/markdown/parent-child/legal/structured）
│   │   ├── embeddings.py         #     BGE-M3 embedding（local/ollama/api）
│   │   ├── vector_store.py       #     向量库三后端（Milvus/Chroma/Memory）
│   │   ├── retriever.py / enhanced_retriever.py / bm25.py / reranker.py
│   │   ├── graph_cache.py        #     知识图谱缓存
│   │   ├── ocr.py                #     扫描件 OCR
│   │   └── parsers/              #     local_parser / mineru_parser / router（按文件类型分流）
│   ├── tools/                    #   工具系统
│   │   ├── registry.py           #     线程安全工具注册表（RLock）
│   │   ├── web_search_tool.py / calculator.py / datetime_tool.py / text_tool.py
│   │   └── mcp/                  #     MCP 客户端桥接（config/manager/demo_server）
│   ├── skills/                   #   Skill 系统（SKILL.md 文件 + 渐进式披露）
│   │   ├── loader.py             #     SKILL.md 解析 + frontmatter 校验
│   │   ├── registry.py           #     两来源磁盘索引（builtin / personal）
│   │   ├── runtime.py            #     三层集合 + activated_skills 工具门控
│   │   ├── read_tool.py          #     read_skill 工具（激活入口）
│   │   └── middleware.py         #     SkillsMiddleware（挂 create_agent）
│   ├── memory/                   #   分层记忆管理
│   ├── prompts/                  #   Prompt 模板（意图/规划/ReAct/生成/校验）
│   ├── services/                 #   agent_service（编排入口）、knowledge_catalog
│   ├── core/                     #   config / exceptions / logger
│   └── api/                      #   [遗留] 旧版 KB 路由（/api/v1/kb/*）
├── backend/                      # 业务后端（分层架构）
│   ├── server/
│   │   ├── main.py               #   FastAPI 装配 + lifespan（建表/增量列迁移/种子/ingestion worker 内嵌启停）
│   │   ├── routers/              #   auth / chat / knowledge / evaluation / mcp
│   │   └── seed.py
│   ├── services/                 #   chat / knowledge / graph / evaluation / model_config
│   │                             #   skill_config / ragas_evaluator / ragas_worker / agent_run
│   │                             #   ingestion_service（索引执行）/ ingestion_queue（Redis Stream 发布/ACK）
│   ├── worker/                   #   ingestion_worker（队列消费者，默认内嵌 uvicorn）
│   ├── repositories/             #   数据访问层（skill_config 等）
│   └── storage/                  #   postgres（models_*.py）/ redis（manager 单例 + RedisLock 工厂）/ minio 客户端
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
├── requirements.txt / requirements-ragas.txt
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

### 5.1.6 轻量动态 Agent（app/agents/dynamic，默认普通问题链路）

2026-08-31 起，``AGENT_MODE=auto`` 的普通问题与 ``AGENT_MODE=dynamic`` 统一走轻量动态
Agent（``create_react_agent`` + 注册表工具，无委派工具）：

- **动态决策**：模型每轮通过函数调用自行决定「直接回答 / 调工具 / 检索知识库」，
  不再依赖固定 intent -> retrieval/tool 管线；简单问题一次 LLM 调用即结束
- **工具集**：仅项目注册表普通工具（web_search / kb_search / calculator /
  datetime_tool / text_tool / MCP 工具），不引入 task/spawn_tasks 委派
- **流式透出**：/chat/stream 与 DeepAgents 分支同构，工具调用/检索状态实时
  SSE 推给前端，徽标显示「动态」
- **兼容**：``AGENT_MODE=single`` 仍保留旧固定管线；deepagents / multi 不变

### 5.2 多智能体（app/agents/deep，DeepAgents 统一实现）

2026-08-26 起，多智能体统一到 LangGraph 原生 DeepAgents（原 Orchestrator-Worker
已退役删除）：

- **主 Agent**（`deep/agent.py`）：`create_react_agent` + checkpointer 会话记忆，
  自行决定何时委派；`AGENT_MODE=multi` 作为 `deepagents` 的兼容别名保留
- **委派工具**：`task`（单任务委派，含熔断与降级）、`spawn_tasks`（DAG 并行委派，
  `depends_on` 拓扑分层 + 线程池）、`revise_plan`（运行中动态重规划）
- **子智能体**（`deep/subagents.py`）：内置 research-agent / coding-agent，支持
  外部 JSON/YAML 配置与动态工具绑定（`*` / `except:` / `@tag`）
- **结构化黑板**（`deep/blackboard.py`）：任务产出物 `{key, producer, summary, data,
  tags, version}` 两级共享 + 依赖订阅注入；区别于旧版 500 字摘要黑板（已删）
- **统一事件流**（`agents/events.py`）：请求级 trace + span + 事件汇聚，跨层串联、
  可回放；委派执行落库（Run/Task/AgentRun）并桥接为前端既有任务面板协议

### 5.3 RAG 管线（app/rag）

- **分块**：fixed / recursive / markdown 结构感知 / parent-child / legal（法律条文）/ structured（通用结构感知：标题+编号条目一级切分、超长 section 滑动窗口）六策略（`CHUNK_STRATEGY` 配置）
- **向量化**：BGE-M3，`EMBEDDING_TYPE=local`（本地）/ `ollama`（本地服务）/ `api`（远程）三实现
- **检索**：Milvus 主后端（Chroma/Memory 供测试），配 BM25 混合 + reranker 重排；命中 child chunk 时 `_unwrap_parent` 回填父块上下文
- **知识图谱**：实体关系抽取注入检索结果（`GRAPH_ENABLED` 开关），`graph_cache` 缓存；实体身份 = `(kb_id, source_file, name)` 命名空间隔离（同名跨文件各自成节点，删除文件级联清理图谱）
- **上传索引**：Redis Stream 队列化（`kb:ingestion` + 内嵌 uvicorn 的 ingestion worker，并发闸门 + 处理锁 + 崩溃认领），发布失败回退进程内后台任务
- **OCR / 解析**：扫描件 OCR；`parsers/router.py` 按类型分流到本地解析器或 MinerU API（`deploy/mineru` 旁路部署，见其 README）

### 5.4 工具系统（app/tools）

- **注册表**：`ToolRegistry` 用 RLock 保证线程安全（MCP 停止时反注册与 LangGraph 遍历可并发），`invoke()` 在锁外执行工具函数避免阻塞
- **内置工具**：web_search（Tavily，可兜底）、calculator、datetime、text_tool
- **MCP 接入**：`app/tools/mcp/` 常驻事件循环线程 + 同步桥接，支持 stdio / HTTP 双传输；`mcp_router.py` 提供服务器启停 API；AI 调用需把 `mcp_<server>_<tool>` 全名加进 Worker 的 tool_names 白名单（详见 easyrag-frontend-workflow skill 的 references/mcp-integration.md）

### 5.5 Skill 系统（app/skills + backend skill_config）

2026-09-04 参照 [Yuxi](https://xerrors.github.io/Yuxi/agents/skills-management.html) 重构为**文件定义 + 渐进式披露**（设计稿：`docs/plans/2026-09-04-skill-management-refactor-yuxi.md`）。

- **定义格式**：一个 Skill = 一个目录，根级 `SKILL.md`（YAML frontmatter + Markdown 正文），可选 `prompts/` 与 `tools/`。必填 `name` / `description`；`slug` 是稳定标识（省略时用 name，此时 name 必须是 slug 形态）
- **两来源，文件为真相**：内置随代码发布在 `skills/`（只读）；个人 Skill 在 `volumes/user-skills/<user_id>/<slug>/`。PostgreSQL `custom_skill_configs` 降级为索引表（slug / owner / 展示元数据），不再存指令正文
- **渐进式披露**：用户勾选定义"本次可用范围"（≤ `SKILLS_MAX_SELECTED`，默认 10），首轮 prompt 只给名称 + 用途摘要；模型判断相关时调 `read_skill(slug)` 读全文，该 Skill 进 `activated_skills`，其 `tool_dependencies` 在**下一轮**解锁
- **依赖闭包**：`skill_dependencies` 只展开进"描述范围"，不等于工具立即暴露；闭包只在用户可访问集合内展开，不能借依赖扩大权限
- **双层工具门控**：`SkillsMiddleware.wrap_tool_call`（Agent 路径）+ `ToolRegistry.invoke` 的 ContextVar 检查（子 Agent 线程 / graph 节点 / MCP 桥接）。未激活 Skill 的工具即使已注册也不能调用
- **公共工具**：`metadata["public"]` 标记的基础工具（`kb_search` / `calculator` / `datetime_tool`）不受 Skill 门控——否则启用 Skill 的请求首轮无工具可用。该标记优先于 Skill 声明
- **注入机制**：`middleware.py` 挂在三处 `create_agent` 上，按 `activated_skills` 逐轮渲染 Skill 区块。非 Agent 路径（意图识别 / 直连兜底）无 read_skill 循环，用 `render_prompt(eager=True)` 直接展开全文
- 历史消息保留 Skill 名称快照；`read_skill` 激活时发 `skill_activated` 事件到 SSE，前端任务状态栏可见

### 5.6 LLM 层（app/llm）

- 统一客户端：同步 / 流式（SSE）/ JSON 模式
- 模型分级接口 + 按用户自定义模型配置（`model_config` 服务，PostgreSQL 存储，加密 API Key）
- 对话页可切换 MiniMax / DeepSeek / Qwen / GLM，或添加任意 OpenAI 兼容服务（Ollama、LM Studio…）

### 5.7 业务后端（backend）

分层：routers（HTTP 契约）→ services（业务逻辑）→ repositories（数据访问）→ storage（Postgres/Redis/MinIO 客户端）。要点：

- **executor 线程连接池隔离**：FastAPI `run_in_executor` 的 worker 线程内 DB 查询一律走随用随建的独立 engine，杜绝异步连接池跨事件循环污染
- **lifespan 自举**：启动时 `init_db()` 建表 + 增量列迁移（开发用，生产应换 Alembic）+ 种子数据 + **内嵌启动 ingestion worker**（uvicorn 重启 = worker 重启，消息持久化在 Redis 不丢）
- **异步任务队列**：文件上传返回 202，索引任务发布到 Redis Stream（`kb:ingestion`，消息只带定位信息），worker 消费执行解析/embedding/图谱抽取，阶段化更新进度；发布失败回退进程内 BackgroundTasks 保底

### 5.8 前端（frontend）

- ChatView：SSE 流式逐 token 渲染、引用来源可点击跳转原文档、模型切换、Skill 选择标签、任务状态栏（可收起/展开）、多智能体任务产出面板
- KnowledgeView：多知识库隔离、多文件并行上传（独立进度/失败留列表/可关弹窗）、文件预览、检索测试工作台（basic/enhanced 双模式）、图谱可视化（按文件筛选）
- 工作进度面板（WorkProgress）：progress_summary 驱动的步骤时间线（进行中展开、完成折叠），展示 deep/single 路径的阶段进度
- 全局设计 token 体系（style.css，main 青色系 + gray 色阶）

---

## 6. 关键设计决策

1. **意图识别分流**：先分类再干活，避免无差别检索（`route_after_intent`）
2. **ReAct 子图**：复杂任务走推理-工具循环，普通任务走直通管道，兼顾质量与延迟
3. **多智能体统一**：DeepAgents 主 Agent 自主委派（task / spawn_tasks DAG 并行 + 黑板共享），取代旧 Orchestrator-Worker 手工拆解派发
4. **SSE 流式**：`/chat/stream` 边生成边推送，前端逐 token 渲染
5. **异步索引**：202 + 阶段进度轮询（10%→30%→80%→100%），大文件可关弹窗后台继续
6. **Skill 执行层强制**：工具门控在后端双层校验（middleware + 注册表），未激活 Skill 的工具不可调用；Skill 以文件定义，新增/修改无需改代码或重启
7. **MCP 桥接**：常驻线程事件循环隔离 async 生命周期，同步桥接暴露给工具执行层
8. **引用可溯源**：检索结果携带 `knowledge_base_id + file_id`，点击跳转文档预览
9. **评估体系**：本地确定性指标（HitRate/MRR/avg_score）+ 可选 Ragas 独立 venv worker（避免升级主服务依赖，见 docs/ragas-evaluator.md）
10. **MinerU 旁路部署**：解析服务 Docker 独立运行，不污染主 Python 环境（见 deploy/mineru/README.md）
11. **上传队列进程化**：Redis Stream（消息只带定位信息、重启不丢、并发闸门 + 短 TTL 自续期锁 + 认领超时），worker 内嵌 uvicorn——不为后台任务引入第三终端
12. **图谱命名空间隔离**：实体身份 = `(kb_id, source_file, name)`，同名跨文件各自成节点；删除文件按命名空间级联清理图谱表与内存缓存
13. **通用结构化分块**：识别通用文档层级（标题/编号条目/章节词），通用层零领域词（用户红线）

---

## 7. API 概览

统一前缀 `/api/v1`（完整交互文档：启动后端后访问 http://localhost:8001/docs）

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
| `GET/POST/PUT/DELETE /chat/skills` | Skill 目录 / 个人 Skill CRUD（列表不含正文） |
| `GET /chat/skills/{slug}/content` | 单个 Skill 的 SKILL.md 全文（编辑器用） |
| `GET/POST/DELETE /chat/models` | 自定义模型配置 CRUD |
| `GET /chat/tools` | 可用工具列表 |

### 知识库 knowledge
| 端点 | 说明 |
|------|------|
| `GET/POST /knowledge/bases` | 知识库列表 / 创建 |
| `POST /knowledge/bases/{id}/upload` | 上传文件（202 异步，Redis Stream 队列索引，进度轮询） |
| `GET /knowledge/bases/{id}/files` | 文件列表（含索引进度/状态） |
| `GET .../files/{fid}/preview` / `raw` | 文件预览 / 原始二进制 |
| `DELETE .../files/{fid}` | 删除文件（级联清理向量 + MinIO + 图谱实体/关系） |
| `POST .../files/{fid}/reindex` | 单文件重新索引 |
| `GET /knowledge/bases/{id}/graph` | 知识图谱（实体 + 关系，`(name, source_file)` 去重） |
| `GET .../graph/neighbors?entity=&source_file=` | 实体邻居（点击详情，可限定文件命名空间） |
| `GET/POST .../graph/config` / `POST/DELETE .../graph` | 图谱构建配置 / 手动全库构建 / 重置 |
| `POST .../retrieval/test` | 检索测试工作台（basic 纯向量 / enhanced 五步流水线） |

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
| **多智能体** | DeepAgents 主 Agent 自主委派（task / spawn_tasks DAG）→ 子智能体并行执行 → 汇总；侧边任务面板展示进度与产出 |
| **知识库管理** | 多知识库隔离 · txt/md/pdf/docx/图片上传 · 多文件并行上传（独立进度）· 队列化索引（Redis Stream，重启不丢）· 扫描件 OCR · 本地/MinerU 双解析 · 文件预览 · 图谱级联删除 |
| **检索增强** | Milvus 向量检索 · 多策略分块 · BM25 混合 + 重排 · 知识图谱实体关系抽取与查询注入 |
| **Agent 工具** | web_search（Tavily）· calculator · datetime · text_tool · MCP 外部工具 |
| **Skill 系统** | SKILL.md 文件定义 + 渐进式披露（read_skill 激活）+ 双层工具门控 |
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
