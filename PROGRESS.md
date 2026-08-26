# EasyRAG → 企业级 Agent 平台 — 演进记录

> 参考项目：[Yuxi](https://github.com/xerrors/Yuxi) (语析) — 多租户 Agent Harness + 企业知识库平台
> 目标：从 EasyRAG 当前代码逐步重构为企业级多智能体平台

---

## 总体路线

```
阶段 1: 后端架构化   FastAPI分层 + PostgreSQL/Redis/MinIO + Repository模式  ✅ 完成
阶段 2: 知识库增强   多策略分块 + OCR链路 + 知识图谱 + 评估管线            ✅ 完成
阶段 3: Agent 体系    Sub-agent + MCP + Skill系统 + 中间件                    🚧 进行中（3A 完成）
阶段 4: 产品化       Vue3前端 + 多租户 + 管理后台                             🚧 前端先行
```

---

## 阶段 2（进行中）：知识库增强

### 2A 多策略分块 (2026-07-31) ✅

| # | 任务 | 状态 |
|---|---|---|
| 1 | 配置项 CHUNK_STRATEGY / PARENT_CHUNK_SIZE（config.py） | ✅ |
| 2 | recursive 递归分隔符切分（段落→句子→词，语义边界断开） | ✅ |
| 3 | markdown 结构感知切分（标题路径注入 chunk 前缀，代码块不拆，metadata 带 section_path） | ✅ |
| 4 | parent_child 父子分块（child 小块索引 + parent 大块上下文，metadata 带 parent_text） | ✅ |
| 5 | retriever 三后端统一 _unwrap_parent：命中 child 后返回去重后的 parent 上下文 | ✅ |
| 6 | 验证脚本 13/13 通过（4 策略 + 边界 + 报错路径） | ✅ |

用法：`.env` 设 `CHUNK_STRATEGY=fixed|recursive|markdown|parent_child`（默认 recursive）。上传 API 的 `strategy` 参数可单次覆盖。parent_child 模式下 `PARENT_CHUNK_SIZE`（默认 1500）控制上下文块大小。

### 2B OCR 链路 (2026-07-31) ✅

| # | 任务 | 状态 |
|---|---|---|
| 1 | app/rag/ocr.py — RapidOCR 本地引擎单例（onnxruntime，无需外部服务） | ✅ |
| 2 | 图片文件（.png/.jpg/.jpeg/.bmp/.webp）直接 OCR 提取文本 | ✅ |
| 3 | 扫描版 PDF 兜底：pypdf 平均每页 <20 字符判定无文字层 → pdfium 渲染 → 逐页 OCR | ✅ |
| 4 | chunker 接入：_EXTRACTORS 增加 5 种图片类型 | ✅ |
| 5 | 上传端点 ALLOWED_EXTENSIONS 放开图片；前端 accept 同步 | ✅ |
| 6 | 验证：真实 OCR 中文图片 + PIL 生成无文字层 PDF 全链路 4/4 通过 | ✅ |

新增依赖（stage1-agent 环境）：rapidocr 3.9.2、onnxruntime 1.28.0、pypdfium2 5.12.1、pypdf 6.14.2（顺带补齐缺失的 pypdf）。模型首次自动下载 PP-OCRv6（det/cls/rec 三个 onnx，存于 site-packages/rapidocr/models）。

### 2C 知识图谱 (2026-07-31) ✅

| # | 任务 | 状态 |
|---|---|---|
| 1 | PostgreSQL 两张新表：knowledge_entities / knowledge_relations（含 kb 外键级联） | ✅ |
| 2 | backend/services/graph_service.py — LLM JSON 模式逐 chunk 抽取实体/关系（失败 chunk 跳过） | ✅ |
| 3 | 图谱检索 query_related：query 关键词匹配实体 → 1 跳邻居关系 → prompt 文本注入 | ✅ |
| 4 | 上传链路接入：GRAPH_ENABLED 时索引后自动抽取（不阻塞主链路） | ✅ |
| 5 | 检索链路接入：knowledge_retrieval 节点注入子图为特殊 doc（metadata.graph=True） | ✅ |
| 6 | 查询端点 GET /knowledge/bases/{id}/graph（供前端可视化） | ✅ |
| 7 | 验证 6/6：schema 建表、子图查询、prompt 格式化、真实 LLM 抽取（10 实体 9 关系） | ✅ |

用法：`.env` 设 `GRAPH_ENABLED=true` 开启（默认 false，行为与阶段 1 一致）。成本控制：`GRAPH_MAX_CHUNKS_PER_FILE`（默认 30）限制单文件抽取 chunk 数。设计取舍：不引入 Neo4j，先用 PG 两表 + SQL 验证图谱价值；query_related 目前是跨知识库全局匹配，按会话锁定 kb 留待多 kb 路由完善。

### 2D 检索评估管线 (2026-07-31) ✅

| # | 任务 | 状态 |
|---|---|---|
| 1 | evaluation_runs 表：命名运行 + 聚合指标（hit_rate/mrr/avg_score）+ 逐条明细 JSON | ✅ |
| 2 | backend/services/evaluation_service.py — run_evaluation 逐条真实检索 + 指标聚合 | ✅ |
| 3 | evaluation_router.py：POST /evaluation/runs（执行+落库）、GET /runs（对比列表）、GET /runs/{id}（明细） | ✅ |
| 4 | 验证 9/9（打真实 API）：已知内容 query hit_rank=1、不存在文件不命中、两次命名运行可对比 | ✅ |

评估集格式：`[{"query": ..., "expected_source": 文件名}]`，expected_source 出现在 top_k 任一结果的 source 即算命中。指标：HitRate@k（命中率）、MRR@k（平均倒数排名）、avg_score。典型用法：切换 CHUNK_STRATEGY 重建索引后各跑一次同名评估，GET /runs 对比指标。

### Ollama 本地嵌入后端

| # | 任务 | 状态 | 日期 |
|---|---|---|---|
| 1 | OllamaEmbedder (POST /api/embed, 批量) | ✅ | 2026-07-31 |
| 2 | 配置项 (OLLAMA_BASE_URL/EMBED_MODEL/TIMEOUT) | ✅ | 2026-07-31 |
| 3 | EMBEDDING_TYPE 工厂路由 "ollama" | ✅ | 2026-07-31 |
| 4 | .env.template 更新 | ✅ | 2026-07-31 |
| 5 | 真实 Ollama 嵌入验证 (bge-m3, 1024维) | ✅ | 2026-07-31 |

用法：`.env` 里设 `EMBEDDING_TYPE=ollama`，默认连 `http://localhost:11434` 的 `bge-m3:latest`。维度 1024 与现有 Milvus collection 兼容，无需改 schema。

---

## 阶段 3（部分）：Agent 工具扩展

### 联网搜索工具 (Tavily)

| # | 任务 | 状态 | 日期 |
|---|---|---|---|
| 1 | web_search 工具 (Tavily REST API) | ✅ | 2026-07-31 |
| 2 | 配置项 + .env.template (TAVILY_API_KEY 等 4 项) | ✅ | 2026-07-31 |
| 3 | 工具注册 + 意图识别 prompt 更新 | ✅ | 2026-07-31 |
| 4 | tool_selection 关键词兜底推断 | ✅ | 2026-07-31 |
| 5 | 全链路验证 (registry/schema/graph compile) | ✅ | 2026-07-31 |
| 6 | 搜索来源追踪 (state.sources + 答案底部有序参考来源) | ✅ | 2026-07-31 |

调用链路：用户问"今天的新闻" → intent_recognition 判定 tool_use/web_search → tool_execution 调 Tavily API → answer_generation 基于搜索结果生成回答。未配置 TAVILY_API_KEY 时返回明确错误提示而非崩溃。

来源展示：web_search 输出尾部嵌入 `<!--SOURCES:[...]-->` 机器可读块，tool_execution 解析进 `state.sources`，answer_validation/fallback 统一在最终答案底部追加有序「参考来源」列表（Markdown 链接）。ChatResponse 同时返回结构化 `sources` 字段供前端使用。

### 3B MCP 外部工具接入（2026-08-07）✅

| # | 任务 | 状态 |
|---|---|---|
| 1 | MCP SDK 接入（stage1-agent 环境装 `mcp` 2.0.0） | ✅ |
| 2 | `app/tools/mcp/config.py` — `mcp_servers.json` 声明 server（name/transport/command 或 url/enabled/allowed_tools） | ✅ |
| 3 | `app/tools/mcp/manager.py` — MCPManager：每 server 独立常驻事件循环线程，list_tools 注册为 `ToolDefinition`（`mcp_<server>_<tool>` 前缀），同步 fn 经 `run_coroutine_threadsafe` 桥接异步 call_tool，start/stop/status 统一启停，stop 注销工具 | ✅ |
| 4 | `app/tools/mcp/demo_server.py` — 零依赖演示 server（stdio + HTTP 双模式，echo/get_time） | ✅ |
| 5 | `backend/server/routers/mcp_router.py` — GET /mcp/servers、POST /mcp/servers/{name}/start\|stop、GET /mcp/servers/{name}/tools；main.py lifespan 随应用启停 enabled server | ✅ |
| 6 | 权限两层：server 级 `allowed_tools` 白名单过滤 + Worker 侧 `tool_names` 白名单（既有机制） | ✅ |
| 7 | 验证：stdio + HTTP 双传输全链路（连接→list→注册→invoke→stop→注销）、权限过滤、TestClient 路由 200/404 | ✅ |

关键坑（踩过）：**async context manager 的 GC 陷阱** —— `stdio_client()` / `ClientSession()` 是 asynccontextmanager，若作为函数局部变量随返回被 GC，生成器收到 GeneratorExit，子进程关闭、流断开，后续调用报 `Connection closed`。必须把 context manager 引用保存在 handle 实例上（`_transport_cm` / `_session_cm`），stop 时显式 `__aexit__`。

用法：`mcp_servers.json` 登记 server，`GET /api/v1/mcp/servers` 查状态，`POST /api/v1/mcp/servers/{name}/start|stop` 启停。stdio 命令里的 `python` 会被替换为当前解释器（保证子进程有 mcp 包）。

---

## 阶段 4（前端先行）：Vue 3 前端

### 目标

在阶段 2/3 之前先搭建 Vue 3 前端，对接阶段 1 后端 API，提供可视化操作界面。

### 完成任务

| # | 任务 | 状态 | 日期 |
|---|---|---|---|
| 1 | Vue3 项目骨架 (Vite + Pinia + VueRouter + Axios) | ✅ | 2026-07-30 |
| 2 | 核心框架层 (router 守卫 + Pinia store + JWT 拦截器) | ✅ | 2026-07-30 |
| 3 | 登录/注册页面 (渐变背景居中卡片) | ✅ | 2026-07-30 |
| 4 | 主布局 (深色侧边栏 + 导航 + 登出) | ✅ | 2026-07-30 |
| 5 | 对话页面 (聊天气泡 + 打字动画 + Enter 发送) | ✅ | 2026-07-30 |
| 6 | 知识库页面 (卡片网格 + 文件表格 + 上传弹窗) | ✅ | 2026-07-30 |
| 7 | 全局样式 (Linear 风格，CSS 变量体系) | ✅ | 2026-07-30 |
| 8 | 代码完整性验证 (20/20 通过) | ✅ | 2026-07-30 |

### 前端文件清单 (14 个文件)

```
frontend/
├── package.json              # Vue 3.5 + Vite 6 + Pinia 2 + Axios
├── vite.config.js            # 开发代理 /api → :8000
├── index.html
└── src/
    ├── main.js               # createApp + Pinia + Router
    ├── App.vue               # <router-view />
    ├── style.css             # 全局样式 (Linear/Vercel 风格)
    ├── api/index.js          # Axios 封装 + JWT 自动附加 + 401 重定向
    ├── stores/auth.js        # Pinia 认证状态 (login/register/logout)
    ├── router/index.js       # 路由表 + beforeEach 守卫
    └── views/
        ├── LoginView.vue     # 登录页
        ├── RegisterView.vue  # 注册页
        ├── LayoutView.vue    # 主布局 (侧边栏 + router-view)
        ├── ChatView.vue      # 对话页 (聊天气泡 + Agent 调用)
        └── KnowledgeView.vue # 知识库管理 (CRUD + 文件上传)
```

### UI 重构：Yuxi 设计语言 (2026-07-31)

参考 [Yuxi 设计规范](https://github.com/xerrors/Yuxi/blob/main/docs/develop-guides/design.md) 全面重构前端 UI（克隆仓库提取 token 体系与布局模式）：

| # | 任务 | 状态 |
|---|---|---|
| 1 | 引入 lucide-vue-next 图标库，全面替换 emoji 图标 | ✅ |
| 2 | style.css 重写为 Yuxi token 体系（main 青色系 + gray 中性色阶、8px 圆角、阴影仅用于浮层） | ✅ |
| 3 | 侧边栏重构：顶部图标横排 → 纵向「图标+文字」导航，白底带边框「新建对话」主操作卡 | ✅ |
| 4 | 消息流重构：去头像，用户消息右侧浅青胶囊气泡（main-50），AI 回复无边框全宽纯文本 | ✅ |
| 5 | 输入框重构：全圆胶囊 → 12px 圆角卡片（灰边框 + 轻阴影），深色圆形发送按钮 | ✅ |
| 6 | 空状态去巨型 emoji + 40px 大标题 → Yuxi greeting（22px 克制标题 + 居中输入框） | ✅ |
| 7 | 知识库卡片去 hover 位移/重阴影，状态标签改语义浅底+深文字 | ✅ |
| 8 | 视觉验证：Edge headless 截图 smoke-chat / smoke-empty 双状态核对通过 | ✅ |

设计要点（Yuxi 规范）：主色 `--main-700 #046a82` 仅用于选中态/主操作；卡片 = 白底 + 1px `gray-150` 边框 + 8px 圆角、无装饰阴影；hover 只改背景/边框，禁用 transform 位移；滚动条 4px 细条。

### 修复：知识库上传文件不落库 (2026-07-31)

问题：前端上传走旧端点 `POST /kb/upload`，只做向量索引、不写 PostgreSQL 文件记录，导致 `GET /knowledge/bases/{id}/files` 永远返回空（页面显示「暂无文件」）。

修复：
1. 后端新增 `POST /api/v1/knowledge/bases/{kb_id}/upload`（knowledge_router.py）：解析分块 → 向量索引（metadata 带 knowledge_base_id）→ PostgreSQL 登记文件记录（pending → completed/failed）。
2. 前端 `KnowledgeView.vue` 上传改调新端点（携带当前知识库 id）。
3. 已上传但未落库的历史文件需重新上传才会出现在列表中。

### 特性：会话级删除（侧边栏「⋯」菜单）(2026-08-05)

| # | 任务 | 状态 |
|---|---|---|
| 1 | 后端 `DELETE /chat/conversations/{id}` — 归属校验 + 级联删除全部消息（FK ondelete=CASCADE） | ✅ |
| 2 | 前端侧边栏会话项 hover 显示「⋯」按钮 → 弹出菜单（生成摘要 / 删除） | ✅ |
| 3 | 删除确认弹窗（标题确认 + 不可恢复提示 + 删除中状态） | ✅ |
| 4 | 删除当前会话时跳转新对话状态；列表自动刷新 | ✅ |
| 5 | 验证脚本 9/9 通过（跨用户 404 / 级联删消息 / 重复删除 404） | ✅ |

原「生成摘要」行内按钮并入「⋯」菜单。DeepSeek 此前误实现为单条消息删除（DELETE /chat/messages/{id} + 气泡 hover 按钮），已全部回退替换。

---

## 阶段 1：后端架构化 ✅

### 目标

将 EasyRAG 从 Gradio 单脚本 + 薄 API 封装，升级为分层架构的企业级后端。

### 完成任务

| # | 任务 | 状态 | 日期 |
|---|---|---|---|
| 1 | PROGRESS.md + docker-compose 扩展 | ✅ | 2026-07-30 |
| 2 | PostgreSQL 数据模型层 (SQLAlchemy + 5 表) | ✅ | 2026-07-30 |
| 3 | Redis 缓存 + MinIO 存储层 | ✅ | 2026-07-30 |
| 4 | Repository 层 (Base + User + Conversation + Knowledge) | ✅ | 2026-07-30 |
| 5 | Service 层 (Auth + Chat + Knowledge) + 3 组 Router | ✅ | 2026-07-30 |
| 6 | JWT 认证中间件 + 种子脚本 | ✅ | 2026-07-30 |
| 7 | .env 配置 + 依赖安装 + 模块验证 | ✅ | 2026-07-30 |

### 新增文件清单 (20 个文件)

```
backend/
├── __init__.py
├── storage/
│   ├── __init__.py
│   ├── postgres/
│   │   ├── __init__.py
│   │   ├── manager.py              # SQLAlchemy async 引擎 + Base
│   │   ├── models_user.py          # Department / User
│   │   ├── models_conversation.py  # Conversation / Message
│   │   └── models_knowledge.py     # KnowledgeBase / KnowledgeFile
│   ├── redis/
│   │   ├── __init__.py
│   │   └── manager.py              # Redis async 客户端
│   └── minio/
│       ├── __init__.py
│       └── client.py               # MinIO 文件存储客户端
├── repositories/
│   ├── __init__.py
│   ├── base.py                     # BaseRepository[T] 泛型基类
│   ├── user_repository.py
│   ├── conversation_repository.py
│   └── knowledge_repository.py
├── services/
│   ├── __init__.py
│   ├── auth_service.py             # 注册/登录/JWT/密码哈希
│   ├── chat_service.py             # 对话 CRUD + Agent 调用
│   └── knowledge_service.py        # 知识库 CRUD + 文件管理
└── server/
    ├── __init__.py
    ├── main.py                     # 新 FastAPI 入口（整合新旧路由）
    ├── seed.py                     # 管理员种子脚本
    ├── routers/
    │   ├── __init__.py
    │   ├── auth_router.py          # POST /auth/register, /auth/login
    │   ├── chat_router.py          # POST /chat/send, GET /chat/conversations
    │   └── knowledge_router.py     # CRUD /knowledge/bases
    └── utils/
        ├── __init__.py
        └── auth_middleware.py       # get_current_user 依赖注入

新增根目录文件:
├── .env.template                   # 完整配置模板（含阶段 1 新增项）
├── .env                            # 从模板复制
├── requirements-stage1.txt         # 阶段 1 新增依赖
├── PROGRESS.md                     # 本文档
```

### 新增 API 端点

| 方法 | 路径 | 说明 | 认证 |
|---|---|---|---|
| POST | `/api/v1/auth/register` | 注册新用户 | 无 |
| POST | `/api/v1/auth/login` | 登录获取 Token | 无 |
| POST | `/api/v1/chat/send` | 发送消息（持久化） | JWT |
| GET | `/api/v1/chat/conversations` | 列出会话 | JWT |
| GET | `/api/v1/chat/conversations/{id}/history` | 获取对话历史 | JWT |
| POST | `/api/v1/knowledge/bases` | 创建知识库 | JWT |
| GET | `/api/v1/knowledge/bases` | 列出知识库 | JWT |
| GET | `/api/v1/knowledge/bases/{id}/files` | 列出文件 | JWT |
| — | 所有旧 `/api/v1/` 端点 | 保持兼容 | 原样 |

### 新增 Docker 服务

| 服务 | 端口 | 镜像 |
|---|---|---|
| postgres | 5432 | pgvector/pgvector:pg17 |
| redis | 6379 | redis:7-alpine |
| minio | 9090(console) / 9091(API) | minio/minio:latest |

### 数据库表结构

```
departments        — 部门/租户隔离
users              — 用户 (含 bcrypt 密码哈希、角色、部门)
conversations      — 会话 (关联用户)
messages           — 消息 (关联会话，含 metadata_json)
knowledge_bases    — 知识库 (关联用户/部门)
knowledge_files    — 文件记录 (关联知识库，含 MinIO 路径)
```

### 启动方式

```bash
# 1. 启动所有服务
docker compose up -d

# 2. 初始化数据库 + 管理员
python -m backend.server.seed

# 3. 启动 API
uvicorn backend.server.main:app --host 0.0.0.0 --port 8000 --reload

# 4. 登录
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

---

### 更新日志

#### 2026-07-30 — 阶段 1 全部完成

**变更内容：**
1. 创建 `PROGRESS.md` + `.env.template` + `requirements-stage1.txt`
2. `docker-compose.yml` 新增 postgres / redis / minio 3 个服务
3. `app/core/config.py` 新增 22 个配置项 (PostgreSQL/Redis/MinIO/JWT)
4. 创建 `backend/storage/` — PostgreSQL 5 表模型、Redis 客户端、MinIO 客户端
5. 创建 `backend/repositories/` — 泛型 BaseRepository + User/Conversation/Knowledge 仓库
6. 创建 `backend/services/` — JWT 认证(直接 bcrypt)、对话服务(含 Agent 调用)、知识库服务
7. 创建 `backend/server/` — 新 FastAPI 入口 + 3 组 Router + JWT 中间件 + 种子脚本
8. 安装全部阶段 1 依赖: sqlalchemy 2.0.36, asyncpg 0.30, redis 4.6, arq 0.26, minio 7.2, python-jose 3.3, python-multipart 0.0.18, bcrypt 5.0 等
9. 所有模块通过导入验证，auth_service 密码哈希/验证/Token 签发全部正常

#### 2026-07-31 — 对话 UI 模块化 + 知识库引用透出

**变更内容：**
1. 聊天界面模块化气泡：用户消息右侧青底胶囊气泡，AI 回复左侧浅灰卡片（细边框 + 大圆角），视觉区分清晰
2. 修复 LLM 回复排版错乱：移除 `.message-text` 的 `white-space: pre-wrap`（与 marked 输出的 `<p>/<br>` 冲突导致双倍空行），补全标题/嵌套列表/代码块/引用/表格等 markdown 样式
3. 知识库引用透出链路：`app/graph/nodes.py` 检索后从 retrieved_docs 提取去重来源生成 `kb_sources`（含知识库/图谱类型标记），`agent_service._build_response` 合并 kb_sources 与 web_search sources 返回
4. 引用持久化：`chat_router` 将 sources 存入消息 metadata_json，`chat_service.get_conversation_history` 解析 meta 返回，历史会话重载后引用块可还原
5. 前端 ChatView 新增「参考来源」引用块：知识库/图谱/网页三类标签 + 有序列表，web 来源可点击新窗口打开
6. 移除答案正文内嵌的 `_render_sources` 文本（避免与前端引用块重复），删除无用函数
7. 验证：后端 py_compile 通过；`npm run build` 通过；Edge headless 截图 + vision 核对气泡区分/排版/引用块三项全部正常

#### 2026-08-03 — 知识库文件预览

**变更内容：**
1. 上传流程增加 MinIO 存储：`upload_to_kb` 在登记文件记录后将原始字节 `put_object` 到 MinIO（路径 `kb/{kb_id}/{file_id}/{filename}`），写入 `minio_bucket`/`minio_object` 字段；MinIO 失败不阻塞主索引链路
2. 新增 `GET /api/v1/knowledge/bases/{kb_id}/files/{file_id}/preview` — 按文件类型返回文本/图片预览：
   - `content_type="text"`：txt/md/docx，复用 `chunker.extract_text` 统一提取（自动编码检测 utf-8/gbk/latin-1）
   - `content_type="pdf_text"`：PDF 提取文本（pypdf + 扫描版 OCR 兜底）
   - `content_type="image"`：图片，前端走 `/raw` 端点获取二进制
3. 新增 `GET /api/v1/knowledge/bases/{kb_id}/files/{file_id}/raw` — 返回原始文件二进制，带正确 Content-Type（图片 `image/*`，txt `text/plain; charset=utf-8`）
4. 前端 KnowledgeView：文件列表行可点击，弹出 720px 宽预览 modal
   - 图片：Blob URL 渲染 `<img>`（带 Authorization header 的 fetch）
   - 文本文件：等宽字体 `<pre>` 展示，PDF 额外标注「PDF 文本提取」badge
5. CSS：新增 `.file-row` 可点击样式（hover 主色高亮）、`.preview-modal` 弹性布局、`.preview-text` 代码块风格
6. **2026-08-03 修复** — KnowledgeFile 增加 `text_content` 列（TEXT），上传时调用 `extract_text` 存全文；预览端点优先读此列再回退 MinIO；无内容时提示「旧版本上传的文件请重新上传」
7. **2026-08-03 修复** — 只有点击文件名才弹出预览（`@click.stop` 在 `.file-name-cell`），整行不再触发；文件名 hover 显示主色下划线
8. **2026-08-03 修复** — PDF/图片预览改为原始格式：PDF 用 `<iframe>` 浏览器原生 PDF viewer 渲染（`raw` 端点返回 `application/pdf`，不强制 attachment），图片用 `<img>` Blob URL；txt/md/docx 保持文本提取预览
9. **2026-08-03 新增** — 文件删除功能：
   - 后端 `DELETE /api/v1/knowledge/bases/{kb_id}/files/{file_id}`：三层删除（向量索引按 source 匹配删除 → MinIO `remove_object` → PostgreSQL 记录删除），任一步失败不阻塞后续
   - retriever 新增 `delete_documents_by_source`：Milvus `delete(expr)` + Memory 数组过滤 + Chroma `where` 条件删除
   - 前端文件列表行尾加删除图标按钮（Trash2），点击弹出确认 modal，二次确认后调用 API 并刷新列表

#### 2026-08-03 — 知识库引用跳转 + SSE 流式输出

**背景：** 对话回复需要（a）底部引用可点击跳转到具体知识库文档详情；（b）思考回复时逐 token 流式输出而非一次性返回。

**Milvus schema 升级（方案 B，彻底改 schema）：**
1. `MilvusRetriever` collection 新增 `knowledge_base_id` 字段（VARCHAR 64）；`__init__` 检测到旧版 4 字段 schema 时自动 drop 并重建 5 字段 collection
2. `add_documents` 写入 kb_id（取自 metadata.knowledge_base_id，上传链路本已 setdefault 注入）；`retrieve` output_fields 带回 kb_id 并填入返回 metadata
3. 旧数据迁移脚本 `scripts/migrate_milvus_kb_id.py`：触发 schema 重建后，遍历 PostgreSQL `knowledge_files`（status=completed 且有 text_content），按当前 CHUNK_STRATEGY 重新分块并显式写入 kb_id 重建索引；188 个 chunk 全部回填 kb_id 成功

**引用透出 file_id：**
4. `app/graph/nodes.py` 新增 `_lookup_file_ids`（同步版）+ `lookup_file_ids_async`（async 版）：按 (knowledge_base_id, source) 批量反查 `knowledge_files.id`，`knowledge_retrieval` 节点给每条 kb_source 注入 `knowledge_base_id` + `file_id`

**SSE 流式输出：**
5. `LLMClient.chat_stream` — async generator，`stream=True` 逐 token yield 增量文本
6. `AgentService.prepare_context` — 复用 `knowledge_retrieval` 做检索 + 按 answer_generation 方式拼装 messages，返回 {messages, sources, intent}；同步阻塞，供 executor 调用
7. 后端新增 `POST /api/v1/chat/stream`（SSE）：事件序列 `conversation_id` → 多个 `delta` → `done`(含 sources/intent/elapsed)；检索走 `run_in_executor`，生成走 `chat_stream`，最终答案 + 引用落库与 `/chat/send` 一致
8. 前端 `api.streamChat` — fetch + ReadableStream 解析 text/event-stream（axios 不支持流式）；ChatView 改为流式渲染：先插空 assistant 消息，delta 逐步追加，「思考中…」仅在等待首个 token 时显示
9. 前端引用跳转：kb/图谱类型引用且有 file_id 时渲染为可点击链接，`goToSource` 路由跳转 `/knowledge?kb=..&file=..`；KnowledgeView 新增 `applyRouteQuery` 按 query 选中知识库并自动打开文件预览，watch route.query 支持页内再次点击

**修复的既有 bug：**
10. `LLMClient._call_kwargs` 用 `dict(kw=..., **extra)` 同名键报 `TypeError: multiple values`（标题生成传 temperature/max_tokens 时触发）→ 改为默认值做底 `merged.update(extra)` 覆盖
11. SSE 端点 executor 线程里 `asyncio.run()` 与主线程 async engine 事件循环冲突（`Future attached to a different loop`）导致 file_id 反查静默失败 → 拆出 `lookup_file_ids_async` 在端点主协程 await 回填

**验证：** `scripts/verify_chat_stream.py` 端到端通过 — 191 个 delta 事件流式到达，done 正常，sources 含正确 `file_id`（4c364425）+ `knowledge_base_id`（73a7f00f）；`vite build` 通过；后端无事件循环报错。

#### 2026-08-04 — 文件上传进度条（两阶段：传输 + 索引）

**背景：** 大文件上传时前端只有一个「上传中…」转圈，真正的耗时在后端解析/embedding/图谱阶段（embedding 是大头），用户无法判断是卡住还是在推进。

**方案 B（选中）：** HTTP 传输真实进度 + 后端索引进度轮询。

**后端：**
1. `knowledge_files` 表新增 `progress`（Integer 0-100，默认 0）+ `error_message`（Text，nullable），`ALTER TABLE ... IF NOT EXISTS` 迁移已执行
2. `knowledge_service.update_file_progress(session, file_id, progress, status?, error_message?)` — 每次调用独立 commit，供轮询读取
3. `POST /knowledge/bases/{id}/upload` 改异步：登记记录 + 存 MinIO 后立即返回 **202** + `status="processing"`；索引链路（解析分块 10% → 存全文 30% → 向量索引 80% → 图谱抽取 → 100%）移入 FastAPI `BackgroundTasks._run_ingestion`，每阶段独立 session 提交进度；失败置 `status=failed` + `error_message`
4. `FileResponse` 增加 `progress` / `error_message` 字段，前端按此渲染
5. **顺手修复既有 bug**：`OllamaEmbedder.embed_texts` 一次性把全部 chunk 塞给 `/api/embed`，大文件（数百 chunk）触发 400 → 改为分批 32 条/次请求（`_embed_batch`）

**前端：**
6. `api.upload(url, formData, onUploadProgress)` 支持 axios 原生传输进度回调
7. KnowledgeView 上传弹窗加进度条：`uploadPhase` 状态机（transferring → indexing → done/failed），传输阶段显示真实百分比；索引进 `indexing` 后每 1.5s 轮询 `/files` 接口按 `progress` 渲染，轮询同时刷新文件列表（关掉弹窗后台继续，列表里 status-badge 仍可见）
8. 传输阶段禁止关弹窗（请求会断）；索引阶段可「后台继续」；进度未定态（progress=0）时进度条走呼吸滑动动画；失败时进度条变红并显示 `error_message`
9. 样式沿用 Airy 变量（--main-500 / --gray-100 / --radius-full），`progress-track` 6px 圆角条

**验证：**
- 1.5MB 测试文件（1173 chunks）：202 立即返回 → 轮询观测 30% → 80% →（图谱阶段 LLM 503 重试约 4 分钟）→ completed 100%，chunk_count=1173 全部入 Milvus
- 小文件（8 chunks）状态机：202 in 2.2s → 30% → 80% → 100%，progress 序列单调递增
- `vite build` 通过（11.13s）

**2026-08-04 追加修复 — upload 接口 PendingRollbackError 500：**
- 现象：MinIO 不可用时上传直接 500，前端显示「上传失败」且后台任务未启动，DB 无文件记录
- 根因：`upload_to_kb` 中 MinIO `put_object` 异常被 except 捕获后未 `session.rollback()`，session 进入 PendingRollback 状态；后续 `record.id` 访问触发 expired 属性重载 → `PendingRollbackError` 抛出 500；嵌套原始异常为 MinIO 失败时 record 的 minio 字段 UPDATE 匹配 0 行
- 修复：commit 后立即取出 `file_id/kb_uuid/filename` 纯值（不再依赖 ORM 属性）；MinIO 失败分支显式 `await session.rollback()`；`put_object` 成功后先 `session.refresh(record)` 再更新 minio 字段走独立短事务
- 验证：停掉 easyrag-minio 容器上传 → 202 返回、索引 completed；恢复 MinIO 上传 → 202、completed

#### 2026-08-04 — 消息丢失根因修复：executor 线程 DB 连接池污染

**现象：** 用户提问后回答"不见了"——DB 里只有 user 消息，assistant 消息一条都没有；`/chat/send` 间歇 500（FK violation: conversation_id 不存在）。

**根因（区别于此前修的同类 bug 的残留）：** `app/graph/nodes.py` 的 `_lookup_file_ids`（:169）和 `knowledge_retrieval` 内的 `_graph_query`（:208）在 FastAPI `run_in_executor` 的 worker 线程里用 `asyncio.run(...)` 执行协程，协程复用**全局 async engine 的连接池**。连接带着另一个事件循环的 Future 归还池中 → 后续请求拿到毒连接，第一个事务（创建会话+用户消息）静默失效/回滚，第二个独立 session 插 assistant 消息时触发 `ForeignKeyViolationError`。日志中的 `attached to a different loop` 即为污染现场。

**修复：** nodes.py 新增 `_run_with_isolated_engine` / `_run_in_thread_isolated` 封装——executor 线程内的 DB 查询一律用**随用随建、用完 dispose 的独立 engine**（pool_size=1），与主 loop 连接池完全隔离；两处 `asyncio.run(get_session())` 全部改走该封装。

**验证：** 端到端 4 场景全过——① /chat/send 带检索（触发 executor 线程 DB 访问）user+assistant 均落库；② 同会话第二轮累积到 4 条；③ /chat/stream 19 个 delta + done，落库完整；④ stream 后再 send（污染回归检查）200 + 落库正常。测试数据已清理。

#### 2026-08-04 — 流式路径意图识别修复

**现象：** 前端（/chat/stream SSE）发送任何内容，意图识别都返回 knowledge_qa——问候、计算、天气全部被强行走向量检索，回答硬扯民法典。

**根因：** `agent_service.prepare_context` 硬编码 `intent="knowledge_qa"`（注释写明"流式路径不做完整意图识别"），从未调用 `intent_recognition` 节点。LLM 分类器本身是正常的（实测 6 类查询全对），问题只在流式路径绕过了它。

**修复：** `prepare_context` 接入完整意图分流，复用现有 LangGraph 节点：
1. `intent_recognition` 分类（失败 fallback knowledge_qa，与完整路径一致）
2. `requires_tool` 或 tool_use → `tool_selection` + `tool_execution`（web_search/calculator/datetime_tool），工具结果注入上下文，web_search 引用合并进 sources
3. `requires_retrieval` 或 knowledge_qa/complex_task → `knowledge_retrieval` 向量检索
4. chitchat（requires_retrieval=False）跳过检索直接对话
返回值新增 `tool_result` 字段，intent 从硬编码改为真实分类结果。

**验证：** 4/4 端到端通过——chitchat(sources=0 正常自我介绍) / tool_use(1+1 calculator) / knowledge_qa(民法典检索+引用) / tool_use(天气 web_search 分流正确, Tavily key 未配走兜底)。

#### 2026-08-05 — 阶段 1：Agent 内核重构（静态 DAG → 可编排 Agent）

**背景：** 此前 workflow.py 是写死的 LangGraph 静态 DAG（意图→检索/工具→生成→校验），所有请求走同一路径，Agent 只是 RAG 的附属品。本阶段升级为可编排 Agent 内核。Spec: `docs/specs/2026-08-04-agent-core-react-design.md`，Plan: `docs/plans/2026-08-04-agent-core-react.md`。

**关键决策（brainstorming 澄清）：** ① 保留 LangGraph，StateGraph 改循环图（不重写调度器）；② ReAct 与快速路径并存（简单问答走快路径，复杂任务进 ReAct）；③ 先用现有模型，留分级接口。

**A. 工具插件化（Task 1, b6c0e18）：**
- `ToolDefinition` 加 `check_fn` 可用性自检；`list_names/to_llm_schema/to_react_prompt` 只含可用工具，invoke 拦截不可用
- `discover_tools()` 扫 `app/tools/` 自动注册导出 `TOOL` 的模块——新工具放模块即插即用，替换硬编码注册
- 4 个现有工具改造为插件格式；web_search 的 check_fn 检查 TAVILY_API_KEY

**B. ReAct 循环子图（Task 2, 3123dde）：**
- 新增 `agent_reasoning` 节点：LLM 每轮读 query+history+observations+工具描述，输出 JSON 决定 action（tool 调用 / final_answer），实现真正的思考→行动→观察→再思考
- `tool_execution` 支持 ReAct 分支：从 pending_tool 取工具，结果追加到 observations，循环回 agent_reasoning；`_retry` 标记处理推理失败自我修正（连续 3 次→fallback）；步数耗尽（AGENT_MAX_ITERATIONS）强制回答
- 分流：complex_task 或置信度<0.6 → use_react → agent_reasoning；其余走现有快速路径
- 新增 REACT_REASONING prompt、route_after_reasoning 路由、AgentState 加 observations/pending_tool/use_react/react_iterations

**C. 结构化记忆（Task 3, a05ab5c）：**
- 工作记忆：现有 AgentState（不动）
- 情景记忆：conversations.summary 字段，每 10 轮 LLM 增量压缩；`get_compressed_history` 用「摘要+最近10轮」替代全部历史（仅长会话生效，短会话不超窗口原样返回）
- 语义记忆：新增 user_facts 表 + app/memory/manager；规则触发（记住/我喜欢/我是等关键词）LLM 提取事实存储；prepare_context 注入 system prompt（executor 线程走 _run_in_thread_isolated 独立 engine 防连接池污染）

**D. 模型分级接口（Task 4, 1542904）：**
- config 加 LLM_FAST_BASE_URL/API_KEY/MODEL（Optional，默认 None）
- `get_llm_client(tier="main"|"fast")`：fast 未配置回退主模型；配置后独立 fast client（per-tier 单例缓存）。本期所有调用点仍用 main，接口留好。

**验证：** 4 个新 verify 脚本全绿（tool_plugin 15/15、react_loop 19/19、memory_layers 14/14、model_tiers 13/13）；message_persistence 15/16（唯一 FAIL 是清理检查——DB 有 6 条旧真实会话数据不该删，功能 15 项全过）；快速路径 4 类意图回归通过。

#### 2026-08-05 — 阶段 3A：Orchestrator-Worker 骨架 + 共享黑板（M1+M2）

**M1 Orchestrator-Worker 骨架：**

| # | 任务 | 状态 |
|---|---|---|
| 1 | `app/agents/` 目录结构 + `__init__.py` | ✅ |
| 2 | `workers/base.py` — TaskBrief / WorkerReport / BaseWorker（name/persona/工具白名单/run 接口） | ✅ |
| 3 | `rag_worker.py` — 复用现有 retriever，白名单 [web_search]，lazy retriever 防模块导入时连接 Milvus | ✅ |
| 4 | `legal_worker.py` — 法律人格 prompt，白名单 [web_search] | ✅ |
| 5 | `code_worker.py` — 代码人格 prompt，白名单 [calculator, text_tool]，提取 code_snippets | ✅ |
| 6 | `orchestrator.py` — LLM 拆解（TaskBrief JSON）+ 派发 + 汇总 + Worker 注册表 | ✅ |
| 7 | `config.py` + `agent_service.py` — AGENT_MODE 开关 + multi 分支 + 崩溃回退 single | ✅ |
| 8 | `verify/verify_multi_agent.py` — 28/28 通过（契约 + mock 全流程 + 真实 LLM 结构断言 + single 回归） | ✅ |

**M2 共享黑板（Blackboard）：**

| # | 任务 | 状态 |
|---|---|---|
| 9 | `app/agents/blackboard.py` — Blackboard 类（threading.Lock 全方法保护） | ✅ |
| 10 | `base.py` + `orchestrator.py` — `run_with_board` 自动上板 / `_resolve_refs` task-N 引用解析 / `ThreadPoolExecutor` parallel 真并发 | ✅ |
| 11 | `verify/verify_blackboard.py` — 24/24 通过（黑板单测 + 8 线程×25 post 并发安全 + sequential/parallel 集成 + 崩溃隔离 + 真实 LLM 黑板透出） | ✅ |

**关键设计：**
- **特性开关**：`.env` 设 `AGENT_MODE=single|multi`（默认 single，行为完全不变）；multi 分支 try/except 包住，orchestrator 崩溃回退 single
- **数据契约**：`TaskBrief`（task_id/goal/context/constraints/worker_hint）→ `WorkerReport`（status/summary/detail/artifacts/steps/error）
- **Worker 基类**：name + persona（system prompt）+ tool_names 白名单；`invoke_tool` 越权抛 PermissionError；LLM client lazy property 可注入 mock
- **Orchestrator 循环**：`_decompose`（LLM 输出 JSON：needs_decomposition / sub_tasks[] / execution_mode / final_instruction）→ `_dispatch`（sequential/parallel）→ `_synthesize`（单任务直返 / 多任务 LLM 整合，失败回退原始拼接）
- **黑板**：`post_artifact` / `read_artifact` / `find_by_tag` / `find_by_task` / `render_for_prompt`（排除自身任务防自引用）；`run_with_board` 包装 `run()` 自动上板；`_resolve_refs` 解析 brief.context 中的 task-N 引用注入前序产出
- **并发**：parallel 模式 `ThreadPoolExecutor` + `as_completed`，收集后按原任务序 sort 恢复；黑板锁保护多线程读写

**验证：**
- `verify/verify_multi_agent.py` 28/28（mock 全流程 + 真实 LLM 结构断言 + single 回归）
- `verify/verify_blackboard.py` 24/24（黑板单测 + 并发安全 + sequential/parallel 集成 + 崩溃隔离 + 真实 LLM 黑板透出）
- 真实 LLM 路径硬断言结构（blackboard/execution_mode/steps），软断言内容——LLM 端点波动导致空回答时打 WARN 不 FAIL

#### 2026-08-09 — 安全修复 1：向量检索按用户/知识库隔离

**问题：** PostgreSQL 中的知识库记录虽然带 `owner_id`，向量数据也保存了
`knowledge_base_id`，但 Milvus、Memory、Chroma 和增强检索实际查询时没有使用该字段。
所有用户共享同一集合，因此登录用户可能检索到其他用户的文档；BM25、图谱缓存和
多 Agent 的 RAG Worker 也存在同样的越权路径。

**设计决策：** 采用“显式授权作用域 + 默认拒绝”。检索接口统一接收
`knowledge_base_ids`；没有授权 ID 时直接返回空结果，不再把 `None` 解释为全库。
过滤在候选召回前执行，避免其他租户的结果挤占当前用户 Top-K，并在结果返回前再次
校验作用域作为纵深防御。

**实现：**
1. `backend/server/routers/chat_router.py` 从数据库查询当前用户拥有的全部知识库 ID，
   分别注入同步聊天、SSE、单 Agent 和多 Agent 路径；新增
   `KnowledgeBaseRepository.list_ids_by_owner()`，不复用 UI 的 50 条分页限制。
2. `AgentState`、`AgentService`、`Orchestrator`、`TaskBrief` 和 `RagWorker` 全链路传递
   授权知识库 ID；未认证/未传作用域的旧调用默认无法读取任何向量文档。
3. Milvus 使用 `knowledge_base_id in [...]` 搜索表达式；Chroma 使用 metadata `where`；
   Memory 在计算 Top-K 前筛选候选。三个后端都在返回阶段再次丢弃越界结果，作用域 ID
   先做 UUID 规范化，避免表达式注入。
4. 增强检索的语义、BM25、实体、关系和迭代补充路径全部携带同一作用域；BM25 同步
   索引保留 `knowledge_base_id` 元数据。
5. 图谱缓存以 `(knowledge_base_id, entity_name)` 作为实体键，关系也记录 `kb_id`，解决
   不同知识库同名实体描述被合并的问题；旧版全局 PostgreSQL 图谱旁路改为只查询授权 ID。
6. 上传索引时强制覆盖 chunk 的 `knowledge_base_id`，不允许上游元数据覆盖授权归属；
   增强检索引用的 `file_id` 改为按 `(knowledge_base_id, filename)` 查询，避免同名文件串租户。

**验证：**
- 新增 `tests/test_retrieval_isolation.py`：覆盖默认拒绝、无效 UUID、Memory、Milvus、
  Chroma、BM25、同名图谱实体、LangGraph 节点和 RAG Worker 作用域传递；
  `python -m pytest -q` 为 **9/9 通过**。
- `verify/verify_auto_route.py` 为 **12/12 通过**。
- `app/`、`backend/`、`tests/` 共 78 个 Python 文件 AST 语法检查通过。

#### 2026-08-09 — 知识库目录上下文：Agent 可读取知识库名与文件名

**问题：** 对话链路此前只向检索器传递知识库 ID，生成模型只能看到命中的 chunk 和
引用来源，看不到完整的知识库名称与文件目录。因此询问“当前知识库有什么文件”时，
即使命中某个文件，Agent 也只能承认无法确定完整清单。

**设计决策：** 增加“授权目录上下文”，不依赖向量命中。目录与检索 ID 由同一条
`owner_id` 限定查询生成，确保两者作用域一致；只读取知识库名、文件名、类型与处理状态，
不加载文件正文。目录元数据按不可信输入处理：清除控制字符、转义标签、逐项限长、整体
限制 12000 字符，并在 system prompt 中明确名称只能作为数据、不得作为指令执行。

**实现：**
1. `KnowledgeBaseRepository.list_catalog_by_owner()` 使用 owner-scoped 外连接一次读取当前
   用户的知识库和文件元数据；不使用 UI 分页，也不查询 `text_content`。
2. `/chat/send` 与 `/chat/stream` 通过 `_load_knowledge_scope()` 从同一目录同时取得
   `knowledge_base_ids` 和 `knowledge_catalog`，同步、SSE 快速路径、多 Agent 及直接 LLM
   兜底均注入目录。
3. `AgentState`、`AgentService`、`Orchestrator`、`TaskBrief` 全链路新增
   `knowledge_catalog`；单 Agent `answer_generation`、SSE `prepare_context` 和多 Agent
   `RagWorker` 都能读取相同目录。
4. 新增 `app/services/knowledge_catalog.py`，负责把授权目录格式化为紧凑、安全、可截断的
   system context；空目录明确告知 Agent 当前用户没有可访问的知识库。

**验证：**
- 新增 `tests/test_knowledge_catalog.py`，覆盖名称/文件名可见、文件状态可见、控制字符与
  标签转义、目录长度上限、仓储 owner 条件、单 Agent、SSE 和多 Agent 目录注入。
- `python -m pytest -q`：**15/15 通过**（含上一项检索隔离回归）。
- `python -m compileall -q app backend tests` 与 `git diff --check` 通过。

#### 2026-08-09 — 知识库管理工作台（前端阶段）

**目标：** 参考知识库管理产品的详情工作台，在不新增后端接口的前提下，先完成知识库
目录、文件管理以及检索/图谱/评估模块的前端信息架构和交互状态，供后续逐项接入后端。

**实现：**
1. `KnowledgeView.vue` 重构为“知识库目录 → 知识库详情工作台”两层页面；目录页新增账号级
   概览、搜索、错误/空结果状态和更完整的知识库卡片，详情页支持复制 ID、返回目录以及
   URL query 保存当前知识库与页签。
2. 详情工作台新增六个页签：文件管理、检索测试、知识图谱、知识导图、RAG 评估、评估基准。
   文件管理继续复用现有真实接口，保留上传与索引进度、文件刷新、预览和删除；统计卡片由
   当前文件列表实时计算。
3. 新模块只实现前端契约和状态，不伪造后端结果：检索测试展示动态请求参数与结果空态；
   图谱、导图使用真实知识库名和文件名生成目录级预览；RAG 评估实现未运行状态、指标卡、
   历史表格空态和配置弹窗；评估基准实现指标开关与数据集空态。所有未接能力均明确标记
   “接口待接入”，点击操作会给出前端提示。
4. 新增 `frontend/src/styles/knowledge-workspace.css`，提供独立的黑白灰工作台视觉、表格、
   图谱画布、导图、评估卡片、弹窗与 1180/900/680px 响应式布局；未改动现有后端协议。
5. 改善上传弹窗关闭行为：文件传输阶段禁止中断，进入索引阶段后可关闭弹窗并让前端继续
   轮询，避免原实现关闭弹窗时停止轮询却留下未完成 Promise。

**验证：**
- `cd frontend && npm run build` 通过（Vite，1808 modules transformed）。

- 使用本地只读模拟响应在 1440×1000 视口检查文件管理与 RAG 评估页：目录数据、页签路由、
  统计卡片、文件表格、处理中状态和评估空态均正常渲染。

#### 2026-08-10 — 对话页多模型切换

**目标：** 在对话输入区提供模型切换，并把 MiniMax-M2.7、DeepSeek-V4-Flash、
Qwen-3.6-Flash、GLM-5.2 接入后端环境配置；同一次请求中的单 Agent、多 Agent 子任务与
最终流式汇总必须使用同一模型，API Key 和供应商地址不得暴露给浏览器。

**设计决策：** 使用“后端模型目录 + 公开 model_id 白名单”。前端只从
`GET /api/v1/chat/models` 读取名称、供应商和可用状态，发送公开 `model_id`；后端再解析为
真实 base URL、API 模型名、温度与密钥。未知模型和未配置模型在消息写库前直接拒绝，避免
客户端注入任意上游地址或模型。多个模型使用同一 OpenAI 兼容网关时，各模型 BASE_URL 可
设为 `LLM_BASE_URL` 并安全复用现有 `DEEPSEEK_API_KEY`；直连官方接口则使用独立密钥。

**实现：**
1. 新增 `app/llm/models.py`，定义四个模型的服务端目录、公开元数据、默认模型、配置可用性
   检查和 allowlist 解析；`app/core/config.py`、`.env` 与 `.env.template` 同步供应商地址、
   API 模型 ID、温度和密钥变量，当前本地 `.env` 四个模型统一复用既有网关。
2. `GET /chat/models` 返回不含 base URL、API Key 和真实内部配置的安全目录；`ChatRequest`
   新增 `model_id`，同步与 SSE 接口均在持久化用户消息前验证选择，助手消息 metadata 记录
   `model_id/model_name`，便于历史追溯。
3. `app/llm/client.py` 新增请求上下文模型选择与按模型缓存；Agent 图节点、Orchestrator、
   Worker、增强检索和流式汇总都从当前请求解析客户端。针对 `ThreadPoolExecutor` 不自动
   传播 contextvars 的行为，多 Agent 并行子任务会显式重新进入同一模型上下文。
4. `ChatView.vue` 在输入框左下角新增模型选择器：记忆上次选择、加载后端默认值、发送期间
   锁定、未配置项禁用；每条回复显示实际模型名。SSE 错误解析也改为优先显示后端 detail。
5. README 补充模型配置与切换说明；`requirements-stage1.txt` 补齐此前遗漏的 FastAPI 与
   Uvicorn 运行依赖声明。

**验证：**
- 新增 `tests/test_chat_model_selection.py`，覆盖四模型目录、敏感配置不外泄、共享网关密钥
  复用、直连缺密钥禁用、未知模型拒绝、请求上下文恢复和并行 Worker 模型传播。
- `python -m pytest -q`：**21/21 通过**。
- `python -m compileall -q app backend tests`、`git diff --check` 通过。
- `cd frontend && npm run build` 通过（Vite，1808 modules transformed）。

- 安装 `requirements-stage1.txt` 后成功加载 FastAPI 应用，并确认
  `GET /api/v1/chat/models` 路由及响应模型已注册。

#### 2026-08-10 — v0.3.1：自定义本地/云端模型配置

**目标：** 全局顶部展示当前应用版本 v0.3.1；对话模型菜单移除“暂无可用模型”占位，新增
“添加自定义模型”，允许用户配置本地或云端 OpenAI 兼容模型，并在保存后立即进入当前账号
的可选模型目录。

**设计决策：** 采用“环境变量内置模型 + PostgreSQL 动态模型”的混合配置。部署者维护的
四个内置模型继续由 `.env` 提供；用户在前端新增的模型按 `owner_id` 存入数据库，避免运行时
修改进程环境或配置文件。动态 API Key 使用 Fernet 加密，密钥由
`MODEL_CONFIG_ENCRYPTION_KEY` 提供，开发环境未配置时回退 `JWT_SECRET_KEY`；列表接口只返回
公开 ID、显示名、供应商、来源、类型和可用状态，不返回真实地址、模型内部参数或密钥。

**实现：**
1. 应用默认版本、`.env`、`.env.template`、前端包版本统一升级为 `0.3.1`；`LayoutView.vue`
   新增全局顶部条，从 `/api/v1/health` 同步后端版本并保留 `v0.3.1` 构建兜底。
2. 新增 `custom_model_configs` 表、`CustomModelConfigRepository` 和模型配置服务；配置包含
   owner、显示名、模型类型、Base URL、模型 ID、加密 API Key、温度与是否需要密钥。
3. 新增 `POST /chat/models` 与 `DELETE /chat/models/{model_id}`；`GET /chat/models` 合并环境
   内置模型与当前用户自定义模型。自定义模型使用 `custom:<uuid>` 公共 ID，所有读写和对话
   解析都带 owner 条件，其他用户无法枚举、使用或删除。
4. 本地模型允许无 API Key，支持 localhost、私有网段、`.local` 和 Docker 服务名；云端
   模型强制 HTTPS，并拒绝字面量本机/私有 IP。URL 禁止嵌入账号、查询参数和 fragment，降低
   错配与 SSRF 风险。
5. LLM 请求上下文从“仅模型 ID”升级为携带完整、已授权的运行时 profile；因此数据库模型
   也能正确传播到单 Agent、多 Agent 并行 Worker、增强检索和 SSE 最终汇总。客户端缓存键
   加入配置指纹，删除后重建同名模型不会误用旧连接。
6. 原生 select 改为可控模型菜单，始终提供“添加自定义模型”；新增本地/云端类型卡、服务
   地址、模型 ID、温度、API Key 与已添加模型管理弹窗。保存成功后重新拉取后端目录并自动
   选中新模型，删除操作只允许自定义项且需要二次确认。

**验证：**
- 新增 `tests/test_custom_model_config.py`，覆盖密钥密文存储/解密、本地无密钥模型、公开响应
  不泄露密钥和地址、本地/云端 URL 安全边界、动态 profile 请求上下文及仓储 owner 条件。
- `python -m pytest -q`：**26/26 通过**。
- FastAPI 应用加载成功，确认 `GET/POST /api/v1/chat/models`、
  `DELETE /api/v1/chat/models/{model_id}` 已注册，应用版本为 `0.3.1`。
- `cd frontend && npm run build` 通过（Vite，1808 modules transformed）。

#### 2026-08-15 — Skill 配置管理与对话显式选择

**目标：** 在对话输入框中显式选择本次请求使用的 Skill，并提供可视化配置入口；先内置一组
可直接使用的基础 Skill，同时允许每个用户创建、修改和删除自己的 Skill。Skill 不仅影响
提示词，还必须在后端约束本次请求可以调用的工具，避免仅靠前端隐藏形成伪权限。

**设计决策：** 采用“代码内置 Skill + PostgreSQL 用户 Skill”的混合目录。内置 Skill 随
版本发布，保证开箱即用；自定义 Skill 按 `owner_id` 持久化，不修改 `.env`。每条消息最多
选择 3 个 Skill，后端根据已授权的 ID 解析运行时上下文，将多项指令合并，并对工具白名单
取并集。未选择 Skill 时保持原有 Agent 自动路由行为，避免升级后改变既有对话语义。

**实现：**
1. 新增 Skill 目录与请求级上下文，预置知识库研究、联网研究、数据分析、专业写作和法律分析
   5 项基础 Skill；每项均声明用途说明、详细指令与允许工具。
2. 新增 `custom_skill_configs` 表、仓储和服务层；自定义项支持名称、分类、描述、指令、图标、
   启用状态与工具白名单，所有查询和写操作均附带当前用户 owner 条件。
3. 新增 `GET/POST /chat/skills`、`PUT/DELETE /chat/skills/{skill_id}`；对话请求新增
   `skill_ids`，同步及 SSE 路径都会在消息写库前校验 Skill，并把名称快照写入消息 metadata。
4. Skill 指令已贯穿意图识别、LangGraph 节点、单 Agent、多 Agent Worker、任务分解、并行
   执行和最终汇总；并行线程会显式传播 Skill 上下文，避免 `contextvars` 在线程池中丢失。
5. 工具注册表在执行层按当前 Skill 白名单过滤和拒绝调用；多选 Skill 使用权限并集。未选择
   Skill 时继续提供原有工具集合，保持向后兼容。
6. `ChatView.vue` 新增可搜索的 Skill 菜单、已选标签、最多 3 项限制、浏览器选择记忆，以及
   Skill 配置弹窗。弹窗可查看基础库、复制内置 Skill、创建/编辑/删除自定义 Skill，并逐项
   配置工具权限；消息区同步展示实际使用的 Skill。
7. README 补充用户配置流程与 API 说明，设计细节记录于
   `docs/plans/2026-08-15-skill-configuration-design.md`。

**验证：**
- `python -m pytest -q tests/test_skill_configuration.py tests/test_chat_model_selection.py tests/test_custom_model_config.py`：**17/17 通过**。
- 完整测试集：**74 passed, 28 skipped**；跳过项为当前测试解释器未安装 `pytest-asyncio` 的
  既有异步用例，不包含本次 Skill 测试。
- `python -m compileall -q app backend tests` 通过。
- `cd frontend && npm run build` 通过（Vite，1808 modules transformed）。
- 当前开发数据库已增量创建 `custom_skill_configs`；针对运行中的 v0.3.1 后端完成临时账号
  冒烟测试，5 项内置 Skill 列表与自定义 Skill 创建、查询、修改、删除全部通过，测试数据已清理。

#### 2026-08-15 — 状态栏展开按钮固定悬浮

**调整：** 对话页在存在任务且状态栏收起时，将“展开状态栏”从消息流顶部改为固定在视口
右侧垂直居中的悬浮按钮。消息滚动不再影响入口位置；面板展开后入口仍按原逻辑隐藏，并补充
键盘焦点、悬浮反馈、进入动画和移动端右侧间距。

**动效优化：** 状态栏改用 Vue Transition 抽屉过渡，展开和收起时在约 520ms 内同步改变
宽度、右侧位移、透明度、内边距及边框，聊天主区域随之平滑伸缩；悬浮入口延迟淡入，避免与
正在收起的面板重叠。拖拽改宽期间禁用过渡，并兼容系统“减少动态效果”设置。

#### 2026-08-15 — 简单问题快速路径：不再"思考很久"也不误调 web_search

**背景：** 用户反馈"我在一家餐馆吃坏肚子了，该怎么办"这类简单常识问题也被意图分类器判成
"联网/工具查询（置信度 75%）"并实际调用 web_search——思考链路长（理解→识别→调工具→生成）、
开销大且不合理。

**方案（两层，互不依赖）：**

1. **规则快速路径（新增 `app/graph/fast_intent.py`）** — 在 LLM 意图分类之前做零成本规则预判：
   - 明确计算请求（"1+1"、"帮我计算 2+2"）→ `tool_use + calculator`（跳过 LLM 分类）
   - 明确日期时间请求（"现在几点"、"今天几号"）→ `tool_use + datetime_tool`（跳过 LLM 分类）
   - 问候闲聊（"你好"、"谢谢"）→ `chitchat`（跳过 LLM 分类）
   - 简单自包含常识/生活/写作问题（"吃坏肚子怎么办"、"如何做红烧肉"）→ 新增 `direct` 意图，
     直接回答：不检索知识库、不调工具、不经过 LLM 分类
   - 其余（涉及实时数据/知识库关键词、依赖上文的追问、长问题、用户显式启用 Skill）→ 返回
     `None`，交回原 LLM 分类器。设计原则：宁可漏（回退 LLM），不可错。
2. **LLM 分类器 prompt 修正（`INTENT_RECOGNITION`）** — 新增 `direct` 意图定义与关键规则：
   常识/生活建议/健康/科普/做法/写作类问题一律判 `direct`（requires_retrieval=false、
   requires_tool=false），不要因为"可能查得到"就调 web_search；只有天气/新闻/股价/汇率/
   当前时间等真正实时数据才走 web_search。即使绕过规则层，LLM 也不再误判。

**配套改动：**
- `intent_recognition` 节点接入快速路径；用户显式启用 Skill 时跳过快速路径（尊重 Skill 工具约束）。
- `route_after_intent` 显式支持 `direct` → 直接生成；`route_after_generation` 对 `direct`
  跳过校验环节（省一次 LLM 调用），workflow 条件边补充 END。
- 意图中文标签与前端思考过程时间线新增 `direct`（直接回答）步骤（`AgentActivity.vue`）。
- 新增配置 `FAST_INTENT_ENABLED=true`（默认开启，关闭即回退旧行为），`.env.template` 同步。

**效果：** 简单问题从"4 步 + web_search 外部调用"降为"理解问题 → 识别意图（直接回答）→ 生成回答"，
全程 1 次 LLM 调用、0 次外部工具。

**验证：**
- 新增 `tests/test_fast_intent.py` **24/24 通过**（常识→direct、计算/时间→工具、问候→chitchat、
  天气/新闻/知识库/追问→安全回退、长问题/空串→回退）。
- 完整测试集 `python -m pytest -q`：除既有的 2 个图谱构建用例（jieba/加速缓存，与本次无关）
  外全部通过；`python -m compileall -q app backend tests` 通过。
- 真实节点验证：`intent_recognition("吃坏肚子怎么办")` → direct（retrieval/tool 均 false）；
  Skill 激活时走 LLM 分支且新 prompt 同样判 direct（此前为 web_search）；图编译通过。

#### 2026-08-15 — 上下文管理 P0 修复：摘要不丢段 + 历史窗口显式化 + 事实提取后台化

**背景（评审结论）：** 上下文管理存在三类高优先级问题：① 摘要折叠只看"最后 10 条"，
某次 LLM 压缩失败后失败点之前的新消息从摘要里永久丢失，且 summary 长期为 None 时
长会话会把上下文撑爆；② 历史查询隐式 `LIMIT 100` 且按 `created_at ASC` 取**最早** 100 条，
超过 100 条消息的会话"最近 20 条"窗口实际是伪最近（真正的尾部被丢掉）；③ 语义记忆
事实提取在主链路同步执行（agent 运行/SSE 开流**之前**），用主模型、无去重，
"我的/以后"等过宽关键词高频误触发。

**实现：**

1. **摘要折叠改为断点式（不丢段）**：`conversations` 新增 `last_summarized_message_id`
   列（模型 + `main.py` lifespan 幂等迁移 `ALTER TABLE ... IF NOT EXISTS`）。
   `maybe_update_summary` 只折叠该断点之后的新消息（单次最多 20 条防超长 prompt），
   **LLM 失败不推进断点，下次触发重试同一段**——中间段消息永不丢失。
   压缩改走 fast tier，降成本与延迟。
2. **历史窗口显式化 + 真实尾部**：`get_conversation_history` 支持显式 `limit/offset`；
   `MessageRepository.list_by_conversation` 支持 offset；新增纯逻辑
   `decide_history_window`（full / compressed / cap_tail 三态）——
   有摘要超窗口 → 摘要 + **真实尾部** 20 条；无摘要超窗口 → 取最近
   `HISTORY_CONTEXT_MAX_MESSAGES`（新配置，默认 100）条并记日志，
   替代旧的"最早 100 条隐式截断 / 无界增长"。
3. **事实提取后台化 + 去重 + 降级**：`add_message` 对含触发词的用户消息改为
   `asyncio.create_task` 后台调度（独立 session，持有引用防 GC，失败不影响主链路），
   不再阻塞 agent / SSE 开流；`extract_and_store_fact` 与 `add_user_fact` 走 fast tier，
   `add_user_fact` 按 (user_id, fact) 内容去重（重复返回旧记录）；
   触发词收紧（移除过宽的"以后"）。摘要更新保持同步（每 10 条消息触发，fast tier）。

**验证：**
- 新增 `tests/test_memory_context.py` **10/10 通过**：`decide_history_window` 三态
  （compressed 取真实尾部、cap_tail 显式兜底）与触发词收紧。
- 真实 PostgreSQL 临时数据验证（临时用户/会话，结束即清理）5/5：20 条消息后摘要生成
  且断点推进；LLM 模拟失败时断点不动、摘要不变；恢复后一次性折叠失败段（含上次
  失败的新消息）直达最后一条消息 id；事实去重同内容只存一条。
- 完整测试集（排除 2 个既有失败）`python -m pytest -q` 全绿，`compileall` 通过。
- `scripts/verify_memory_layers.py` 增加 3 项断言：新列存在、10 轮后断点推进、
  追加轮次后断点到达最后一条消息（不丢段）+ 压缩尾窗为真实最近窗口。

#### 2026-08-15 — 上下文管理 P1 修复：改写下沉 graph 入口 + 多智能体/ReAct 历史注入

**背景（评审结论）：** 指代消解只在 SSE 快速路径（prepare_context）执行，graph
路径（/chat/send）不执行——同一句"那明天呢"两条路径理解不一致；多智能体
（orchestrator）与 ReAct 循环的 prompt 完全不带对话历史，子任务/推理循环里
"那第二个呢"这类追问无从解析。（问题 12 摘要/事实切 fast tier 已在 P0 一并完成。）

**实现：**

1. **rewrite 下沉 graph 入口（两条路径统一）**：新增 `query_rewrite` 节点
   （Node 0），workflow 入口从 `intent_recognition` 改为 `query_rewrite` →
   `intent_recognition`。非追问查询零成本（`rewrite_query_with_history` 内部
   短路，不调 LLM）；改写失败/无历史原样返回。deepagents 路径本就有完整历史
   注入（`messages.extend(history)`），无需改动。
2. **ReAct 注入历史**：`REACT_REASONING` 模板新增 `{history}` 占位符，
   `agent_reasoning` 节点用 `_format_history_for_prompt` 注入最近 4 轮，
   ReAct 循环内可解析指代。
3. **多智能体注入历史**：`TaskBrief` 新增 `history` 字段（orchestrator.run
   统一赋值）；三个 Worker（rag/legal/code）经 `BaseWorker._history_context_message`
   注入"对话背景" system 消息（最近 4 轮 × 120 字截断，长度有界）；
   `_decompose` 拆解 prompt 注入历史并新增规则 6（追问必须结合历史理解、
   子任务必须自包含）；`_synthesize` 与 SSE 流式汇总 prompt（chat_router）
   同样注入历史。

**验证：**
- 新增 `tests/test_context_injection.py` **9/9 通过**：query_rewrite 节点改写/
  保原样、图编译含新入口、REACT_REASONING 模板带历史、TaskBrief.history 默认
  空、`_history_context_message` 有/无历史两种形态、RagWorker 消息注入与跳过、
  `_decompose` prompt 含历史。
- 真实 LLM 冒烟（临时会话）："那明天呢"（带无锡天气历史）→ query_rewrite 改写
  为自包含问题 → intent=tool_use → web_search 真实返回 → 回答校验通过，
  图入口改动端到端无回归。
- 完整测试集（排除 2 个既有失败）+ compileall 全绿。

#### 2026-08-15 — 多智能体触发规则 1 修订：领域字典 ≥2 领域命中

**背景：** 原规则 1 用 `domain_pairs` 手工列举关键词组合，但组合几乎全部围绕
法律领域（法律×代码、法律×法律比较、检索×写作），后续上传知识库类型多样
（金融/医疗/教育/生活…）时大量真正的跨领域查询（如"分析股票走势并整理成
报告"）不会触发多智能体。

**实现（`app/services/agent_service.py`）：**
- 新增 `_DOMAIN_KEYWORDS` 领域字典（12 个领域：法律/代码/写作/检索/分析/
  金融/医疗/科技/职场/生活/教育/历史），每个领域一行关键词；
- 规则 1 改为"查询命中 ≥2 个**不同领域**关键词 → multi"——任意跨领域组合
  天然命中，新增领域只需加一行，不再依赖手工 pair 枚举；
- 顺带修正注释（阈值 80 字符，旧注释误写 100）；
- 规则 2（长查询 + 连词）与 degenerate 兜底不变——规则 1 宁可放宽（误触发
  会被 LLM 拆解器打回单 Agent），不可漏掉真跨领域。

**行为变化：** "帮我写一个python脚本"（代码+写作）从 single 变为 multi；
原"法律×法律跨法条比较"（安全生产法 vs 民法典）不再因关键词 pair 触发，
由拆解器自行判断（单 Agent 知识库检索同样能处理）。

**验证：** 新增 `tests/test_multi_agent_routing.py` **17/17 通过**——旧跨域组合
保留触发、新增金融/医疗/教育/生活等多样性组合、单领域/无关查询不触发、
长查询+连词触发、领域字典覆盖 ≥8 领域且任意两领域组合命中。完整测试集
（排除 2 个既有失败）+ compileall 全绿。

#### 2026-08-21 — DeepAgents 成熟化 S1：知识库检索接入（消除幻觉风险）

**背景（见 `docs/plans/2026-08-21-deepagents-maturation-plan.md`，缺口 S1）：**
`_run_deep` 此前从不执行知识库向量检索，系统提示却声称"检索结果会作为上下文
提供"——DeepAgents 模式下知识库问答退化成纯 LLM 生成；且工具注册表没有任何
kb 检索工具，主/子 Agent 均无法检索知识库。

**实现：**
1. **请求级知识库授权上下文**（`app/services/knowledge_context.py`，仿
   `skills/context.py` 的 ContextVar 模式）：`use_authorised_kb_ids(ids)` /
   `get_authorised_kb_ids()`，跨线程需 `with` 设置，避免越权。
2. **`kb_search` 注册表工具**（`app/tools/kb_search.py`，自动发现即注册）：
   从请求级授权范围读取 kb_ids（无授权拒绝并提示），调增强检索，知识块/平铺
   格式化输出（含来源）。主 Agent 全量可见，research-agent 白名单已加并
   system prompt 引导"企业内部资料优先 kb_search"。
3. **`_run_deep` 前置检索注入**：组装 messages 时（有授权 kb_ids 时）执行增强
   检索，命中则注入 system 上下文（"回答时优先采用，并标注来源"），记录
   retrieve step/artifact，收集 `result.sources` 到返回结构；检索失败不阻塞。
   主 Agent 执行包在 `use_authorised_kb_ids` 作用域内，task 委派的 SubAgent
   调 kb_search 也能读到授权范围。

**验证：**
- 新增 `tests/test_deepagents_kb.py` **7/7 通过**：kb_search 已注册、无授权拒绝、
  有授权返回格式化结果、授权上下文设置/恢复、`_run_deep` 注入检索 system 消息
  且 sources 收集、无 kb 时跳过检索、research-agent 白名单含 kb_search；
  `tests/test_deep_agents.py` 11 项回归不破坏。
- 完整测试集（排除 2 个既有失败）+ compileall 全绿。
- **真实环境冒烟**（dev 库"法律"知识库 + 真实 LLM）：
  - 多轮路径：前置检索命中 12 条 ✓，主 Agent 推理中**自主调用 kb_search** 并
    拿到消费者权益法检索结果 ✓；
  - 单轮路径完整闭环：检索 → 注入 → 基于知识库回答（食品安全法召回规定）→
    sources 返回真实来源（食品安全法.pdf / 消权法.pdf）✓；
  - **暴露既有缺陷（S8 范畴，非本次引入）**：DeepSeek 思考模式下多轮工具调用
    报 `reasoning_content must be passed back to the API`（400）——ChatOpenAI
    适配层未处理思考内容回传，DeepAgents 多轮工具路径的真实可用性受影响，
    建议作为下一步修复项。

#### 2026-08-21 — DeepAgents 成熟化 S2：配置与入口补全（可达性）

**背景（见规划文档，缺口 S2）：** `AGENT_MODE` 类型定义缺 "deepagents"（代码
判断了该值但 Literal 没有，IDE/配置校验层面不认可）；`DEEP_SUBAGENTS_FILE`
未在 config 声明（pydantic-settings 未声明字段读不到 env → 外部 SubAgent
覆盖**实际失效**）；`.env.template` 无 AGENT_MODE/DEEP_* 文档；无启动日志。

**实现：**
1. `config.py`：`AGENT_MODE: Literal["auto","single","multi","deepagents"]` +
   注释；声明 `DEEP_SUBAGENTS_FILE: str = ""`、`DEEP_MAIN_RECURSION_LIMIT: int = 20`、
   `DEEP_SUBAGENT_RECURSION_LIMIT: int = 20`
2. `subagents.py` 修复外部覆盖失效：`load_subagents` 直接读 `cfg.DEEP_SUBAGENTS_FILE`
   （去掉失效的 getattr 兜底）；`_load_subagents_file` 健壮化——文件不存在/
   解析失败/无 subagents 列表 → 回退内置并告警（不因坏配置崩启动）；
   无 name 条目跳过
3. recursion_limit 接入配置：`build_main_agent` / `build_task_tool` 默认读
   `DEEP_SUBAGENT_RECURSION_LIMIT`（显式传参优先）；`_run_deep` 主 Agent 用
   `DEEP_MAIN_RECURSION_LIMIT`
4. `.env.template`：文档化 AGENT_MODE 四种取值与 DEEP_*（含外部 SubAgent
   文件 JSON/YAML 格式示例）
5. `main.py` lifespan：启动日志打印 AGENT_MODE；deepagents 模式额外打印
   SubAgent 名册（可发现性）

**验证：**
- 新增 `tests/test_deepagents_config.py` **11/11 通过**：AGENT_MODE Literal 接受
  deepagents、DEEP_* 字段声明且 env 可覆盖、外部文件加载（有效/缺失/坏 JSON/
  无 name 条目跳过/未配置回退内置）、recursion_limit 默认读配置且显式传参优先。
- 完整测试集（排除 2 个既有失败）+ compileall 全绿；真实 env 读取验证
  （AGENT_MODE=auto，DEEP_* 默认值正常）。

#### 2026-08-21 — DeepAgents 成熟化 S8：DeepSeek 思考模式多轮回传修复

**背景（S1 真实冒烟暴露的既有缺陷）：** DeepSeek reasoning 模型在响应中返回
`reasoning_content`（思考内容），OpenAI 兼容协议要求多轮对话把上一轮
assistant 消息的 `reasoning_content` 原样回传，否则 400
`The reasoning_content in the thinking mode must be passed back to the API`。
langchain-openai 1.4.1 的 `_convert_dict_to_message`（响应侧）与
`_convert_message_to_dict`（请求侧）都会丢弃该字段 → DeepAgents 主 Agent
多轮工具调用（调用 kb_search/web_search 之后）必然失败，只能单轮回答。

**实现（`app/agents/deep/llm.py`）：** 新增 `DeepSeekChatOpenAI(ChatOpenAI)`：
- 响应侧 override `_create_chat_result`：从响应（dict 或 openai.BaseModel）
  提取 `reasoning_content` 存入 `AIMessage.additional_kwargs`；
- 请求侧 override `_get_request_payload`：复刻 BaseChatOpenAI 实现，对
  assistant 消息从 `additional_kwargs` 补回 `reasoning_content` 字段
  （含带 tool_calls 的工具调用轮）；responses API 路径原样走基类；
- `get_langchain_model` 改用它；对非 reasoning 模型行为与 ChatOpenAI 完全一致。

**验证：**
- 新增 `tests/test_deepagents_reasoning.py` **8/8 通过**：请求侧回传（含
  tool_calls 轮）、响应侧保存（dict 与 openai.BaseModel）、无 reasoning_content
  时行为不变、`get_langchain_model` 返回适配类。
- 完整测试集（排除 2 个既有失败）+ compileall 全绿。
- **真实环境冒烟**（dev 库"法律"知识库 + 真实 DeepSeek）：多轮路径完整闭环
  —— 前置检索命中 12 条 → 主 Agent 推理 → **调用 kb_search** → 基于检索结果
  生成 1491 字回答（消权法消费者权利表，标注来源），此前该路径在工具调用后
  报 400 无法产出最终回答。

#### 2026-08-21 — DeepAgents 成熟化 S3：子 Agent 步骤透传（委派过程可见）

**背景（见规划文档，缺口 S3）：** task 委派的 SubAgent 执行过程是黑盒——
`_run_deep` 的 on_step/on_artifact 只覆盖主 Agent 的 stream，`run_subagent`
内部用 `agent.invoke` 一次性拿结果，前端 SSE 只能看到 "调用 task(...)" 与
一条"工具返回"，看不到子 Agent 的推理/工具调用。

**实现：**
1. **请求级观察者**（`app/agents/deep/observe.py`，ContextVar 模式）：
   `use_task_observers`（`_run_deep` 设置，包住主 Agent 执行）→ task 工具
   读取；`use_subagent_observers`（task 工具转发）→ `run_subagent` 读取。
   两层隔离避免子 Agent 步骤与主 Agent 步骤混淆。
2. **`run_subagent` 改 `agent.invoke` → `agent.stream(stream_mode="values")`**：
   有观察者时把子 Agent 的推理（thought artifact）/工具调用/工具返回/生成
   步骤以 `{subagent_name}/step` 形式透传；无观察者时跳过解析（行为不变）。
3. **`task_tool._task`**：读 task 观察者，存在时 `with use_subagent_observers`
   包住 `run_subagent`（SSE 回调同线程同步可见）。
4. **`_run_deep`**：主 Agent 执行作用域与 `use_authorised_kb_ids` 并列设置
   `use_task_observers(_step, _artifact)` —— 子 Agent 步骤直接进现有
   SSE artifact 通道，**前端无需改动**。

**验证：**
- 新增 `tests/test_deepagents_observe.py` **6/6 通过**：观察者设置/恢复、
  run_subagent 透传（tool/tool_done/generate 步骤 + thought artifact 带
  前缀）、无观察者行为不变、task 工具转发链路（有/无观察者）、`_run_deep`
  端到端委派步骤可见。
- 完整测试集（排除 2 个既有失败）+ compileall 全绿。
- **真实环境冒烟**（dev 库 + 真实 LLM）：主 Agent 委派 research-agent 后，
  步骤流完整呈现子 Agent 内部过程——`research-agent/tool: 调用 kb_search(...)`
  （2 次，知识库无命中）→ `research-agent/tool: 调用 web_search(...)`（2 次）
  → `research-agent/generate: 子智能体生成回答中...` → 子 Agent 输出研究
  报告（食品安全法消费者权益保护总结）返回主 Agent → 主 Agent 生成最终
  1109 字回答。前端 SSE 可完整看到委派过程。

#### 2026-08-21 — DeepAgents 输出模式改版：操作流（Cursor/Copilot 风格）

**背景（用户提供参考截图——AI agent 工作界面，Think/Read/Write/Edit 操作流）：**
要求 DeepAgents 的执行过程以"和你对话"的截图式操作流展示。用户确认：
操作流铺开（去折叠面板）+ 英文动词式 + 仅过程展示（最终回答保持 markdown）。

**实现（前端）：**
1. `AgentActivity.vue` 重写：去掉可折叠面板/头部，操作条目直接铺开在答案
   上方；步骤映射为**英文动词**——Rewrite（查询改写）/ Classify（意图）/
   Think（推理思考）/ **Read**（检索知识库）/ **Run**（调用工具，kb/web
   搜索动态变 **Search**）/ **Write**（生成回答）/ Answer / Check /
   Plan / Delegate / Merge / Fallback；等宽字体动作词 + 对象（工具名/
   内容摘要）截断 + 子 Agent 标签（research-agent 等，S3 透传的
   `subagent/step` 前缀自动解析）
2. 动作类型配色（截图式）：Think 蓝 / Read·Search 绿 / Run 紫 / Write 橙 /
   Check 青 / Delegate·Plan·Merge 粉 / Fallback 红
3. `style.css`：移除面板/折叠/头部样式，新增操作流样式（容器滚动、动词/
   对象/子 Agent 标签、kind 配色）
4. `ChatView.vue`：移除不再需要的 `v-model:expanded` 绑定

**验证：**
- `npm run build` 通过（修复步骤配对 bug：X_done 需把同 key 的 running
  步骤置 done，避免 tool/tool_done 重复行）
- 逻辑仿真（Node）：DeepAgents + 子 Agent 真实步骤流 → Read → Think →
  Run(task) → Search[kb_search] → Search[web_search] → Write[子 Agent] →
  Write 全部正确配对并带子 Agent 标签
- 视觉效果：需 EasyRAG 前端 dev server 刷新查看（HMR 生效则自动更新）

#### 2026-08-21 — DeepAgents 输出模式改版二期：act and reasoning（内容透出）

**背景（用户再次反馈"想的是 act and reasoning 这种感觉"，参考截图每条
`Think · <实际思考内容>` / `Tool call · <工具名> · <参数>` 内容可见）：**
一期只改了前端壳（英文动词+对象），后端步骤 detail 仍是固定短语
（"推理思考..."），思考内容被丢进下方折叠的 artifact 卡片，看不到"在想
什么、在做什么"。

**实现：**
1. `_run_deep`（agent_service.py）：`agent_reasoning` 步骤 detail 改为
   **实际思考内容**（400 字符截断摘要）；`tool` 步骤 detail 带**工具参数**
   （160 字符截断），如 `调用 kb_search {"query": "消费者权利 消权法"}`
2. `run_subagent`（subagents.py）：子 Agent 思考内容作为**独立 reason 步骤**
   透出（`research-agent/reason`）；工具步骤同样带参数
3. 前端：ACTION_MAP 加 `reason → Think`；Think 内容允许**两行折行**展示
   （obj-reason）；tool 配对**保留调用信息**（不被"工具返回: ..."覆盖，
   返回内容由 artifact 卡片承载）

**验证：**
- 全量测试（排除 2 个既有失败）+ compileall 全绿（observe 测试断言更新：
  reason 步骤 + 工具参数格式）
- 逻辑仿真：`Think · 知识库里有消费法律，先检索确认消费者权利再回答` →
  `Search · kb_search {"query": "消费者权利 消权法"}` → `Think [research-agent] ·
  先查企业知识库确认消权法内容` → ... —— 每条动作带实际推理/参数
- **真实冒烟**（dev 库 + 真实 LLM）：步骤流含 `agent_reasoning: 知识库中没有
  检索到相关内容，让我联网核实一下相关法律条款。`（真实推理文本）、
  `tool: 调用 web_search {"query": "食品安全法 消费者权利 知情权 赔偿..."}`

#### 2026-08-21 — 对话终止：停止生成 + 被终止的一轮不保存记录

**背景（用户需求）：** 添加终止对话的逻辑；当前这轮对话被终止时，
不被保存到记录（历史里不留"悬空"的用户提问）。

**实现：**
1. **前端**：
   - `api/index.js`：`streamChat` 支持 `options.signal`（AbortController）
   - `ChatView.vue`：生成中发送按钮变为红色**停止按钮**（Square 图标）→
     `stopGeneration()` abort 当前请求；AbortError 非错误（消息标记
     `stopped`，显示"已停止生成 · 本轮对话未保存"），失败路径不变；
   - `style.css`：`.btn-stop`（红色停止态）、`.message-stopped-note`
2. **后端**（chat_router.py `event_gen` + chat_service）：
   - `chat_service.delete_message()`：按 id 删除单条消息
   - `event_gen` 扩展：客户端断开（Starlette 以 `CancelledError` 取消
     生成器）→ `completed_normally=False` → 清理该轮：
     - 新建会话（is_new）→ 整会话删除（级联消息），不留空壳
     - 已有会话 → 删除该轮用户消息（assistant 在生成器内保存，流中断
       天然不落库）
   - run 终态处理（multi）原有 shield 逻辑保持不变

**关键坑：** 任务处于取消状态时，finally 里直接 `await` 会立即再次抛
`CancelledError`——清理必须用 `asyncio.create_task` + `asyncio.shield`
（与既有 run finalize 同款处理）。首次实现未 shield 导致清理静默失败，
真实冒烟（conversations 6→7 残留）暴露后修复。

**验证：**
- 真实服务冒烟（uvicorn + 真实 LLM + dev 库）：
  - 新会话发起流 → 3 个事件后强制断开 → 会话列表数量不变
    （terminated 新会话已删）
  - 已有会话发起流 → 断开 → DB `messages` 中无该轮记录
  - 修复前残留脏数据已清理（count 归 0）
- 全量测试（排除 2 个既有失败）+ compileall 全绿；`npm run build` 通过
- 最小复现实验确认：uvicorn 0.51（ASGI 2.3）下客户端断开 → 生成器收到
  `CancelledError` → finally 可执行（前提：DB 操作走 shield）

#### 2026-08-21 — Agent 路径可观测：DeepAgents 徽标（agent_mode 透出）

**背景（用户提问"如何知道当前是否调用了 deepagents 的能力"）：**
现状只有 done 事件的 intent="deepagents"（前端显示为模糊的"意图: 智能体"），
流式过程中完全看不到走了哪条链路；且 **.env 未配置 AGENT_MODE 时（默认 auto）
DeepAgents 永远不会被调用**（use_deep 仅 `== "deepagents"`），用户无法感知。

**实现：**
1. **后端**（chat_router.py）：计算 `agent_mode`（deepagents | multi | single），
   随 `conversation_id` 事件（流一开始就送达）与 `done` 事件下发，并写入
   落库 metadata（历史回放可见）
2. **前端**（ChatView.vue + style.css）：消息 meta 区新增**路径徽标**——
   DeepAgents（紫色渐变、醒目）/ 多智能体（粉色）/ 单 Agent（灰色），
   流一开始即显示（无需等 done）；历史消息从 metadata 读出同样显示

**验证（真实服务冒烟）：**
- AGENT_MODE=auto + "你好" → `agent_mode=single`（intent=chitchat）
- AGENT_MODE=auto + 跨领域问题 → `agent_mode=multi`（intent=multi_agent）
- AGENT_MODE=deepagents（环境变量覆盖）→ `agent_mode=deepagents`，
  完整跑通（1123 字回答）
- 全量测试 + compileall + `npm run build` 全绿

**启用 DeepAgents 的方法：** `.env` 设 `AGENT_MODE=deepagents`（或环境变量）
→ 每轮都走主 Agent + SubAgent 链路，消息上显示紫色 DeepAgents 徽标。

#### 2026-08-21 — 深度研究开关（按请求选择 DeepAgents，前后端打通）

**背景（用户需求）：** 前端添加"深度研究"按钮；选中时走 deepagents 工作流，
不选走原逻辑——不再依赖全局 AGENT_MODE 配置。

**实现：**
1. **后端**（chat_router.py）：
   - `ChatRequest` 新增 `deep_research: bool = False`
   - `use_deep = cfg.AGENT_MODE == "deepagents" or req.deep_research`；
     `use_multi = (not use_deep) and (...)` —— 深度研究优先，避免创建多余
     multi run
2. **前端**（ChatView.vue + style.css）：
   - 输入区新增**深度研究 toggle**（Sparkles 图标；选中紫色渐变，生成中禁用，
     title 说明用途），与发送按钮同组**紧贴发送按钮左侧**（`.composer-send-group`
     右侧按钮组；`space-between` 下左右分组各居一端，避免按钮悬在中间）
   - 请求体带 `deep_research`；用户消息显示紫色"深度研究"标记
   - 路径徽标（上一项）自动显示 DeepAgents

**验证（真实服务冒烟，默认配置 auto）：**
- 未选 + "你好" → `single`（chitchat）
- 未选 + 跨领域 → 原逻辑（本次未触发 multi，属正常）
- **选中 + 简单问题 → `deepagents`**（941 字，steps 含
  retrieve/tool/agent_reasoning）
- **选中 + 跨领域 → `deepagents`**（1297 字；深度研究优先于 multi）
- 全量测试 + compileall + `npm run build` 全绿

#### 2026-08-21 — Milvus 连接故障修复（HTTP_PROXY 环境变量干扰 gRPC）

**背景（用户贴日志）：** `[enhanced] BM25 sync failed / Path semantic failed /
gap query failed: MilvusException (Fail connecting to server on localhost:19530)`，
检索全面失效（含 eager build、多路召回、gap 查询）。

**排查过程：**
1. docker port/TCP 均通（127.0.0.1:19530 可达），容器 healthz OK
2. 容器间（easyrag-minio → milvus-standalone:19530）HTTP 响应正常
3. Windows → 19530 发 HTTP/1.1 请求有响应（REST 网关），发 HTTP/2 前奏
   收到 SETTINGS 帧——**网络层完全正常**，问题在 gRPC 客户端
4. grpcio 调试日志（GRPC_TRACE=tcp,handshaker）发现：
   `grpc.internal.endpoint_peer_address=ipv4:127.0.0.1:7897` —— **grpcio 把
   连接导向了 127.0.0.1:7897（本地代理）**！
5. 确认：环境变量 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897` +
   Windows 系统代理 ProxyServer=127.0.0.1:7897（Clash 类）——grpcio 自动
   经代理 CONNECT 连 Milvus，代理对 localhost gRPC 处理异常 → 握手超时。
   附带发现：`requests` 同样受代理影响（冒烟时曾出现 502，需 trust_env=False）
6. 期间还发现 pymilvus 被意外升级到 3.0.1（requirements 锁定 2.5.11，
   3.x 与 Milvus 2.5.14 服务器不兼容）→ 降回 2.5.11

**修复：**
- **`app/core/config.py`**：模块加载时自动把 `127.0.0.1, localhost, ::1`
  加入 `no_proxy` 环境变量——本地服务（Milvus/Postgres/Redis）gRPC 直连，
  外部 API 请求仍按需走 HTTP_PROXY（科学上网不受影响）
- 环境：`pymilvus==2.5.11`（匹配 requirements）、`grpcio==1.67.1`（匹配
  pymilvus 依赖约束）
- 清理：回滚了排查期间尝试的 Docker settings-store.json networkMode 改动
  （NAT 默认恢复）

**验证：**
- `no_proxy` 自动设置生效（无手动环境变量时 gRPC 连接成功）
- `utility.has_collection("rag_docs")` = True
- 真实检索冒烟：MilvusRetriever 6 hits（食品安全法条款内容正常返回）
- 全量测试 + compileall 全绿


#### 2026-08-24 — RAGAs 检索评估产品化 + 配置规范化 + 检索质量诊断

**背景（用户反馈）：**
- 检索“民法典第10条”结果不相关，怀疑是数据预处理（清洗）缺失
- 疑问：RAGAs 相关检索测试是否接入

**检索质量诊断（结论：不是预处理问题）：**
- 法律知识库（2d0fbee1）只入库了两份文档：`食品安全法.pdf`（182 chunks）、
  `消权法.pdf`（66 chunks）——**民法典从未入库**，检索“民法典第10条”必然
  返回低相似度的近似结果（实测 top 分数仅 ~0.52–0.56）
- 存量 chunk 质量良好：MinerU(pipeline) 解析 + `legal` 按条切分，中位长度
  138 字，带 `[章节]` 前缀；扫描无图片残留/页码/乱码/空块
- 实测对照：库内存在的查询精准命中——“消费者享有公平交易的权利”→
  `消权法第十条`（0.776），“食品安全法第八十条”→ 原文（0.79）
- **建议**：把民法典等目标法律文档上传入库即可，解析管线会自动检测法律条文
  并按「第X条」切分（`chunk_parsed_document` 内 `_looks_like_legal` 自动检测；
  配置 `CHUNK_STRATEGY` 注释说明该选项，未全局开启以保留结构化分块）

**RAGAs 评估产品化（阶段 4 前端先行）：**
- 新增前端页 `frontend/src/views/EvaluationView.vue` + 路由 `/evaluation` +
  侧边栏「检索评估」入口（Yuxi 单色系风格）：知识库/文件选择、用例编辑
  （问题+期望文件+可选参考答案）、运行评估、指标卡（HitRate/MRR/Recall/
  Precision/nDCG + RAGAs 状态）、历史列表与详情弹层
- 后端配套：`EvalCaseIn.expected_chunk_id/reference_answer` 改为可选 →
  文件级评估自动展开整文件 chunk 作为相关集（`_file_chunk_ids` +
  retriever 三后端新增 `list_chunks_by_source`）；`EvalRunSummary` 增加
  `ragas_status`；`RAGAS_PYTHON_EXECUTABLE` 显式配置优先，留空自动探测
  `.venv-ragas`/`venv-ragas`/`ragas-env`
- 端到端验证：HTTP 冒烟（登录→知识库→文件→POST /evaluation/runs→GET 历史）
  真实落库一条 run，本地指标 HitRate=1.0/MRR=0.5，RAGAs 子进程 completed
  （0.4.3，id_context_precision=0.625）
- 修复：`evaluation_router` 缺 `Optional` 导入导致 pydantic 解析失败；环境
  补装 `pytest-asyncio`（测试依赖）

**配置规范化：**
- `.env` 因混编码损坏（UTF-8/GBK 混杂、注释与键合并）→ 按 `.env.template`
  以纯 UTF-8 重建，键集与模板完全对齐（115 键，仅真实值与占位符差异）
- `.env.template` 去重 Ragas 注释头、补齐 `GRAPH_EXTRACT_REASONING_EFFORT`、
  `CHUNK_STRATEGY` 说明注释；RAGAS_LLM_* 键同步
- 测试：评估/RAGAs/检索相关 61 例全绿；`npm run build` 通过

**补充修正（用户反馈）：评估页位置归位**
- 独立侧边栏「检索评估」+ `/evaluation` 路由已移除（LayoutView/router）
- 评估功能嵌入知识库详情页的「RAG 评估」标签：KnowledgeView 用
  `<EvaluationView :kb-id :kb-name>` 替换原占位区块（原为 frontendOnly 假页面）
- EvaluationView 增加 `kbId/kbName` props：嵌入时隐藏页头与知识库选择器、
  历史按当前知识库过滤、隐藏冗余「知识库」列；`ev-embedded` 去内边距
- 清理：移除 `showEvaluationSetup` 弹窗/`openEvaluationSetup`/`enabledCriteria`
  及失效图标导入；tab「前端」角标改为仅标记 `map`/`benchmarks`
- `npm run build` 通过


#### 2026-08-24 — 规范化 RAG 评测体系（数据/指标/执行/报告四层）

**背景：** 用户要求把检索评估升级为「规范化 RAG 评测体系」，而不是临时测试脚本，
能够在面试中讲清楚检索测试的编排方法。基于上一轮对 RAGAs 官方口径的核对结论实施。

**核心修正（reference 口径对齐 RAGAs 官方语义）：**
- `EvaluationCase` 新增 `expected_chunk_ids`（question-specific 相关 chunk 集）与
  `expect_miss`（负样本）；reference 解析优先级：负样本 > chunk_ids > 单 chunk >
  整文件兜底，并在逐条明细里记录 `reference_mode` 便于解释指标
- 修复「整份文件当相关集导致 Recall 被压扁 / Precision 虚高」的系统性偏差
- 负样本单独走误报检测（`false_positive`），不计入常规 missed 统计

**新增：**
- 评测数据集（Golden Set）持久化：`evaluation_datasets` 表 + 同名保存递增 version，
  `backend/services/evaluation_datasets.py` 提供序列化/导入导出/CRUD；
  运行表增加 `dataset_id` 关联，同一数据集可反复跑不同配置做 A/B
- 运行环境快照 `run_metadata`（embedding 类型/模型、chunk 策略、score 阈值、
  enhanced/graph 开关），保证结果可复现
- 失败分析 `analysis`：自动归类 missed / low_recall / false_positive 三类问题
- Markdown 报告导出：`GET /evaluation/runs/{id}/report`（环境快照 + 聚合指标 +
  RAGAs 指标 + 逐条明细 + 失败分析），`backend/services/evaluation_report.py`
- Golden set 标注辅助：`POST /evaluation/chunk-candidates` 返回目标文件内的候选
  chunk（片段+分数），前端勾选后写入 expected_chunk_ids
- 前端 EvaluationView：评测集保存/加载、负样本勾选、候选 chunk 标注面板、
  报告下载、明细表新增「参考类型」列
- 指标集按运行覆盖：`POST /evaluation/runs` 支持 `ragas_metrics` 覆盖全局
  `RAGAS_METRICS`；`get_ragas_evaluator(settings, metrics=None)` 支持按运行指定

**文档：** 新增 `docs/RAG_EVALUATION.md` —— 四层体系方法论、Golden Set 构造、
指标分类与取舍、A/B 实验流程、失败分析解读、面试讲解话术。

**验证：** 评测/检索/RAGAs 相关 60+ 测试全绿（含新增 reference 语义测试、
数据集序列化测试、报告生成测试）；`npm run build` 通过。


#### 2026-08-24 — 修复存量库 evaluation_runs.dataset_id 缺失（500）

**现象：** `GET /api/v1/evaluation/runs` 返回 500，
`asyncpg.UndefinedColumnError: column evaluation_runs.dataset_id does not exist`。

**根因：** 新增 `EvaluationRun.dataset_id` 列后，`create_all` 只创建缺失的表、
不会给已存在的 `evaluation_runs` 表补列，存量库缺列导致 ORM 查询失败。

**修复：** `backend/storage/postgres/manager.py` 增加幂等迁移
`_migrate_legacy_evaluation_runs`：先查 `information_schema.columns`，
仅在列缺失时执行 `ALTER TABLE ADD COLUMN` + 索引 + 外键；
`init_db()` 在 `create_all` 后调用。新旧库均安全（新库由 create_all 建列后跳过）。

**验证：** 对线上库执行 `init_db()` 迁移成功；`list_runs` 查询正常；
HTTP 冒烟 `GET /evaluation/runs` 200、`POST/GET /evaluation/datasets` 正常、
`GET /evaluation/runs/{id}/report` 输出 Markdown 报告；评测相关测试全绿。

#### 2026-08-24 — 安装 @liustack/modlens 视觉桥插件（Codex 读图）
**背景：** 用户发检索测试截图时模型显示 `[Unsupported Image]`，纯文本模型需补视觉能力。
**安装：**
- 全局安装 CLI：`npm i -g @liustack/modlens@3.24.1`（bin: modlens，node 24 满足 22.19+ 要求）
- skill 复制到 `C:\Users\Administrator\.codex\skills\modlens\`（`modlens doctor` 已识别 `codex: pins 3.24.1`）
**配置（~/.modlens/config.json，默认 provider=openai）：**
- `openai.baseUrl = https://coding.dashscope.aliyuncs.com/v1`
- `openai.apiKey = sk-sp-…789f`（用户提供，仅存用户目录，不进仓库）
- `openai.model = qwen3.7-plus`
**验证：**
- `modlens doctor`：openai ok，故障切换链 `openai -> claude-cli`，guard 放行
- 真实读图 `frontend/smoke-chat.png`：OCR/布局/语义完整（Edge「未找到文件」错误页），model=qwen3.7-plus，30.9s
**注意：** skill 列表在会话启动时加载，当前会话看不到 modlens skill；新会话即可直接粘贴图片/拖入文件路径触发。

#### 2026-08-26 — DeepAgents 统一多智能体改造（阶段 1-5，Orchestrator-Worker 退役）

**背景：** 原 Orchestrator-Worker 多智能体（LLM 拆解 → Worker 派发 → 汇总）存在
固定编排、单例并行竞态、工具白名单静态化等问题；评审后决策统一到 LangGraph 原生
DeepAgents：**由主 Agent（create_react_agent）根据工具描述自主决定拆解/委派/汇总**，
拆解器不再作为独立流程存在。分五个阶段实施（容错可观测 → 动态授权 → DAG+黑板 →
动态规划 → 异步增量与退役）。

**阶段 1 — 容错与统一事件流：**
- `app/tools/registry.py`：工具调用超时（可配置）+ 指数退避重试 + 事件钩子，
  `ToolDefinition` 新增 tags / timeout / max_retries 元数据字段
- `app/agents/events.py`：统一事件流（请求级 TraceContext + span + emit + sink），
  `snapshot_request_context()` 快照请求级 ContextVar（聊天模型/KB 授权/Skill），
  供线程池任务重放（ThreadPoolExecutor 不自动传播 contextvars）
- `deep/task_tool.py` 熔断：SubAgent 连续失败熔断后续委派，降级返回失败说明
- `_run_deep` 超限降级：recursion limit 超限时降级为直接生成而非整轮失败

**阶段 2 — 动态工具授权：**
- SubAgent 配置支持动态工具绑定：`*`（全量）/ `except:`（排除）/ `@tag`（按标签）
  + 缓存指纹（工具集变化自动失效）
- 越权调用错误信息附带当前可用工具清单，便于模型自纠
- kb_search 授权范围从 `use_authorised_kb_ids` ContextVar 读取——工具签名不携带
  请求上下文通道，模型无法通过工具参数越权（取代旧 TaskBrief 传参隔离）

**阶段 3 — DAG 委派 + 结构化黑板：**
- `deep/planner.py` `spawn_tasks`：`depends_on` 拓扑分层 + 线程池并行执行，
  依赖产出摘要自动注入下游任务；请求上下文快照逐任务重放
- `deep/blackboard.py` 结构化黑板：产出物 `{key, producer, summary, data, tags, version}`
  摘要/全量两级共享 + 依赖订阅通知
- 路由收敛：`AGENT_MODE=multi` 变为 `deepagents` 兼容别名（仅告警），
  `agent_mode` 徽标收敛为 `deepagents | single` 两值；orchestrator 路径冻结
- `_run_deep` 注入知识库目录（`format_knowledge_catalog` system 消息）

**阶段 4 — 动态规划：**
- `revise_plan` 委派工具：运行中增/改/取消任务（结构化尾部输出）
- `run_subagent` 结构化尾部：SubAgent 结果摘要标准化供主 Agent 整合

**阶段 5 — 可观测/持久化/前端/退役：**
- `observability/tracing.py`：OTel 可选 span（`tool.invoke.<name>` /
  `subagent.<type>` / `spawn_tasks`），未装 OTel 时为 no-op；工具进度回调接入事件流
- `backend/services/delegation_service.py`：统一事件 → Run/Task/AgentRun 三表落库
  （best-effort，复用既有表）；`bridge_delegation_event` 把统一事件桥接为既有
  任务面板协议（sub_tasks/status/worker_output/progress_summary），前端零新组件复用；
  `/chat/send` 的 run_id 完全由 `persist_delegation` 返回值提供（不再预创建 run）
- chat_router stream 端点删除 404 行 orchestrator 分支（回调/流式汇总/落库），
  决策块折叠为 `use_deep`；旧 `_synthesize` token 治理由主 Agent 整合与委派结果截断承接；
  对话历史由 deep agent checkpointer 会话记忆承接（原 TaskBrief.history 语义）
- **退役删除**：`app/agents/orchestrator.py`、`app/agents/blackboard.py`、
  `app/agents/workers/`（base/rag/legal/code worker 全部）；
  `app/agents/__init__.py` 重写为 DeepAgents 层描述
- 测试迁移：`test_agent_run_persistence` / `test_chat_model_selection` /
  `test_context_injection` / `test_retrieval_isolation` / `test_knowledge_catalog`
  中 Orchestrator/Worker 用例迁移为 deep 路径等价物（上下文快照并发模型继承、
  kb_search ContextVar 隔离、目录注入 system 消息等），30 测试全绿；
  新增 `scripts/verify_deepagents_unified.py` 静态+可选 live 冒烟 13/13 通过
- 文档同步：README / `docs/ARCHITECTURE.md` / `docs/ARCHITECTURE_DETAILED.md`
  （架构图、目录树、5.2/5.4 路由、第 10 节重写、第 18 节端到端时序）；
  `.env.template` 与 `app/core/config.py` 注明 `multi` 为废弃别名
