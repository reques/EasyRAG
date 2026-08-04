# EasyRAG → 企业级 Agent 平台 — 演进记录

> 参考项目：[Yuxi](https://github.com/xerrors/Yuxi) (语析) — 多租户 Agent Harness + 企业知识库平台
> 目标：从 EasyRAG 当前代码逐步重构为企业级多智能体平台

---

## 总体路线

```
阶段 1: 后端架构化   FastAPI分层 + PostgreSQL/Redis/MinIO + Repository模式  ✅ 完成
阶段 2: 知识库增强   多策略分块 + OCR链路 + 知识图谱 + 评估管线            ✅ 完成
阶段 3: Agent 体系    Sub-agent + MCP + Skill系统 + 中间件
阶段 4: 产品化       Vue3前端 + 多租户 + 管理后台                    🚧 前端先行
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
