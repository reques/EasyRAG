# EasyRAG 项目架构总览

> 最后更新：2026-07-30 | 当前阶段：阶段 1 完成

---

## 完整目录树

```
EasyRAG/
│
├── .env                          # 环境变量（从 .env.template 复制）
├── .env.template                 # 完整配置模板
├── requirements.txt              # 原始依赖（langchain, pymilvus 等）
├── requirements-stage1.txt       # 阶段 1 新增依赖
├── docker-compose.yml            # Docker 编排（6 个服务）
├── PROGRESS.md                   # 演进记录 / 变更日志
├── ARCHITECTURE.md               # ← 你正在看这个
├── README.md                     # 项目介绍
├── LICENSE
│
├── run.py                        # [旧] FastAPI 入口 (uvicorn run:app)
├── gradio_app.py                 # [旧] Gradio 调试前端 (端口 7860)
├── test.py                       # [旧] Milvus 连接测试
├── append_gradio.py              # [旧] Gradio 扩展
├── patch_gradio.py               # [旧] Gradio 补丁
├── fix_summary.py                # [旧] 摘要修复
├── 基于langgraph的知识库问答系统.md  # 中文技术文档
│
├── models/                        # 模型文件目录（需自行下载 BGE-M3）
├── volumes/                       # Docker 数据卷（etcd/milvus/minio 等）
│
├── frontend/                      # ─── 阶段 4: Vue 3 前端 ───
│   ├── package.json               #   Vue 3.5 + Vite 6 + Pinia 2 + Axios
│   ├── vite.config.js             #   代理 /api → :8000
│   ├── index.html
│   └── src/
│       ├── main.js                #   入口
│       ├── App.vue                #   根组件
│       ├── style.css              #   全局样式
│       ├── api/index.js           #   Axios + JWT 拦截器
│       ├── stores/auth.js         #   Pinia 认证
│       ├── router/index.js        #   路由 + 守卫
│       └── views/
│           ├── LoginView.vue      #   登录
│           ├── RegisterView.vue   #   注册
│           ├── LayoutView.vue     #   主布局（侧边栏）
│           ├── ChatView.vue       #   对话
│           └── KnowledgeView.vue  #   知识库管理
│
├── app/                           # ─── 旧核心代码（保留兼容）───
│   ├── __init__.py
│   ├── core/                      # 基础设施
│   │   ├── config.py              #   pydantic-settings 配置中心
│   │   ├── logger.py              #   应用日志
│   │   └── exceptions.py          #   分层异常体系
│   ├── llm/                       # LLM 客户端
│   │   └── client.py              #   OpenAI 兼容 SDK 封装 (sync/async/JSON)
│   ├── rag/                       # 检索增强
│   │   ├── embeddings.py          #   嵌入模型 (local BGE-M3 / OpenAI API)
│   │   ├── chunker.py             #   文档解析 + 分块 (.txt/.md/.pdf/.docx)
│   │   ├── retriever.py           #   检索引擎 (Memory/Milvus/Chroma)
│   │   └── vector_store.py        #   [旧] 向量库抽象（已迁移到 retriever）
│   ├── graph/                     # LangGraph 工作流
│   │   ├── state.py               #   AgentState 共享状态
│   │   ├── nodes.py               #   8 个节点函数
│   │   ├── router.py              #   6 个条件路由函数
│   │   └── workflow.py            #   图组装 + 编译
│   ├── tools/                     # 工具调用
│   │   ├── registry.py            #   工具注册中心
│   │   ├── calculator.py          #   安全计算器
│   │   ├── datetime_tool.py       #   日期时间工具
│   │   └── text_tool.py           #   文本处理工具
│   ├── prompts/                   # Prompt 模板
│   │   └── templates.py           #   6 个 Prompt 模板
│   ├── services/                  # 服务层
│   │   └── agent_service.py       #   AgentService + SessionStore (内存)
│   └── api/                       # API 路由
│       ├── routes.py              #   /health, /chat, /ingest, /tools
│       └── kb_routes.py           #   /kb/upload, /kb/search, /kb/ask 等
│
└── backend/                       # ─── 阶段 1 新架构代码 ───
    ├── __init__.py
    ├── storage/                   # 存储抽象层
    │   ├── __init__.py
    │   ├── postgres/
    │   │   ├── manager.py         #   SQLAlchemy async 引擎 + 会话工厂
    │   │   ├── models_user.py     #   Department / User
    │   │   ├── models_conversation.py  # Conversation / Message
    │   │   └── models_knowledge.py     # KnowledgeBase / KnowledgeFile
    │   ├── redis/
    │   │   └── manager.py         #   Redis async 客户端
    │   └── minio/
    │       └── client.py          #   MinIO 文件存储客户端
    ├── repositories/              # 数据访问层 (Repository 模式)
    │   ├── base.py                #   BaseRepository[T] 泛型基类
    │   ├── user_repository.py     #   User CRUD
    │   ├── conversation_repository.py  # Conversation + Message CRUD
    │   └── knowledge_repository.py     # KnowledgeBase + KnowledgeFile CRUD
    ├── services/                  # 业务逻辑层
    │   ├── auth_service.py        #   注册/登录/JWT/密码哈希 (bcrypt)
    │   ├── chat_service.py        #   对话 CRUD + 调用 Agent
    │   └── knowledge_service.py   #   知识库 CRUD + 文件管理
    └── server/                    # HTTP 服务层
        ├── main.py                #   新 FastAPI 入口（整合新旧路由）
        ├── seed.py                #   管理员种子脚本
        ├── routers/
        │   ├── auth_router.py     #   POST /auth/register, /auth/login
        │   ├── chat_router.py     #   POST /chat/send, GET /chat/conversations
        │   └── knowledge_router.py    # CRUD /knowledge/bases
        └── utils/
            └── auth_middleware.py #   get_current_user (JWT 依赖注入)
```

---

## 分层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      server/routers/                        │  ← HTTP 层
│   auth_router  │  chat_router  │  knowledge_router          │
│        + 旧 app/api/routes.py + app/api/kb_routes.py       │
├─────────────────────────────────────────────────────────────┤
│                      services/                              │  ← 业务逻辑层
│   auth_service  │  chat_service  │  knowledge_service       │
│        + 旧 app/services/agent_service.py                  │
├─────────────────────────────────────────────────────────────┤
│                     repositories/                           │  ← 数据访问层
│   UserRepo  │  ConversationRepo  │  KnowledgeRepo           │
├─────────────────────────────────────────────────────────────┤
│                      storage/                               │  ← 存储抽象层
│   postgres/        redis/        minio/                     │
├─────────────────────────────────────────────────────────────┤
│   app/llm/  │  app/rag/  │  app/graph/  │  app/tools/       │  ← 领域核心
│   app/prompts/                                              │
├─────────────────────────────────────────────────────────────┤
│                    app/core/                                │  ← 基础设施
│   config.py  │  logger.py  │  exceptions.py                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Docker 服务拓扑

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ postgres │   │  redis   │   │  minio   │   ← 阶段 1 新增
│  :5432   │   │  :6379   │   │ :9090/91 │
└──────────┘   └──────────┘   └──────────┘
      │              │              │
      ▼              ▼              ▼
┌─────────────────────────────────────────┐
│           FastAPI :8000                  │
│   (backend/server/main.py)              │
└─────────────────────────────────────────┘
      │              │              │
      ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│ milvus   │   │  etcd    │   │minio-s3  │   ← 旧有（Milvus 三件套）
│  :19530  │   │  :2379   │   │ :9000/01 │
└──────────┘   └──────────┘   └──────────┘
```

---

## API 端点全览

### 认证 (阶段 1 新增)
| 方法 | 路径 | 认证 |
|------|------|------|
| POST | `/api/v1/auth/register` | 无 |
| POST | `/api/v1/auth/login` | 无 |

### 对话 (阶段 1 新增)
| 方法 | 路径 | 认证 |
|------|------|------|
| POST | `/api/v1/chat/send` | JWT |
| GET | `/api/v1/chat/conversations` | JWT |
| GET | `/api/v1/chat/conversations/{id}/history` | JWT |

### 知识库 (阶段 1 新增)
| 方法 | 路径 | 认证 |
|------|------|------|
| POST | `/api/v1/knowledge/bases` | JWT |
| GET | `/api/v1/knowledge/bases` | JWT |
| GET | `/api/v1/knowledge/bases/{id}/files` | JWT |

### 系统 + Agent + RAG (旧接口，保持兼容)
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 系统健康检查 |
| GET | `/api/v1/tools` | 已注册工具列表 |
| POST | `/api/v1/chat` | 旧 Agent 对话（无 DB 持久化） |
| POST | `/api/v1/ingest` | 文本直接入库 |
| POST | `/api/v1/kb/upload` | 文件上传 + 索引 |
| POST | `/api/v1/kb/ingest_texts` | 文本片段入库 |
| POST | `/api/v1/kb/search` | 语义检索 |
| POST | `/api/v1/kb/ask` | RAG 问答 |
| GET | `/api/v1/kb/info` | 知识库统计 |
| GET | `/api/v1/kb/health` | 知识库健康检查 |
| DELETE | `/api/v1/kb/collection` | 清空知识库 |

---

## 数据库表 (PostgreSQL)

```
departments         users               conversations
┌─────────────┐    ┌─────────────┐     ┌────────────────┐
│ id (UUID)   │◄───│ dept_id     │     │ id (UUID)      │◄──┐
│ name        │    │ username    │     │ title          │   │
│ description │    │ email       │     │ user_id (FK) ──┼───┤
│ is_active   │    │ password    │     │ created_at     │   │
│ created_at  │    │ role        │     │ updated_at     │   │
└─────────────┘    │ is_superuser│     └────────────────┘   │
                   │ created_at  │                          │
                   └─────────────┘      messages            │
                                        ┌────────────────┐  │
knowledge_bases    knowledge_files      │ id (int)       │  │
┌─────────────┐    ┌──────────────┐     │ conv_id (FK) ──┼──┘
│ id (UUID)   │◄───│ kb_id (FK)   │     │ role           │
│ name        │    │ filename     │     │ content        │
│ description │    │ file_type    │     │ metadata_json  │
│ owner_id    │    │ minio_bucket │     │ created_at     │
│ dept_id     │    │ minio_object │     └────────────────┘
│ collection  │    │ chunk_count  │
│ created_at  │    │ status       │
└─────────────┘    └──────────────┘
```

---

## 数据流 (一次对话请求)

```
POST /api/v1/chat/send  {query, conversation_id}
        │
        ▼
chat_router.send_message()
        │
        ├── get_current_user()         ← JWT 中间件解析用户
        ├── chat_service.get_conversation()  ← PostgreSQL 查询会话
        ├── chat_service.add_message()       ← 保存用户消息到 DB
        │
        ├── app/services/agent_service.run() ← 调用 LangGraph Agent
        │       │
        │       ├── intent_recognition       ← LLM 意图识别
        │       ├── knowledge_retrieval      ← Milvus 检索
        │       ├── answer_generation        ← LLM 生成回答
        │       └── answer_validation        ← LLM 质量检查
        │
        ├── chat_service.add_message()       ← 保存助手回复到 DB
        └── return ChatResponse
```
![alt text](image.png)