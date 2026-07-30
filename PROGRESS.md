# EasyRAG → 企业级 Agent 平台 — 演进记录

> 参考项目：[Yuxi](https://github.com/xerrors/Yuxi) (语析) — 多租户 Agent Harness + 企业知识库平台
> 目标：从 EasyRAG 当前代码逐步重构为企业级多智能体平台

---

## 总体路线

```
阶段 1: 后端架构化   FastAPI分层 + PostgreSQL/Redis/MinIO + Repository模式  ✅ 完成
阶段 2: 知识库增强   多策略分块 + OCR链路 + 知识图谱 + 评估管线
阶段 3: Agent 体系    Sub-agent + MCP + Skill系统 + 中间件
阶段 4: 产品化       Vue3前端 + 多租户 + 管理后台                    🚧 前端先行
```

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
