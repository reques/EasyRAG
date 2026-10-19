# 工作区与独立会话实施计划

> 后续修正：用户要求空白新会话只保留临时草稿，首次发送时才持久化。以 `2026-10-18-workspace-drafts-and-project-menu.md` 和 `docs/workspace-sessions.md` 的现行行为为准。

**目标：** 实现类似截图的项目分组侧边栏，一个工作区容纳多个持久化会话，每个会话立即获得独立 session ID。

**架构：** 在现有 Conversation 上增加可空 workspace_id；Conversation.id 继续作为 session/thread/checkpoint 的唯一标识，不改变执行和观测链路。Workspace 仅组织会话，不自动合并消息或上下文；现有用户级记忆、知识库权限保持原语义。

**技术：** FastAPI、SQLAlchemy / PostgreSQL、Vue 3 / Pinia。

## 设计取舍

- 采用数据库持久化工作区和会话归属，支持跨浏览器恢复。仅浏览器分组无法满足这一点。
- 不采用嵌套会话或为每个工作区复制 Agent 运行时，避免混淆 workspace、session 和 execution 三种 ID。
- 旧会话保留原 ID，显示在“未分组”；移除工作区只解除分组，保留会话与历史。
- 工作区支持创建、重命名、移除、折叠；会话支持创建、重命名、移动、删除、复制 ID。默认显示近期五条，可以展开全部。
- 选择会话更新 URL 的 session 参数，刷新和链接恢复同一会话；本地选择状态按用户隔离。

## 实施步骤

1. **模型与迁移**：修改 `backend/storage/postgres/models_conversation.py` 和 `manager.py`，创建 workspaces 表、增加 conversations.workspace_id 及索引。迁移可重复执行，不重写历史。
2. **API**：新增 `backend/server/routers/workspace_router.py` 与 `backend/services/workspace_service.py`，接入 `backend/server/main.py`。工作区及会话读写都按当前用户鉴权；移动必须同时拥有源会话和目标工作区，删除工作区与归属变更使用同一行锁顺序。
3. **兼容现有对话**：`chat_router.py` 会话列表加入 workspace_id / session_id，提供分页；预创建空会话的首轮仍能自动生成标题，手动改名不被覆盖。保持 SSE 断线、取消和 trace 标识不变。
4. **状态管理**：改造 `frontend/src/stores/chat.js`，分页拉取全部会话，恢复用户隔离的选择状态，处理并发加载及过期响应；创建会话先落库再选中。
5. **界面**：`LayoutView.vue` 使用独立分组侧边栏组件；增加工作区菜单和管理表单，顶部显示工作区 / 会话与 session ID。保持已有知识库、记忆、MCP 入口。
6. **验证**：隔离 PostgreSQL 中验证旧库迁移幂等、两用户越权、独立历史、移动及移除工作区不丢数据；回归断线、取消和恢复。前端构建、状态测试和真实浏览器操作覆盖创建两个会话、切换、刷新、重命名及移动。

## 验收边界

这次不添加共享项目文件、工作区级提示词或 Git worktree。默认远程模型网关证书问题不阻塞会话管理 CRUD 验收，也不通过关闭 TLS 校验绕过。
