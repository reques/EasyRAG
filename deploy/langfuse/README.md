# 本地 Langfuse

独立的可选服务栈：Langfuse Web/worker **4.63.0**、PostgreSQL 17.11、Redis 7.4、ClickHouse 25.12 和 S3 对象存储。根据 [官方 Compose](https://github.com/langfuse/langfuse/blob/v4.63.0/docker-compose.yml) 适配，满足 Python SDK 5.0.2 对服务端 ≥4.59.0 的要求。

所有命令在仓库根目录运行，使用已安装 requirements 的 Python 3.11+。Windows 示例：

```powershell
.\.venv\Scripts\python.exe scripts/local_langfuse.py init
.\.venv\Scripts\python.exe scripts/local_langfuse.py up
.\.venv\Scripts\python.exe scripts/local_langfuse.py connect
.\.venv\Scripts\python.exe scripts/local_langfuse.py app build backend frontend
.\.venv\Scripts\python.exe scripts/local_langfuse.py app up -d --no-deps backend frontend
```

`init` 独占创建本目录 `.env`，生成随机数据库密码、加密密钥、项目 API keys 和控制台密码；再次运行不更换已有凭据。使用 [官方 headless initialization](https://langfuse.com/self-hosting/administration/headless-initialization) 创建组织、项目和管理员，关闭公开注册及产品遥测。

控制台地址为 `http://localhost:3000`，账号 `admin@easyrag.local`；密码位于本目录 `.env` 的 `LANGFUSE_INIT_USER_PASSWORD`。该文件被本目录 `.gitignore` 和根 `.gitignore` 忽略。请通过本地编辑器查看，不要把文件、完整 Compose 配置或 `docker inspect` 输出贴到日志/提交中。

如端口冲突，在初始化后修改本目录 `.env` 的 `LANGFUSE_WEB_PORT`（默认 3000）或 `LANGFUSE_BLOB_PORT`（默认 9190），再运行 `up` 和 `connect`。仅 Web 和对象 API 发布到 **127.0.0.1**；数据库、Redis、ClickHouse、对象控制台和 worker 不发布宿主端口。对象 API 使用 9190，避开项目现有 RustFS 的 9090/9091。

若默认镜像域名下载大层时发生 EOF，可在本目录 `.env` 添加 `LANGFUSE_IMAGE_REGISTRY=docker.io` 或 `ghcr.io` 后重试；二者均由 [官方发布流水线](https://github.com/langfuse/langfuse/blob/v4.63.0/.github/workflows/pipeline.yml) 发布同版本镜像。切换前核对 manifest digest，不使用来历不明的镜像代理。

`connect` 先检查 Web 健康、项目 key 认证及项目 ID，全部通过才更新根 `.env` 中的观测字段，保留其他配置。后端上报地址 `http://langfuse-web:3000` 与浏览器地址分开；整次执行采样率为 1，内容策略为 `metadata_only`。

`app` 是根 Compose 命令的包装：自动包含原有 `docker-compose.override.yml`（若存在）和可选的 `docker-compose.langfuse.yml`。后者把 backend 接入 `easyrag-observability` 网络，并补齐 NO_PROXY。使用 `app` 更新相关容器，可以避免后续普通 `docker compose up` 丢失附加网络。

Langfuse 的数据库、Redis、ClickHouse 和对象数据各用独立命名卷。私有网络内服务使用 `lf-*` 名称；共享网络仅连接 Web，避免与业务的 `postgres`/`redis` DNS 别名冲突。不会复用或清空 EasyRAG 数据目录。

## 真实验收

```powershell
.\.venv\Scripts\python.exe scripts/verify_langfuse_live.py --live --backend http://localhost:8000 --report tmp/langfuse-live-report.json
```

脚本会创建一个合成 EasyRAG 普通账号，真实调用配置的模型，要求主智能体委派 coding-agent 并由子智能体调用 calculator。它还会主动断开 SSE、模拟刷新后的任务查询、重新订阅以及幂等重发，最后从 Langfuse v4 observations API 等待索引并验证：

- 同一次执行只有一条 trace、一个 `execution.chat` 根节点；模型、工具、子智能体均可沿 parent ID 回溯到该根。
- 子智能体的模型和 calculator 确实位于子智能体下面；用户、会话、execution ID、实际模型名和时间字段完整。
- `metadata_only` 下没有模型/工具输入输出正文；普通账号无法访问管理员观测跳转接口。
- 模型请求进入执行后（包括等待响应、尚未输出正文）的主动取消最终为 cancelled；检查真实 generation 已存在，根与子节点不被误标为 ERROR，停止到业务终态少于 10 秒。

脚本不会以假模型替代真实验收；缺少任何必需节点都会失败。合成账号会保留，便于核对，凭据仅写入忽略的 `verify.env.local`。API keys、token、问题和回答不会写入报告。管理员跳转页面还需在实际部署的浏览器中验收。

本地无服务时可运行 `scripts/verify_langfuse.py` 验证内存 SDK 行为；**它不证明 Langfuse 已部署或已收录 trace**。实际状态见 [实施记录](../../docs/langfuse-implementation-verification.md)。

## 运维与关闭

```powershell
.\.venv\Scripts\python.exe scripts/local_langfuse.py status
.\.venv\Scripts\python.exe scripts/local_langfuse.py stop
```

`stop` 保留数据卷。关闭上报时把根 `.env` 的 `LANGFUSE_ENABLED=false`，再通过 `app up -d --no-deps backend` 重新创建后端。本地聊天 journal/checkpoint 继续负责结果与恢复。

升级前备份所有 Langfuse 数据卷、本目录 `.env` 及项目根 `.env`；特别要保留 `SALT` 和 `ENCRYPTION_KEY`。不要在已有部署中删除凭据后重新 init。Compose 中版本固定了 Langfuse 发布版；MinIO 镜像沿用官方示例的 `cgr.dev/chainguard/minio`，它使用滚动标签，受限环境可先镜像到内部仓库并固定 digest。
