# EasyRAG 执行沙箱

本实现把工具权限判定与进程隔离分开。`ToolRegistry.invoke` 负责策略和审计；独立的 `sandbox-runner` 服务独占 Docker 控制权，后端只通过内部 HTTP 接口提交固定类型的任务。runner 不挂载后端的 `.env`、源码或 `./volumes`，也不连接 `easyrag` 数据网络。它在同一 Docker 主机上持有 Docker socket，因此生产环境若面对不受信任的公网用户，建议将 runner 放在独立主机并使用独立的 Docker daemon；跨主机调用应配置 HTTPS 的 `SANDBOX_RUNNER_URL`。

## 已实现

- `run_python`：Python 3.11 标准库环境，单次容器，默认断网。代码最多 256 KiB；可传最多三个当前会话附件 ID，文件放在 `/work/inputs`，总计 10 MiB；生成文件写在 `/output`，允许 txt/md/csv/json/html/pdf/docx/xlsx，每个最多 5 MiB。执行超时 45 秒，runner 外层 60 秒强制销毁。
- runner 同时最多执行 4 个 Python 任务并维持 8 个 MCP 会话；超额请求返回 429。
- MCP stdio runner 模式：固定配置为 `filesystem`、`postgres` 和 `rail12306`。runner 在隔离容器中保持 MCP 会话，通过私有 API 中继 `list_tools` / `call_tool`。filesystem 容器只读挂载专用 Docker volume，默认无网络；postgres 容器只接入数据库内部网络并使用独立只读账号；12306 容器只接入出口内部网络，通过仅允许 `kyfw.12306.cn`、`search.12306.cn`、`www.12306.cn` 的 HTTPS CONNECT 代理出网。runner 连接失败不会回退到后端本地 stdio。
- 权限：`config/sandbox_policy.json` 现为 `enforce`，`run_python` 需要 `SANDBOX_EXEC_ENABLED=true`、runner 地址和令牌、已认证的用户及会话。策略文件丢失或无效时，`run_python` 不出现。
- 文件归属：输入通过 `artifact_service.read_file(user_id, conversation_id, id)` 读取，输出经 `create_generated_file` 存到相同用户和会话的产物前缀。

## 启用

1. 启动 Docker Desktop 的 Linux Engine。
2. 在 `.env` 设置一个随机令牌：`SANDBOX_RUNNER_TOKEN=<随机长字符串>`。要开放 Python 工具，再设 `SANDBOX_EXEC_ENABLED=true`。
3. 构建执行镜像：

   ```powershell
   docker compose --profile sandbox-images build sandbox-python-image sandbox-mcp-filesystem-image sandbox-mcp-postgres-image sandbox-mcp-12306-image
   docker compose --profile sandbox up -d --build sandbox-runner backend
   ```

4. 如需把已有文件系统 MCP 切到 runner，设置 `MCP_FILESYSTEM_SANDBOX_PROFILE=filesystem` 并重启后端。runner 使用独立的只读 volume；不会自动读取原来的 `./volumes/workspace`。要导入旧文件，可先在 Compose 的 backend 服务中临时挂载 `sandbox-workspace:/app/sandbox-workspace`，再由管理员复制需要的文件；完成后移除临时挂载。不要把整个 `./volumes` 或 Docker socket 交给执行容器。
5. 如需把 Postgres MCP 切到 runner，先创建只授予所需 schema `SELECT` 权限的独立数据库账号，并在 `.env` 设 `SANDBOX_POSTGRES_READONLY_URL=postgresql://<只读用户>:<密码>@postgres:5432/<数据库>` 和 `MCP_POSTGRES_SANDBOX_PROFILE=postgres`，然后重启 runner 与后端。连接串未配置时 runner 拒绝启动该 MCP 会话。
6. 如需把 12306 MCP 切到 runner，设 `MCP_12306_SANDBOX_PROFILE=rail12306` 并重启后端。代理镜像随 `--profile sandbox` 构建并启动。固定包版本为 `12306-mcp@0.3.10`，使用 Node 24 的环境代理支持；若上游包将来改用其他域名，应先审核用途，再修改代理白名单并验证。参见 [Node.js 环境代理文档](https://nodejs.org/download/release/latest-v24.x/docs/api/cli.html#node_use_env_proxy1)。

未启用各 `MCP_*_SANDBOX_PROFILE` 时，现有 MCP 仍按原配置由后端启动。system-health MCP 默认关闭，尚未迁移；它需要细分多个内部服务的访问权限。迁移前不要将它视为已被进程隔离。

需要禁止所有后端本地 stdio MCP 时，设置 `SANDBOX_REQUIRE_MCP_RUNNER=true`。未配置 runner profile 的服务会明确报错，包括尚未迁移的 system-health。

## 验收

```powershell
docker compose --profile sandbox --profile sandbox-images config --quiet
python -m pytest -q tests/test_sandbox_runner.py tests/test_tool_sandbox.py tests/test_mcp_config_persistence.py
```

Docker 可用后，从 Agent 调用 `run_python`：测试 `print(1+1)`、读当前会话附件、写 `/output/result.csv`、尝试访问网络及后端 `.env`、无限循环、跨会话附件 ID。期望前 3 项成功，后 4 项被拒绝或按时终止。执行容器不应出现在 `easyrag` 网络，也不应包含后端环境变量、源码或 Docker socket。

### 本机验收记录（2026-10-19）

- 构建了 Python、三个 MCP、runner、出口代理及后端镜像；Compose 配置解析通过，runner 健康检查通过。
- runner 无令牌请求返回 403；后端镜像通过私有网络携令牌调用 Python runner 成功，`print(6*7)` 返回 `42`。
- Python 容器读取 `/work/inputs/data.txt` 并生成 `/output/result.csv`；测试中无法访问后端 `.env`、数据库口令或直接连接外网。超时任务由 runner 终止，临时容器已清理。
- filesystem MCP 能启动、列目录并关闭；12306 MCP 能启动，执行容器只连接 `sandbox-egress`，直接外连被阻止；代理仅允许预设的 12306 HTTPS 主机。
- 已创建专用 PostgreSQL 只读账号。postgres MCP 实际查询返回该账号、`transaction_read_only=on`、`public` schema 无 `CREATE` 权限；后端镜像通过 runner 成功执行 `SELECT 7`。
- 完整 EasyRAG 服务栈已启动，后端与前端健康检查均返回 200；后端注册了 `run_python` 和三个 runner MCP 服务。三个 MCP 执行容器分别仅连接 `none`、`sandbox-db`、`sandbox-egress` 网络。
- 通过后端 `ToolRegistry.invoke` 完成附件输入、Python 执行、生成文件写入 RustFS、按会话读取及跨会话拒绝的整条链路；测试使用临时身份和会话 ID，结束时清理了产物。真实登录用户通过 Agent 对话触发工具的交互仍需在 UI 验收。
- 相关测试 44 项通过。
