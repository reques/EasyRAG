# 沙箱机制设计方案（工具调用权限沙箱 + MCP 进程 Docker 隔离）

> 日期：2026-09-02 ｜ 状态：规划 ｜ 范围：`app/tools/*`、`app/skills/*`、`backend/server/routers/mcp_router.py`、`mcp_servers.json`、`docker-compose.yml`

## 1. 背景与目标

EasyRAG 的工具体系已经长出两条"会执行外部代码 / 触碰外部资源"的路径：

1. **MCP stdio 子进程**：`MCPServerHandle._connect_stdio()` 用宿主机权限直接拉起子进程。典型配置如 `npx -y @modelcontextprotocol/server-filesystem ./volumes` —— `npx -y` 意味着**运行时从 npm 拉取任意版本的包并执行**，代码路径完全不受项目控制。
2. **进程内工具调用**：`ToolRegistry.invoke()` 里所有工具 `fn`（内置工具、MCP 桥接工具）都跑在后端进程的工作线程上，目前只受 Skill 白名单约束，没有能力模型、没有出网/文件限制、没有审计。

本方案目标（对应此前确认的两个需求）：

- **A. 工具调用的通用权限沙箱**：在 registry 层建立"能力 + 策略 + 审计"三层机制，对所有工具调用统一生效，纯代码改造、不依赖 Docker。
- **B. MCP / 外部工具进程的 Docker 隔离**：把不可信的 stdio 子进程关进一次性容器，复用现有 Docker Desktop 基础设施；HTTP 型 MCP 与出网流量走默认拒绝 + 白名单。

非目标（本期不做，但设计需预留扩展点）：LLM 生成代码的执行沙箱（`code_executor` 工具）、多租户配额。两者都可以直接复用 B 层的 `SandboxRunner` 抽象。

## 2. 现状梳理

### 2.1 工具执行路径

```
Agent(dynamic / deep / graph nodes)
  └─ ToolRegistry.invoke(name, **kwargs)
       ├─ get_active_skill_context().allows_tool(name)   # 唯一的权限检查：Skill 白名单
       ├─ tool.is_available()                            # check_fn 自检
       └─ ThreadPoolExecutor 包裹超时/重试 → tool.fn(**kwargs)   # 进程内线程执行
```

关键事实：

- `ToolDefinition` 已有 `metadata` 字段（阶段 2 工具发现用），是挂"能力标签"的天然位置；
- `invoke()` 已有统一的事件埋点（`tool_start / tool_end / tool_error`，`app/agents/events.py`），是挂"审计"的天然位置；
- Skill 权限通过 `contextvars` 传递请求级上下文 —— 权限沙箱复用同一模式即可拿到 user/session 上下文。

### 2.2 MCP 路径

- 配置来自 `mcp_servers.json`（手工编辑，`mcp_router` 只有 start/stop/查询接口，**没有写配置的 API**，这是一道现成的防线，后续加 API 时要守住）；
- 权限两层：server 级 `allowed_tools` 白名单 + Worker 侧 tool_names 白名单；
- stdio 子进程继承后端进程全部权限：环境变量（含 `app/core/config.py` 里的各 API Key）、整个用户目录、Docker 网络。

### 2.3 威胁模型

| # | 威胁 | 现状 | 危害 |
|---|------|------|------|
| T1 | 恶意/带漏洞的 MCP server 代码（npm/pip 供应链） | 无任何隔离 | 读 `.env`、源码、SSH key；横向到宿主机 |
| T2 | 工具入参被 prompt injection 操纵（如 read_file 逃逸根目录、抓取内网 URL） | 依赖各 server 自己的校验 | 越权读写；SSRF |
| T3 | SSRF 到 Docker 网络内服务：Postgres/Redis/MinIO 弱口令（`easyrag_secret` 等）、Milvus 19530 | 无出网限制 | 知识库数据整库外泄/篡改 |
| T4 | 工具死循环 / 内存膨胀（线程杀不掉，超时只是"放弃等待"） | 只有超时兜底 | 后端服务降级 |
| T5 | 无审计：出事后无法回答"谁在哪个会话调了什么工具、参数是什么" | 事件流存在但非常驻落盘 | 无法追溯 |
| T6 | 未来加 `code_executor` 时直接继承以上所有问题 | — | 需在架构上预留 |

## 3. 总体设计：两层沙箱

```
                     ┌─────────────────────────────────────────┐
 Agent 请求上下文      │  A 层：权限沙箱（进程内，策略判定）        │
 user/session/skills ─▶  ToolRegistry.invoke                    │
                     │    ├─ RequestContext (contextvars)      │
                     │    ├─ PolicyEngine.check(cap, args)     │
                     │    └─ AuditLog (JSONL 落盘)              │
                     └───────────────┬─────────────────────────┘
                                     │ 放行
                     ┌───────────────┴─────────────────────────┐
                     │  B 层：进程沙箱（Docker，执行隔离）        │
                     │  ├─ builtin 工具 → 进程内（受 A 层约束）   │
                     │  ├─ untrusted MCP stdio → SandboxRunner  │
                     │  │    docker run --rm -i --network none  │
                     │  └─ MCP http / 出网工具 → EgressProxy     │
                     │         （默认拒绝 + 域名白名单）          │
                     └─────────────────────────────────────────┘
```

设计原则：

1. **A 层先行**：纯代码、可单测、对所有工具（含未来任何执行后端）统一生效；B 层只解决"代码真的被执行"的问题。
2. **默认拒绝，显式放行**：新注册的工具/server 默认无 `net`/`fs`/`exec` 能力，必须在其 `ToolDefinition.metadata` 或 server 配置里声明。
3. **信任分级**：`builtin`（进程内）→ `mcp-managed`（进程内但走白名单）→ `mcp-untrusted`（必须进容器）。`npx -y` / `pip install` 动态拉包的 server 一律归为 untrusted。
4. **不改既有契约**：`ToolDefinition`、`MCPManager.start/stop`、`mcp_router` 接口保持兼容，新能力全部是加字段、加可选配置。

## 4. A 层设计：权限沙箱

### 4.1 能力模型（capabilities）

每个工具在 `ToolDefinition.metadata["capabilities"]` 声明所需能力，未声明视为 `["none"]`（最干净的一类）：

| 能力标签 | 含义 | 现有工具示例 |
|----------|------|--------------|
| `net.out` | 主动出网 | web_search、http 型 MCP、filesystem 之外的多数 MCP |
| `fs.read` / `fs.write` | 读/写宿主或挂载文件 | filesystem MCP 的各工具 |
| `proc.exec` | 拉起子进程/执行代码 | stdio MCP 桥接工具 |
| `kb.read` | 读知识库数据 | kb_search |
| `none` | 纯计算 | calculator、datetime_tool、text_tool |

MCP 桥接工具注册时自动继承 server 配置声明的能力（见 5.2），内置工具在各自 `register()` 处显式声明。`app/tools/registry.py` 提供 `TOOL_CAPABILITIES` 只读视图，`to_status()` / 前端 MCP 管理页可展示。

### 4.2 请求上下文

新增 `app/tools/sandbox/context.py`，复用 Skill 的 contextvars 模式：

```python
@dataclass(frozen=True)
class SandboxContext:
    user_id: str | None
    session_id: str | None
    agent_mode: str          # dynamic / deep / graph...
    skill_ids: tuple[str, ...]
```

在 `agent_service` / `chat_router` 入口处随 skill context 一起注入；`invoke()` 里传给 PolicyEngine。DeepAgents 的 task 委派链路需确认 contextvars 跨线程传播（子 Agent 线程池处显式 copy_context）。

### 4.3 策略引擎

新文件 `app/tools/sandbox/policy.py`，策略文件 `sandbox_policy.json`（项目根，风格对齐 `mcp_servers.json`，路径可被环境变量 `SANDBOX_POLICY_FILE` 覆盖）：

```json
{
  "defaults": { "mode": "enforce", "deny_capabilities": ["proc.exec"] },
  "network": {
    "allow_domains": ["api.tavily.com", "*.anthropic.com"],
    "deny_cidrs": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"],
    "note": "deny_cidrs 覆盖 docker 网段与内网，专治 T3"
  },
  "rules": [
    { "tool": "mcp_filesystem_*", "capabilities": ["fs.read"],
      "args": { "path": { "prefix": "./volumes" } } },
    { "context": { "agent_mode": "deep" }, "deny": ["net.out"] },
    { "tool": "web_search", "allow": true }
  ],
  "limits": { "max_output_chars": 20000, "per_session_tool_calls": 200 }
}
```

`PolicyEngine.check(tool_def, kwargs, ctx) -> Decision`，在 `invoke()` 中位于 skill 白名单之后、`fn` 执行之前；返回 `allow / deny`，deny 时抛 `ToolExecutionError`（复用现有异常，LLM 可读其消息自我修正）。规则匹配按顺序、首条命中生效，支持工具名 glob。**第一阶段只做 `enforce` 与 `audit-only` 两种 mode**，灰度用后者。

### 4.4 SSRF 出网守门

`app/tools/sandbox/net_guard.py`：提供 `guarded_urlopen(url)` / `assert_url_allowed(url)`——解析域名后校验 IP 不落在 `deny_cidrs`（防 DNS rebinding：解析与连接用同一校验结果），域名不在 `allow_domains` 即拒绝。适用范围：**任何接受调用方指定 URL 的工具**（目前没有，未来加 `web_fetch`/`code` 类工具时必须接入）与 http 型 MCP 的连接地址校验。注：`web_search_tool` 现在只 POST 固定 Tavily 端点，接入守门收益很小，可不做——当前真正的 SSRF 面在 MCP server 内部自行发起的请求，那要靠 B 层 `--network none` / egress 代理兜底，net_guard 是进程内工具的前置防线，两层互补。

### 4.5 审计日志

新文件 `app/tools/sandbox/audit.py`：追加式 JSONL（`logs/tool_audit.jsonl`，按天轮转），每条记录 ts、user/session、tool、capabilities、参数摘要（复用 `_args_digest` 的 400 字截断）、决策结果、耗时。直接订阅 `app/agents/events.py` 的 tool 事件 + PolicyEngine 的 deny 事件，不侵入各工具。

### 4.6 资源限制（尽力而为）

线程杀不掉是 Python 既有事实，A 层只能做：输出大小截断（`max_output_chars`）、按会话的工具调用次数配额（超限即 deny，防失控循环刷工具）。真正的 CPU/内存硬限制放 B 层容器。

## 5. B 层设计：MCP 进程 Docker 隔离

### 5.1 SandboxRunner 抽象

新文件 `app/tools/sandbox/docker_runner.py`：

```python
class SandboxRunner:
    def open_stdio_session(self, spec: SandboxSpec) -> StdioSession: ...
```

`StdioSession` 提供 `send(line)/recv_line()/close()`，内部是 `docker run -i --rm <flags> IMAGE CMD...` 子进程的 stdin/stdout 管道（用 docker SDK 或 subprocess 均可，推荐 subprocess 依赖少）。对 MCP 而言，**transport 仍是 stdio**，`_connect_stdio()` 只是把"本地拉起子进程"换成"容器内拉起"，`ClientSession`、常驻 loop 线程、防 GC 等既有设计全部不动——这是本方案改动面小的关键。

容器 flag 基线：

```
--rm                     # 一次性，退出即销毁
-i                       # stdio 桥接需要
--network none           # 默认断网；需出网的 server 接 egress 网络（5.3）
--memory 512m --cpus 1 --pids-limit 128
--read-only --tmpfs /tmp:size=64m
--cap-drop ALL --security-opt no-new-privileges
--user 10001:10001
-v <仅白名单目录>:ro/rw   # 如 ./volumes:/workspace:rw
-w /workspace
```

### 5.2 配置扩展（`mcp_servers.json`）

`MCPServerConfig` 新增字段（缺省全部保持现行为，向后兼容）：

```json
{
  "name": "filesystem",
  "transport": "stdio",
  "sandbox": {
    "mode": "docker",                  // "host"(默认,现状) | "docker"
    "image": "easyrag/mcp-sandbox:1",
    "mounts": [{ "host": "./volumes", "container": "/workspace", "rw": true }],
    "network": "none"                  // none | restricted(走 egress 代理)
  },
  "capabilities": ["fs.read", "fs.write"]   // 供 A 层策略与桥接工具 metadata 继承
}
```

加载校验规则：`mode=host` 且命令含 `npx`/`pip install`/`curl|sh` 等动态拉包特征 → 启动时告警并在 `to_status()` 里置 `risk` 字段；正式 enforce 后直接拒绝以 host 模式运行 untrusted server。

### 5.3 镜像与出网

- 基础镜像 `deploy/sandbox/Dockerfile`：python3.11 + node20 + 少量常用 CLI，构建纳入现有 compose 项目（`docker compose build`）；**常用 MCP server 版本固化烤进镜像**（`npm i -g @modelcontextprotocol/server-filesystem@x.y.z`），容器内不再 `npx -y` 动态拉包，从根上消掉 T1 的供应链面。
- 需出网的 server：接入自建 `easyrag-egress` 网络，唯一出口是一个小型代理容器（squid 或 20 行 Python 的转发白名单），域名白名单复用 `sandbox_policy.json` 的 `network.allow_domains`。

### 5.4 Windows 主机注意事项

后端跑在 Windows、Docker Desktop（WSL2 虚拟机）上，这是已确认的部署形态：

- `docker` CLI 在 Windows PATH 可用（Docker Desktop 安装后即有），subprocess 调用无障碍；
- bind mount Windows 路径走 Docker Desktop 的路径翻译，性能一般，但 `./volumes` 这类小文件场景可接受；跨平台路径规范化统一放 `SandboxSpec` 构造处处理；
- 资源限制（memory/pids）依赖 Docker Desktop 的 Linux 引擎，无 WSL 集成问题；
- 若 Docker Desktop 未运行：`SandboxRunner` 启动 server 时抛出明确错误（新异常 `SandboxUnavailableError`），`MCPManager.start` 将其转为 `to_status().error`，**绝不静默回退到 host 模式**——回退只允许通过显式配置 `mode: host` 发生。

### 5.5 存活与生命周期

`MCPServerHandle.running` 需感知容器退出：`docker run` 子进程意外死亡时（读管道 EOF），bridge `fn` 抛 `ToolExecutionError`，并支持**按需重启**（首次调用遇到 dead session → 重新拉起容器，至多一次）。这弥补 stdio 桥接现状"宿主进程挂了连接静默失效"的问题。

## 6. 分阶段实施计划

| 阶段 | 内容 | 交付 | 预估 |
|------|------|------|------|
| A1 能力与上下文 | capabilities 声明、SandboxContext 注入、DeepAgents 跨线程传播验证 | 代码 + 单测 | 1 天 |
| A2 策略引擎 | PolicyEngine + `sandbox_policy.json` + invoke 接线（先 audit-only） | 代码 + 策略样例 | 1.5 天 |
| A3 审计与配额 | audit JSONL、输出截断、会话配额；前端事件流加 sandbox 决策展示（可选） | 代码 | 1 天 |
| A4 SSRF 守门 | net_guard 工具函数 + http 型 MCP 连接地址校验；上线 deny_cidrs enforce | 代码 + 用例 | 0.5 天 |
| B1 SandboxRunner | docker_runner + StdioSession + 单测（用 busybox 假 server） | 代码 | 1.5 天 |
| B2 MCP 接线 | MCPServerConfig.sandbox、`_connect_stdio` 切换、容器死亡重启、mcp_router 状态透出 risk 字段 | 代码 | 1.5 天 |
| B3 镜像与默认策略 | deploy/sandbox/Dockerfile、烤入 filesystem server、示例配置切 docker 模式 | 镜像 + 文档 | 1 天 |
| B4（可选）出网代理 | easyrag-egress 网络 + 代理容器白名单 | 服务 | 1 天 |

依赖关系：A1→A2→(A3, A4)；B1→B2→B3→B4。A 与 B 可并行推进，A 层独立完成即可将整体姿态从"无防护"提升到"有策略有审计"。

验收要点（每阶段合入前）：

- A2 灰度：audit-only 模式下跑全部现有测试（`pytest`）+ 人工发起含 filesystem/web_search 的对话，核对 JSONL 决策与预期一致后才切 enforce；
- B2：容器内 demo server 跑通 `mcp_router` 全套 start/stop/tools 调用；`kill` 容器后下一次工具调用能自愈重启；Docker 未启动时报错清晰且不回退；
- 全量回归：`demo` server 保持 host 模式作为兼容性证据，`filesystem` server 切 docker 模式。

## 7. 改动文件清单

新增：`app/tools/sandbox/{__init__,context,policy,net_guard,audit,docker_runner,spec}.py`、`sandbox_policy.json`、`deploy/sandbox/Dockerfile`、`tests/test_sandbox_*.py`。
修改：`app/tools/registry.py`（invoke 接 PolicyEngine/audit）、`app/tools/mcp/config.py` + `manager.py`（sandbox 字段、runner 接线、http transport 连接校验、状态透出）、`app/services/agent_service.py` 与 `backend/server/routers/chat_router.py`（SandboxContext 注入，与 `use_skill_context` 同点位）、`app/core/config.py`（`SANDBOX_ENABLED`、策略文件路径等开关）、`mcp_servers.json`（示例迁移）、`docker-compose.yml`（egress 服务，B4）。

## 8. 风险与开放问题

1. **误杀风险**：enforce 一刀切会打断现有对话流程 → 用 audit-only 灰度 + `defaults.mode` 全局开关兜底；
2. **Docker Desktop 依赖**：单机部署下后端可用性被 Docker 绑定 → B 层只影响 sandbox 模式的 server，host 模式工具不受影响；明确报错文案；
3. **容器 stdio 延迟**：每行 JSON-RPC 多一层管道，对高频小调用有毫秒级开销 → demo server 基准测试验证，必要时批量管道优化；
4. **开放问题**：DeepAgents 子 Agent 线程池的 contextvars 传播需在 A1 实测确认；`mcp_router` 未来若加"UI 写配置"API，server 命令必须白名单化（当前只能手编 json 是隐性防线，不要无意丢掉）；Windows 下 `--user 10001` 与 bind mount 权限映射需实测一次。
