# Langfuse 接入实施与验证

日期：2026-10-18。

## 已实现

- admission 预分配本地 trace 和可选 Langfuse trace 关联；采样决策随 ChatStream 持久化；恢复尝试保留前序 stream 关联。
- producer 统一执行根节点；线程池、子 Agent、并行任务、实际工具和模型调用共享上下文；已有通用 OTel 接口保持兼容。
- LangChain 仅模型 callback、自研同步/异步/流式 LLMClient generation；JSON 解析重试保留错误阶段。
- 明确区分取消、中断和错误，保留自定义可取消 HTTP transport；停止不等待 flush。
- MCP 离线、`isError`、超时和错误字符串保留结构化 outcome；根终态依据业务事务。
- 本地 ExecutionTrace 支持预建后补全和终态同步；取消时没有最终回答也能定位；旧库迁移可重复执行。
- 管理员本人执行的关联查询和前端入口；普通用户与跨 owner 访问被阻止。
- 默认关闭；独立 TracerProvider、字段白名单、正文过滤、导出前脱敏、限时 shutdown；标题/记忆后台工作脱离聊天 trace。
- 配置模板、配置/回滚说明、README 导航与可提交的合成验证脚本。

## 本地验证结果

最终相关 Python 回归 **229 项通过**，覆盖原有模型、工具、权限、checkpoint、SSE 恢复与新增 Langfuse 场景。根目录测试按既有约定只在本地保留。

- 实际 Langfuse SDK 5.0.2 + 内存 exporter：8 个 observation、1 个根节点，父子关系与 user/session 归属正确；无正文泄漏、无外部请求。
- Langfuse 开启时的真实 localhost OpenAI/LangChain SDK I/O：等待响应头、reasoning 流停顿场景能取消；单次模型调用不重试，取消不标为 ERROR。
- 导出器阻塞时，业务取消不等待上报；未结束子调用不会被补写成成功。
- localhost OTLP HTTP 接收端：实际 SDK 的导出地址、v4 ingestion header、压缩 protobuf、trace/parent ID 和内容过滤通过；返回 401 也不重复或中断业务。
- PostgreSQL 17 隔离实例：断线重放、幂等准入、取消与提交竞态、保存失败、过期 producer、JWT 与 owner 权限、无回答 trace 关联、预建记录更新、旧库迁移重复执行通过。
- 前端 Node 回归 **17 项通过**；Vite 生产构建通过。现有知识库页面大 chunk 提示仍存在，与本次观测接入无关。
- 合成页面浏览器验证：普通视图无观测入口；管理员入口能显示精确执行关联、受控跳转和“远端接收与索引状态未确认”；停止后仍可访问关联。使用合成 API，不表示远端 Langfuse 页面已经验收。
- `pip check` 无依赖冲突；编译检查通过。

PostgreSQL 验证使用独立端口 55439 和临时数据目录，没有修改业务数据库。Docker 因本机遗留 IPC 文件启动失败，其目录清理被自动审批拒绝，因此改用官方 PostgreSQL 二进制进行隔离测试，没有重置 Docker 或删除数据卷。

## 本地部署与真实候选验收（2026-10-18）

Docker 恢复后，已部署 Langfuse Web/worker 4.63.0、PostgreSQL 17.11、Redis 7.4、ClickHouse 25.12 和独立对象存储。控制台仅监听本机 3000，项目初始化、健康检查与 key 认证通过；根 `.env` 已开启采样率 1.0 的 metadata_only 观测，密钥仅保存在忽略文件中。

部署前完成业务 PostgreSQL 备份，备份位于忽略的 `volumes/backups/`。增量迁移已应用，没有删除业务数据或重置数据卷。独立栈及包含本机 override 的业务 Compose 配置检查通过。

真实模型使用本机 Ollama 的 **qwen3.5:latest**。候选后端绑定 18000、使用独立 checkpoint 路径和临时模型环境，未更改正式默认模型配置。验收脚本直接通过 API 请求并查询 Langfuse 已索引 observations，不使用假模型代替。

| 项目 | 实际结果 |
| --- | --- |
| 主 Agent → coding-agent → calculator → 模型 → 结果提交 | 同一 trace，11 个 observation、1 个逻辑根节点；5 个 generation、1 个 tool、2 个 agent、3 个 span |
| 父子层级 | 每个子节点可回溯至 execution.chat；子模型与 calculator 位于 coding-agent 下 |
| 去重 | 5 次模型调用对应 5 个独立 modelCallId，无继承回调造成的重复记录 |
| 身份关联 | user/session/execution ID 一致，实际模型名与起止时间正确 |
| 断线、刷新查询、重连、幂等重发 | 沿用同一次执行与 trace |
| 主动取消 | 真实 generation 已进入执行；约 2.14 秒进入 cancelled；3 个节点均无 ERROR，root 与模型状态可定位 |
| 内容过滤 | metadata_only 下模型和工具输入/输出正文为空 |
| 访问控制 | 合成普通账号请求管理员观测接口返回 403 |
| 模型错误 | 原默认网关调用的 OpenAIConnectionError 已被 Langfuse 收录；根状态按最终保存的业务结果记录 |

真实验收还发现并修复两项问题：

1. 选中 Skill 后，内置 task/spawn_tasks/revise_plan 被 Skill 门控误拦截。主 Agent 现在仅对构建时显式提供的内置编排工具实例豁免；按对象身份匹配，同名插件不继承豁免，子智能体实际工具仍通过注册表授权。
2. LangChain 自动继承主图 callback，子图另建 callback 会使子模型重复记录。现在同一次执行复用同一个线程安全 handler，以模型 run UUID 记录 modelCallId，结束与并发处理保持幂等。

新增保护回归和观测相关检查共 **60 项通过**，前端执行过程与断线恢复 **17 项通过**，前后端镜像构建及容器内 pip check 通过。根目录测试继续只在本地保留。

### 已知外部限制

当前默认远端模型网关的 HTTPS 证书已过期，宿主机与容器均校验失败。因此成功链路验收使用本地 Qwen；没有关闭证书校验，也没有擅自替换正式默认网关。恢复该远端模型需网关维护方更新有效证书，或由用户另行选择模型配置。

本地 Qwen 在当前 OpenAI 兼容流协议中未返回完整 usage，观测如实标记 unknown；未据此伪造 token 数或计费结果。自定义模型定价需另行配置。

### 分支同步与正式服务更新

候选验收通过后，功能提交 `a53eed0` 已原子推送至 main/home。随后重新创建正式 backend/frontend 与 Langfuse Web/worker，前后端健康检查通过，7 个原有依赖服务继续运行，6 个 Langfuse 服务就绪。候选容器已停止并移除，没有删除其挂载的业务数据或 checkpoint。

正式 backend 进程确认 `initialized`、采样率 1.0、metadata_only。重启后通过真实 API 请求再次验证：新执行的 4 个 observation 已在 Langfuse 收录，其中模型证书连接错误为 ERROR，根节点按已保存的业务结果保持 completed。前端 `/health` 与 Langfuse `/api/public/health` 均返回成功。默认模型网关和原有其他环境配置保持原样。

Langfuse 控制台已在浏览器实际打开完整成功 trace，显示主 Agent、coding-agent、calculator、5 次模型调用和结果提交层级，跳转路径已核验。

本次提交排除了原有 `.gitignore` 修改、根目录 tests、tmp、实际密钥和备份。部署命令、控制台账号及凭据位置见 [本地部署说明](../deploy/langfuse/README.md)。
