# Langfuse 执行观测

本项目支持可选 Langfuse 观测，默认关闭。PostgreSQL、持久 SSE journal 和 checkpoint 继续决定任务状态；Langfuse 不参与任务调度、恢复和结果保存。

## 配置

安装当前 requirements 后，设置后端的运行环境；不要把实际密钥放入 Git、前端构建参数或智能体输入：

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=<project-public-key>
LANGFUSE_SECRET_KEY=<project-secret-key>
LANGFUSE_BASE_URL=https://<selected-langfuse-endpoint>
LANGFUSE_TRACING_ENVIRONMENT=development
LANGFUSE_SAMPLE_RATE=1.0
EASYRAG_OBSERVABILITY_CONTENT_MODE=metadata_only
EASYRAG_OBSERVABILITY_WEB_BASE_URL=https://<browser-accessible-endpoint>
EASYRAG_OBSERVABILITY_PROJECT_ID=<project-id>
```

Python SDK 固定为 5.0.2，需要 Python ≥3.11，以及 Langfuse Cloud 或自托管服务 ≥4.59.0。服务地址必须明确配置，空值不会隐式选择 Cloud 区域。[官方兼容要求](https://langfuse.com/docs/compatibility)。

Compose 的 backend 已读取根目录 `.env`。容器访问同机服务时，`localhost` 指向 backend 容器；独立容器应使用共享 Docker 网络地址，宿主机服务可用 `host.docker.internal`。必要时补充 NO_PROXY。Web base URL 是浏览器地址，可与上报地址不同。

增加依赖后需要重建镜像，仅更改环境配置需要重新创建 backend：

```powershell
docker compose build backend
docker compose up -d --no-deps backend
```

先应用 `scripts/migrations/20261016_langfuse_trace_links.sql`，再部署新代码。当前开发启动生命周期也会执行相同的幂等增量 DDL。迁移只加 nullable 字段和唯一索引；历史记录保持无观测关联，不猜测、不回填云端 trace。

## 一条 trace 对应什么

- 每个新 ChatStream 对应一次执行；ID 在 admission 时固定，模型尚未开始也能关联。
- 同一请求幂等重发、SSE 断线、浏览器刷新沿用原执行与 trace，不增加执行根节点。
- 服务进程中断后从 checkpoint 恢复是新的执行尝试，有新 trace，记录前次 stream/trace 的关联。
- Agent、子智能体、工具及工具内部辅助模型调用归入同一执行 trace。LangChain 模型使用仅模型回调，自研 LLMClient 使用 generation scope，避免与完整官方 CallbackHandler 双重计数。
- 嵌套图复用同一次执行的模型回调，以 modelCallId 标识实际调用，避免继承回调重复记录。内置编排工具由主 Agent 显式提供，子智能体实际工具仍经过注册表与 Skill 权限校验。
- 模型、工具失败保留在子 observation；根状态以最终业务提交结果为准。主动取消有独立 cancelled 状态，不计作模型异常。关闭连接并不能保证供应商撤销远端计算或计费。
- 标题生成与记忆评估清理聊天观测上下文，首期不跟踪；它们不会延长或污染已结束的聊天 trace。

推荐在 Langfuse 按 `user_id`、`session_id`、metadata.executionId、agentId、agentMode、modelProfileId 筛选；generation.model 是实际模型名。工具每次 invocation 独立 observation，重试 attempt 另有子节点；不能把两个层级同时当作工具逻辑调用次数。

## 排查入口

管理员在聊天执行过程里点击“查看观测”，可看到 execution ID 和关联状态，配置 Web URL 与 project ID 后可跳转 Langfuse。运行中、失败和取消时也支持定位。

后端 `GET /api/v1/chat/streams/{stream_id}/observability` 同时要求管理员身份与该执行的 owner 权限；管理员身份不会授予其他用户聊天的访问权。普通用户只能看到原有本地执行信息。Langfuse 项目权限仍需在 Langfuse 单独管理，不公开分享 trace。

“已关联”仅表示本地关联和 SDK 初始化，不代表服务端已经接收或建立索引。界面明确显示远端可见性未确认。实际部署版本的 Web 路径需要用真实 trace 核验；当前导航采用 `/project/{project_id}/traces/{trace_id}`。

## 内容与失败处理

`metadata_only` 不上传问题、历史、system prompt、reasoning、工具参数原文、检索正文、图片和文件内容，仅保留允许的关联和指标。`redacted` 可显式开启有限的纯文本输出诊断，仍不上传任意对象、模型输入历史或完整工具参数。内容过滤后再经过 SDK 导出前脱敏；密钥、JWT、URL、邮箱等会替换，异常只记录类型与阶段。

usage 缺失或取消前未返回时标为 unknown，不虚构 0；推理输出与可见正文首 token 分开记录。自定义网关无法确认价格时不提供成本承诺。

SDK 缺失、配置不完整、初始化失败、服务离线都不会改变聊天逻辑；导出异步批量执行，有限队列与超时。请求和停止按钮不调用 flush；退出时限时关闭导出器。进程硬退出可能丢失内存中未上报的 observations，首期没有持久 outbox，也不承诺每个模型调用都能离线补发。现有本地事件并不等同于完整模型调用审计。

降低采样率会整次执行一起抽样，不能保证错误任务都被采样。删除本地会话不自动删除 Langfuse 数据，应独立配置保留与清理策略。

## 验证与关闭

无需外部地址即可验证实际 SDK 的树形关系、用户/会话归属、跨线程父子关系、取消级别与内容过滤：

```powershell
.\.venv\Scripts\python.exe scripts/verify_langfuse.py
```

该脚本使用合成数据和内存 exporter，不向远端发送数据。真实环境还需用合成账号验证一次普通回复、工具、子智能体、模型错误、主动停止、断网重连和刷新，检查各次执行的 ID 与 Langfuse 可见数据。不要把本地验证报告当成远端接收证明。

关闭时设置 `LANGFUSE_ENABLED=false` 并重新创建 backend；本地任务与 checkpoint 继续工作。回滚代码时保留新增 nullable 列，无需删除业务数据或 volumes。

完整设计与验收矩阵见 [接入规划](plans/2026-10-16-langfuse-integration.md)。可选本地自托管栈、凭据初始化与真实验收步骤见 [本地部署](../deploy/langfuse/README.md)；准备了部署文件不代表已启动或通过远端验收，实际进度见 [实施记录](langfuse-implementation-verification.md)。
