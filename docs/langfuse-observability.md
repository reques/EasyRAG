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

`metadata_only` 不上传问题、历史、system prompt、reasoning、工具参数原文、检索正文、反馈原因、图片和文件内容，仅保留允许的关联、检索文档 ID、排名和指标。`redacted` 显式开启经过筛选的问题、最终回答、检索片段和反馈原因；不上传任意对象、模型输入历史或完整工具参数。内容过滤后再经过 SDK 导出前脱敏；密钥、JWT、URL、邮箱等会替换，异常只记录类型与阶段。

## 问答与检索过程

新执行的 `execution.chat` 根节点以数据库中的用户消息作为问题。`redacted` 下，根 input 为 `{question, contexts}`，output 为成功提交的最终回答。取消、保存失败不会把未提交的回答写成最终结果。根节点的 contexts 汇总本次实际检索片段，供 observation evaluator 显式映射；这不是完整历史或完整模型 prompt。

基础检索各后端生成 `retrieval.basic`；增强检索生成 `retrieval.enhanced`，并显示查询分解、并行召回、融合、重排、answerability、覆盖检查、聚类与缺口补充节点。检索节点记录知识库范围、Top K、结果数，以及允许的 chunk/file/KB ID、排名和分数。增强检索线程继承执行上下文，辅助模型和子检索仍归入同一 trace。

正文默认仍不上传。要查看问答和检索正文，在后端运行环境设置 `EASYRAG_OBSERVABILITY_CONTENT_MODE=redacted` 并重新创建 backend。结构化内容只允许显式字段，默认每个 input/output 的文本总预算为 2000 字符，最多 20 条片段；长文本会截断。不能把截断后的上下文当作完整证据评判模型，也不会因为启用内容模式而采集图片或 reasoning。

## 用户反馈

已保存的助手消息下方有“有帮助”“需改进”按钮，后者可填写原因。评分支持修改，刷新后恢复。普通用户通过应用提交反馈，不需要 Langfuse 账户；只有消息所属用户可读取或修改反馈，用户消息不能评分。

- `GET /api/v1/chat/messages/{message_id}/feedback` 读取反馈。
- `PUT /api/v1/chat/messages/{message_id}/feedback` 保存 `{rating: "like" | "dislike", reason?: string}`；reason 最长 2000 字符。
- 本地 `message_feedback` 为权威记录。只有消息 metadata 中的精确 stream 属于同一用户/会话且已采样，才建立远端关联；历史记录和未采样执行不猜测 trace。
- `user-feedback` 为 BOOLEAN Score，赞=1，踩=0。固定 Score ID 使修改覆盖同一评分，避免重复计数。`metadata_only` 下不上传原因。
- `observability_scores` 在同一事务中记录待发送评分。后台每 10 秒尝试投递，失败退避至最多 300 秒；重启后继续处理，租约避免多进程同时领取。上传过程中修改评分会保留待处理的新修订。
- delivery_state 的 `pending` 表示等待投递/重试，`accepted` 表示评分 API 已接收，不代表远端已完成索引；`not_linked` 表示本地已保存但无可用 trace。

在 Langfuse 按 `user-feedback=0` 筛选，结合对应问题、回答及检索节点检查原因。可把有代表性的失败样例加入后续回归集。

## 现有评测与 Experiment

普通“运行评估”和“运行基准”均保留原有检索、可选回答生成与 Ragas 流程。启用 Langfuse 后，每条真实执行的评测用例生成一条独立 trace，根名为 `evaluation.case`。原生 OTel Experiment 属性将这些根与子节点关联为同一个 Experiment；不会为了同步结果重复调用检索器、模型或 Ragas。

数据继续由本地评测系统管理，使用 Langfuse 支持的 local-data experiment 模式：Dataset 关联 ID 是知识库 ID 加完整实际用例快照的 SHA-256；相同快照在不同运行中保持相同 ID，修改题目或标注会产生新身份。item ID 同样稳定，Experiment ID 与本地 EvaluationRun ID 一致。不会复制或覆盖远端托管 Dataset。元数据包含应用版本、模型、指标版本及已知的检索配置、数据集/基准版本。

每条用例的 `observability` 返回 `trace_id`、`observation_id`、`experiment_id` 和 `item_id`。现有 HitRate/MRR/Recall/Precision/nDCG 以 `retrieval.*` Score 关联到同一个实际根 observation，负样本使用 `retrieval.expected_miss`；可用的逐条 Ragas 值写为 `ragas.*`。缺失、NaN、检索失败和相关生成失败不会伪造零分。评分队列和本地结果在同一事务提交，随后后台投递。

评测结果与历史详情中可打开“在 Langfuse 比较实验”。链接使用配置的 Web base URL/project ID，采用当前 SDK 5 的 `/project/{id}/experiments/results?baseline={experiment_id}` 路径。界面显示关联用例数量，并明确提示远端接收与索引未确认。关闭或缺失 SDK 时本地评测正常返回，关联状态显示不可用。

比较时选择相同快照、相同评测规则的 baseline/candidate，逐条检查分数和输出；修改模型或检索参数后创建新运行。只有基准启用 Faithfulness 才会沿现有流程生成回答，单纯检索评测的 output 是检索结果，不应当作完整 Agent 回答质量。完整 Agent 多步任务 runner 不在本轮范围。

新表迁移见 `scripts/migrations/20261025_langfuse_quality.sql`；开发启动的 `init_db` 同样创建缺失表。上线前应用迁移、重建并重新创建 backend/frontend。Langfuse 控制台的 Judge 连接、评分标准与在线采样规则仍需按实际任务配置。

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
