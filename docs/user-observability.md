# 用户侧观测与评测

登录后打开 `/observability`，或点击侧边栏「观测与评测」。普通用户无需 Langfuse 账号即可查看自己的记录。聊天回答下的「查看执行详情」也能进入对应 Trace。

## 页面与使用方式

| 入口 | 功能 | 实际数据来源 |
| --- | --- | --- |
| 总览 | 执行次数、成功率、平均耗时、记录的 Token、反馈、模型与工具用量 | PostgreSQL 中当前用户的执行与反馈记录；所有统计应用相同筛选条件 |
| 执行记录 | 按时间、问题/会话/Trace ID、状态、模型、智能体筛选；分页；返回原会话 | ExecutionTrace、Conversation、源用户消息 |
| 执行详情 | 最终回答、反馈、层级瀑布图、步骤输入/输出、Token、AI Judge、Score 投递状态 | 本地消息/事件/评分 + 后端受限查询 Langfuse Observations V2 |
| 评测结果 | 历史运行、最多三次运行对比、快照一致性提示、逐题步骤与评分、导出报告 | 现有 EvaluationRun、原始逐题指标、现有 Score outbox |
| 评测工作台 | 准备/导入/导出数据集、编辑基准、运行真实检索或已有语义评测 | 直接复用 EvaluationView 和现有评测 API |

执行详情中先选择可用的评分模型，再点击「评测此回答」。仅这个动作调用模型，会产生模型用量；打开或刷新页面不会自动评分。评分规则 `answer-v1` 包含切题程度、证据支持度、简短评语。没有记录到工具证据时，证据支持度保持未知，不生成这个 Score。评分模型来自当前用户原有模型配置，后端沿用现有权限校验与密钥解密流程。

评测结果页勾选运行进行比较；数据集快照或知识库不同会显示提示。进入运行后，点击某题「查看步骤」读取该题实际采集的 Langfuse 步骤与五项检索 Score。执行失败的样例会在列表和详情提示，避免把依赖故障导致的零分误认为检索算法退步。已有知识库评测界面也增加了跳转原生观测的链接。

## 代码入口与实现逻辑

| 文件 | 负责的逻辑 |
| --- | --- |
| `frontend/src/views/ObservabilityView.vue` | 四个页签、筛选、统计、分页、历史运行对比、URL 状态；复用评测编辑器；异步请求代次检查避免迟到响应覆盖当前页面 |
| `frontend/src/components/TraceTable.vue`、`TraceDetail.vue` | 执行列表和详情；并行于本地详情独立加载远端观测；反馈保存后刷新 Score；显式触发 Judge |
| `frontend/src/components/ObservationWaterfall.vue` | 按 parent ID 建树与耗时轴，选择步骤查看详情；本地事件与远端 Observation 分开标识、切换 |
| `frontend/src/components/EvaluationRunDetail.vue` | 逐题指标、观测、Score 与报告下载 |
| `frontend/src/utils/observability.js` | 空筛选清理、瀑布图父子排序及循环保护、缺失值格式化、快照对比、外链校验 |
| `frontend/src/styles/observability.css` | 与现有白色/青色工作区一致的卡片、表格和响应式布局；样式限于 `.ob-page` |
| `frontend/src/views/LayoutView.vue`、`router/index.js` | 用户侧导航和 `/observability` 路由 |
| `frontend/src/components/AgentTrace.vue`、`views/ChatView.vue` | 普通用户从单条回答的 local Trace ID 打开原生执行详情 |
| `frontend/src/components/MessageFeedback.vue`、`EvaluationObservability.vue` | 反馈保存事件、现有评测界面跳转原生观测 |
| `backend/server/routers/observability_router.py` | 认证入口；先查本地归属，再读远端；评测样例校验 run UUID + index 派生的确定性 Trace 关联 |
| `backend/services/user_observability.py` | 同一 owner-scoped SQL 基础查询派生列表/总览；只通过消息的精确 trace_id/stream_id 关联最终回答与反馈；查询自有 Score 和 Judge 结果 |
| `backend/services/observation_reader.py` | 服务端持有 Langfuse 凭据；查询指定 Trace + User + 时间范围的 V2 Observations；递归脱敏、字段白名单、内容/数量/响应大小限制和降级状态 |
| `backend/services/turn_judge.py` | 评分模型解析、评分输入构造、持久化租约/冷却/幂等、严格分数校验、结果落库与稳定 Score ID 入队 |
| `backend/storage/postgres/models_observability.py`、`manager.py` | 新增 `TurnJudgement` 表，保留最近成功结果与当前执行状态；注册启动建表 |
| `backend/services/evaluation_service.py` | 记录实际执行开始/结束时间；保存到原 EvaluationRun，保证远端步骤查询时间范围准确 |
| `backend/server/routers/evaluation_router.py` | 兼容扩展运行摘要：基准版本、Experiment/快照信息、执行状态、失败样例数 |
| `backend/server/main.py` | 注册新 router |
| `app/llm/client.py` | 为短生命周期评分客户端增加 `aclose()`，释放同步/异步连接 |
| `scripts/migrations/20261103_turn_judgements.sql` | 可重复执行的增量建表迁移 |
| `backend/tests/test_user_observability.py`、`frontend/tests/observability.test.mjs` | 权限隔离、筛选聚合、远端范围与脱敏、评分并发/幂等/失败恢复、时间持久化、瀑布图与快照对比测试 |
| `scripts/verify_user_observability_live.py` | 显式开启的本地合成数据端到端验收，检查真实 PostgreSQL、Langfuse Observation/Score/Experiment |

### 问答观测读取链路

```text
回答中的 local Trace ID / 用户执行列表
  → JWT 当前用户
  → ExecutionTrace.user_id 与 Conversation.user_id 双重归属检查
  → 精确关联已保存的最终回答、本地事件、反馈、Score、Judge
  → 从该本地记录取 observability_trace_id、采样标记和生命周期时间
  → 后端查询 Langfuse /api/public/v2/observations
     traceId + userId + fromStartTime + toStartTime
  → 校验返回行身份，白名单与脱敏投影
  → 前端按 parent_id 建树，在真实开始/结束时间上绘制瀑布图
```

浏览器不能提交任意远端 Trace ID 来枚举项目。所有读取要求登录，其他用户的执行/评测返回 404，远端请求在本地授权成功之后才发生。Langfuse 公钥/私钥只在服务端使用。

### 回答评分与反馈链路

```text
显式评分 → 验证拥有 Trace 且已完成/有最终回答 → 解析自有可用模型
  → TurnJudgement 行锁 + 唯一键 + 120 秒租约；60 秒冷却
  → 题目、最终回答、工具结果脱敏并限制长度
  → 90 秒内调用模型 → 严格验证有限数字 0..1 与非空评语
  → 同一事务保存结果、规则版本、模型、完成时间，并写 Score outbox
  → 现有后台投递 worker 重试，Langfuse 确认接收后标记 accepted

用户反馈 → 复用原 message feedback API
  → 更新同一条消息的反馈和稳定 Score ID
  → 刷新原生详情及总览；沿用同一投递 worker
```

成功的同 request UUID 重试直接返回已存结果；并发评分返回 409，冷却期间返回 429。无效输出、供应商故障不会生成分数，失败只存异常类型；进程中断后租约过期可重试。重新评分时保留上次成功结果，并明确标注其状态。Score ID 使用 `judge-{local_trace_id}-{metric}`，重新评分更新该分数，不重复新增。

### 评测关联链路

沿用既有「数据集 → 基准快照 → 运行」流程和 Experiment 发射逻辑。摘要 API 补齐关联字段供原生列表使用。逐题 API 先通过知识库归属检查运行，再验证 `trace_id == uuid5(run_id, str(item_index)).hex`，读取该用户的样例 Score 和远端步骤。新运行记录真实执行时间；历史运行原先只保存结束时刻，读取时对其查询起点回退一天，且始终保留精确 Trace/User 筛选。

## API

所有路径以 `/api/v1` 为前缀，均要求 Bearer JWT。

```text
GET  /observability/overview
GET  /observability/traces
GET  /observability/traces/{local_trace_id}
GET  /observability/traces/{local_trace_id}/observations
POST /observability/traces/{local_trace_id}/judge
     {"model_id":"现有模型配置 ID", "request_id":"UUID"}
GET  /observability/evaluations/{run_uuid}/items/{zero_based_index}
```

总览/列表共用 `window=7d|30d|all`、`q`、`status`、`model`、`agent`、`conversation_id` 筛选。列表默认每页 20 条，最大 100 条。空会话筛选不传 UUID 参数。

## 状态、边界与配置

- Langfuse 超时/不可用时，本地详情仍可查看。未关联、未采样、未启用、配置不可用、等待索引、响应过大均有独立状态。API 确认接收 Score 与远端完成索引分开表达。
- 每次远端读取超时 4 秒、不跟随重定向，解压后最多 4 MiB、最多显示 500 个步骤；超限提示截断。本地事件同样最多展示 500 条。工具统计最多最近 10,000 条调用事件，并提示范围。
- `LANGFUSE_ENABLED=true` 使用既有 Langfuse 配置。`EASYRAG_OBSERVABILITY_CONTENT_MODE=redacted` 允许展示脱敏的远端输入输出；`metadata_only` 时不返回远端输入输出。本地用户自己的已保存消息仍沿用应用权限可见。
- 输入输出使用纯文本，屏蔽系统消息、密钥/鉴权/密码等字段与已知敏感内容；元数据白名单过滤。Judge 只使用公开工具结果作为证据，不读取模型推理或角色 system prompt。
- 执行 Token 来自已记录的模型调用，不把缺失记录补成推测值。没有可靠定价时总费用显示未知；步骤上的 Langfuse 费用标注为估算。独立 Judge 的 Token/费用目前未采集，不混入原问答用量。
- Judge 是给定材料上的辅助评分，不能代替人工事实核验。RAGAs 仍由原基准配置决定，本次未更改全局模型、RAGAs 开关或评分阈值。
- 老数据没有 remote Trace 或细粒度步骤时无法追补历史观测。原始本地事件的状态是记录当时的快照，不等同于远端 span 现在仍在运行。

## 本地部署与验收

新表由启动 `create_all` 创建，也可显式应用 SQL 迁移；本次已应用并重建前后端。现有数据卷保留，部署前数据库和镜像已备份。复现生产构建可使用项目现有 Langfuse compose wrapper：

```powershell
python scripts/local_langfuse.py app build backend frontend
python scripts/local_langfuse.py app up -d --no-deps backend frontend
```

本地合成验收需先有 `verify_langfuse_quality_live.py` 产生的忽略目录内测试账号与知识库，并确保现有向量模型/Ollama 可用：

```powershell
python scripts/verify_user_observability_live.py --live --new-evaluation
# 仅在确实需要再次调用评分模型时加 --judge；会产生模型用量
python scripts/verify_user_observability_live.py --live --judge
# 验证从页面创建的某次自有运行
python scripts/verify_user_observability_live.py --live --run-id <run-uuid>
```

2026-11-03 实测：普通用户看到 21 个远端问答步骤、13 条本地事件；页面提交反馈并修改同一个 Score；页面 AI Judge 成功并写入两个远端 Score；从页面工作台运行原有基准，两道题、10 个样例 Score、Experiment 可查询，Recall/MRR 均 1.0；跨用户远端读取返回 404。首次评分供应商临时 500 正确留失败状态，页面重试成功。首轮评测因本机 Ollama 停止产生的失败运行保留，并另跑成功运行。重建/重启后反馈与 Judge 结果保留；聊天回答和原生 Trace 精确往返跳转通过，390px 视口下 document/body 宽度均为 390px。Python 定向回归 48 项、前端 Node 70 项通过，Vite 生产构建通过。

验收报告和截图保存在忽略目录 `tmp/user-observability-live/`，不包含账号密码；测试凭据仍在原忽略目录，不进入源码。
