# AgentState 管理重构规划 — 参照 Yuxi（xerrors/Yuxi）

> 日期：2026-09-02 ｜ 状态：规划 ｜ 前置阅读：`2026-08-21-deepagents-maturation-plan.md`
>
> 参照项目：https://github.com/xerrors/Yuxi （LangGraph v1 + langchain 1.x 中间件架构）

## 1. 为什么值得参照，以及根本差异

Yuxi 的 Agent 状态体系有三条支柱，正好对应 EasyRAG 当前的四个痛点：

| Yuxi 机制 | EasyRAG 现状 | 痛点 |
|---|---|---|
| `BaseState(AgentState)` + `Annotated[list, reducer]` 增量合并 | `app/graph/state.py` 一个 40+ 字段大杂烩 TypedDict，且只剩 `AGENT_MODE=single` 在用；dynamic/deep 用裸 messages | 状态定义与真实执行路径脱节 |
| `BaseContext` dataclass（Configurable 参数，auth metadata 过滤） | 请求上下文散在 3 个 ContextVar + 各处函数参数 | 权限/trace 传递靠快照重放，易漏 |
| `AgentStatePayload`（给前端的序列化契约）+ `SubAgentRunState` | 前端消费的 meta/steps/artifacts 结构散落在 chat_router 的 json.dumps 里 | 无契约，前后端字段漂移 |
| 中间件（summary/memory/skills/token_usage）挂 `create_agent` | 各功能直接写进 agent 构建函数与 router | 功能耦合，无法按 Agent 组装 |

**根本差异：版本代差。** Yuxi 基于 langchain≥1.3.9 + LangGraph v1 的 `create_agent`/middleware API（`langchain.agents.AgentState`、`@dynamic_prompt`、`@wrap_model_call` 等）。EasyRAG 钉死 langchain 0.3.26 / langgraph 0.x（`requirements.txt` 第 19-26 行明确"勿升 1.x"），`create_agent`/middleware 体系在 0.3 线不存在。已确认用户决策：**升级后直接对齐**，即先做版本升级，再按 Yuxi 模式重写状态管理。

### 1.1 升级可行性盘点（已核实）

好消息： EasyRAG 代码实际触碰的 langchain 表面非常小，全部导入只有——

```
langchain_core.tools        (4 处, StructuredTool)
langchain_core.messages     (2 处, AIMessage/HumanMessage/SystemMessage)
langchain_openai            (2 处, ChatOpenAI + base 私有 API)
langgraph.prebuilt          (3 处, create_react_agent)
langgraph.errors            (2 处, GraphRecursionError)
langgraph.graph             (1 处, StateGraph/END — 仅 single 管线)
```

`langchain-community/experimental/huggingface/deepseek/text-splitters` 在 `app/`、`backend/` 中**零引用**（requirements 里是死重）；RAG 检索用自研 + sentence_transformers，**llama_index 在代码中零引用**。真正需要适配的只有三类：

1. **`langgraph.prebuilt.create_react_agent`**：1.x 中被 `langchain.agents.create_agent` 取代（prebuilt 仍保留但签名变化——`prompt` 参数变为 `system_prompt`/middleware）。三个调用点（`deep/agent.py`、`deep/subagents.py`、`dynamic.py`）都要动，但恰好是本次重构要重写的位置。
2. **`DeepSeekChatOpenAI`（`deep/llm.py`）**：依赖 `langchain_openai.chat_models.base._convert_message_to_dict` 私有 API 回传 `reasoning_content`。1.x 的消息转换链路已重构（`_convert_from_v1_to_chat_completions` 等被移除）。方案：1.x 的 `ChatOpenAI` 支持 `extra_body` 与 `additional_kwargs` 透传，reasoning_content 回传改用 `model_kwargs={"extra_body": ...}` 或自定义 `BaseChatModel` 子类覆写公开的 `_generate`，不再碰私有函数。
3. **`StateGraph`（single 管线）**：langgraph 1.x 仍保留 `langgraph.graph.StateGraph`，基本兼容；但见 §4 该管线本身计划退役。

连带升级：`openai>=2.0`（langchain-openai 1.x 要求）、`langgraph>=1.0`、移除 `langchain-deepseek/community/experimental` 依赖行；torch/sentence-transformers 等与 langchain 无关，不受影响。

## 2. 目标架构：三层状态模型（对齐 Yuxi）

```
┌─────────────────────────────────────────────────────────────┐
│ L1 AgentState（图内状态，逐轮更新，Annotated reducer 合并）      │
│   BaseState: messages, todos, artifacts, token_usage         │
│   ChatState: subagent_runs, sources, steps_summary           │
├─────────────────────────────────────────────────────────────┤
│ L2 AgentContext（运行时配置，一次 Run 只读，dataclass 声明）     │
│   user/session/model/skills/KB授权/沙箱策略/递归上限            │
│   → 取代散装 ContextVars 成为主要权限与配置载体                  │
├─────────────────────────────────────────────────────────────┤
│ L3 RunPayload（对外契约：前端消费 + 落库 + 历史回放）             │
│   AgentStatePayload + SubAgentRunState 序列化协议              │
└─────────────────────────────────────────────────────────────┘
```

区分三者的判据（写进代码注释作为团队约定）：

- **进 L1**：节点/工具循环中会被读写、需要 reducer 合并、随 checkpoint 持久化的执行状态；
- **进 L2**：一次 Run 开始时确定、执行期只读的配置与权限（绝不放进 State——Yuxi 把 model/system_prompt 都放 Context 就是这个原因）；
- **进 L3**：跨进程边界（SSE/DB/前端）的序列化形态，独立于 L1 演进。

### 2.1 L1 — State 定义（`app/agents/state.py` 新建）

```python
from langchain.agents import AgentState          # 1.x 官方基础 State（middleware/types.py）：
                                                 # messages: Required[Annotated[list[AnyMessage], add_messages]]
from typing import Annotated, Literal, TypedDict

def merge_artifacts(existing, new): ...          # 按 id 去重合并（照抄 Yuxi 语义）

class BaseState(AgentState):
    artifacts: Annotated[list[dict], merge_artifacts]

class SubAgentRunState(TypedDict, total=False):  # 对齐 Yuxi chatbot/state.py
    id: str; run_id: str; subagent_type: str
    status: Literal["pending","running","completed","failed","skipped","cancelled"]
    description: str; error: str | None
    created_at: str; completed_at: str

def merge_subagent_runs(existing, new): ...      # 按 run_id 合并（照抄 Yuxi）

class ChatState(BaseState):
    subagent_runs: Annotated[list[SubAgentRunState], merge_subagent_runs]
    sources: Annotated[list[dict], merge_sources]     # kb/web 引用去重合并
```

关键收益：现在 `spawn_tasks` 的结果聚合（`aggregate_results` 拼文本回主 Agent）和前端任务面板状态（chat_router 里手工维护的 `taskPanel` 协议）可以统一为 `subagent_runs` reducer——**委派状态成为图状态的一部分**，reducer 自动处理并发写入，不再需要 planner 手工传 dict。

旧 `app/graph/state.py` 的处理：`AgentState` 退役，`AGENT_MODE=single` 一并下线（见 §5 阶段 0）。其中仍有价值的字段去向：`history/user_id/knowledge_base_ids` → L2 Context；`sources` → L1 ChatState；`steps/is_fallback` → L3 Payload；其余（`gap_rounds/query_decomposition` 等增强检索字段）本来就只被 enhanced_retriever 内部使用，不进 Agent 状态。

### 2.2 L2 — Context 显式化（`app/agents/context.py` 新建）

```python
@dataclass(kw_only=True)
class BaseContext:                       # 对齐 Yuxi BaseContext 语义
    thread_id: str                       # = conversation_id
    user_id: str
    run_id: str | None = None
    model_id: str = ""
    skill_ids: tuple[str, ...] = ()
    knowledge_base_ids: tuple[str, ...] = ()
    deep_research: bool = False
    # 权限字段带 auth metadata（Yuxi 的 filter_config_by_role 模式，
    # 序列化给前端/日志时按角色剔除）
    sandbox_policy: str = "default"

class ChatContext(BaseContext):
    image_data: str | None = None        # 大 payload 也放 Context（只读、不进 checkpoint）
    history_window: tuple = ()
```

与现有机制的关系：

- **替换而非并存**：`app/agents/events.py` 的 trace ContextVar、`app/skills/context.py` 的 skill ContextVar、`use_authorised_kb_ids` 的 KB ContextVar 内部实现保留（langchain 1.x 的 runtime.context 在 langgraph 0.x 工具线程里不可用的场景仍需 ContextVar），但**收敛为一个 `RequestContext` 门面**：`BaseContext` 是声明层，三个 ContextVar 是传播层，`snapshot_request_context()/run_with_request_context()` 机制原样复用。`use_skill_context`/`use_authorised_kb_ids` 改为从 `BaseContext` 构造，不再各自从请求参数散装拼装。
- 各 `run()`/`_run_deep()`/`_run_dynamic()` 的十余个同名参数（user_id/knowledge_base_ids/knowledge_catalog/...）收拢为一个 context 参数——这是本次重构对调用方代码量的最大削减。

### 2.3 L3 — RunPayload 契约（`app/agents/payload.py` 新建）

对齐 Yuxi `AgentStatePayload`，定义 SSE `done` 事件与 `messages.metadata_json` 的**唯一**序列化结构：

```python
class SubAgentRunPayload(TypedDict): ...   # 与 L1 SubAgentRunState 同构（status 枚举一致）
class AgentRunPayload(TypedDict):
    todos: list; artifacts: list; subagent_runs: list[SubAgentRunPayload]
    token_usage: dict | None; sources: list; steps: list
    intent: str; agent_mode: str; model_name: str; elapsed_seconds: float
```

- `chat_router` 三条路径落库的 `metadata_json` 与 `done` 事件都从 `to_payload(state)` 生成，消除三处手工 dict 的字段漂移（这正是之前"幽灵消息/前端渲染"类 bug 的温床）；
- 前端 `buildHistoryWorkItems`/`taskFromRun` 按 payload 协议消费，字段命名对齐后前端可删掉一层适配代码。

## 3. 中间件对位表（升级后可用的 1.x 能力）

Yuxi 的中间件清单 vs EasyRAG 现状，**不一次性照搬**，标记本期做/不做：

| Yuxi middleware | 对应 EasyRAG 现状 | 本期 |
|---|---|---|
| context_aware_prompt / context_based_model | `use_chat_model` ContextVar + 构建函数参数 | ✅ 换成 middleware，`ChatContext.model` 驱动 |
| SkillsMiddleware | `app/skills/` context + prompt 注入 | ✅ 迁移 |
| create_subagent_task_middleware | `task`/`spawn_tasks`/`revise_plan` 三个手工 StructuredTool | ⚠️ 保留现有工具实现（已含熔断/黑板/DAG），仅把委派状态写入改为 subagent_runs reducer；不搬 Yuxi 的 subagent 中间件 |
| summary middleware（100k 触发、L1/L2 压缩） | `get_compressed_history`（请求前压缩） | ❌ 本期不做；现有方案够用，列入后续 |
| TodoListMiddleware | taskPanel 的 todos 由前端拼 | ⚠️ 与 subagent_runs 一起整理 |
| TokenUsageMiddleware | 无 | ✅ 顺手加（一次模型调用的 usage 已有，缺聚合） |
| memory middleware | `app/memory/manager.py` | ❌ 不动 |
| ModelRetryMiddleware | LLM client 自带 max_retries | ❌ 不动 |

主 Agent 构建从 `create_react_agent(prompt=MAIN_SYSTEM_PROMPT, tools=[...])` 迁移到：

```python
agent = create_agent(                # langchain.agents.factory.create_agent（已核对 1.3.9 签名）
    model=load_chat_model(ctx.model_id),          # str | BaseChatModel
    tools=[...registry_tools, task_tool, spawn_tool, revise_tool],
    system_prompt=MAIN_SYSTEM_PROMPT,             # 取代旧 prompt= 参数
    middleware=[context_model_mw, skills_mw, token_usage_mw, ...],
    state_schema=ChatState,
    context_schema=ChatContext,
    checkpointer=PostgresSaver(...),   # §4
)
```

dynamic Agent / SubAgent 同构迁移，`deep/agent.py`、`deep/subagents.py`、`dynamic.py` 三处的缓存策略（`_main_agent_cache`）保持：**模型相关部分移入 middleware 后，图本身与模型解耦，缓存键反而更干净**（现在换模型即缓存失效的隐患消除）。

## 4. 会话状态持久化（用户选定范围）

现状：`_plans`（PlanState）、熔断计数、Blackboard 全部进程内存，重启即失；Agent 执行中途崩溃无断点恢复。

方案（分两档）：

1. **Checkpointer（L1 状态落库）**：langgraph 官方 `langgraph-checkpoint-postgres`（已核实最新 3.1.2，含 `AsyncPostgresSaver`，适配 langgraph 1.x），接现有 Postgres（pgvector 镜像已就位，`checkpoints` 表由 `saver.setup()` 自动建）。价值排序：主 Agent 中断恢复 > 子 Agent。本期只给主 Agent 与 dynamic Agent 挂 checkpointer，SubAgent 保持无状态（其失败由主 Agent 重试/熔断兜底，落库反而拖慢 spawn 并发）。
2. **编排会话状态（路径 3 状态迁移）**：`_plans` → Postgres `agent_plans` 表（key=thread_id，jsonb 存 PlanState，随 spawn/revise 写入）；熔断计数 → Redis（现有实例，TTL 600s 天然对齐 `TASK_BREAKER_TTL_S`）；Blackboard 保持内存（生命周期=单次 spawn，落库无意义）。

多 worker 前提说明：当前单进程部署，持久化不是正确性问题而是**重启恢复 + 审计**问题；若未来 uvicorn 多 worker，熔断改 Redis 反而成为正确性前提。

## 5. 分阶段实施计划

| 阶段 | 内容 | 验收 | 预估 |
|---|---|---|---|
| 0 决策：single 退役 | `AGENT_MODE=single` 与 `app/graph/workflow.py` 固定管线（9 节点）下线删除，`.env` 默认 auto | 全测试绿；`AgentState` 可安全废弃 | 0.5 天 |
| 1 版本升级 | langchain 1.x / langgraph 1.x / langchain-openai 1.x / openai 2.x；`DeepSeekChatOpenAI` 改公开 API；`create_react_agent` → `create_agent` 机械迁移（暂不改状态） | 现有全部 pytest 绿；dynamic/deep 手工冒烟 | 2-3 天 |
| 2 L2 Context | `app/agents/context.py`；三个 ContextVar 收敛为门面；各 run() 签名收拢 | 类型检查 + 现有测试改造通过 | 2 天 |
| 3 L1 State | `app/agents/state.py`（BaseState/ChatState/reducers）；主 Agent 挂 state_schema；spawn 结果写 subagent_runs | 委派场景前端面板数据来自 reducer | 2 天 |
| 4 L3 Payload | `app/agents/payload.py`；三条路径落库/done 事件统一走 to_payload | 同轮对话 done 事件与历史回放字段一致（对拍测试） | 1.5 天 |
| 5 持久化 | PostgresSaver（主 Agent）、agent_plans 表、熔断→Redis | 重启后 revise_plan 可续、熔断跨重启生效 | 2 天 |
| 6 前端适配 | ChatView 消费 payload 协议；删适配层代码 | 回归通过 | 1 天 |

依赖：1 → 2 → (3, 4) → 5 → 6；阶段 0 独立可先行。**风险最高的是阶段 1**（DeepSeek reasoning_content 回传、langchain-openai 私有 API），建议升级单独开分支，`pip freeze` 前后各跑一遍 RAG 评估管线（`docs/RAG_EVALUATION.md`）确认检索链路无回归——它不依赖 langchain，是天然的对照组。

## 6. 风险与开放问题

1. **`langchain-openai` 1.x 的 reasoning_content 处理**是最大单点：`deep/llm.py` 现在覆写私有函数，1.x 需要重写为公开扩展点。若 1.x 原生支持 DeepSeek reasoning 模型（langchain 1.x 对 reasoning content 有官方字段），直接删除自定义类。
2. **`create_agent` 的 state_schema 与工具线程**：现有 `registry.invoke` 在 ThreadPoolExecutor 跑工具，1.x 的 LangGraph 仍以 messages 增量更新，reducer 并发安全——但 spawn_tasks 的层内并发写入 subagent_runs 必须经 reducer 而非直接改 dict，需在迁移时写并发单测。
3. **ContextVar 与 runtime.context 双轨期**：工具函数内部读 skill/KB 权限仍走 ContextVar（langgraph 工具线程拿不到 runtime.context），双轨要在 §5 阶段 2 一次收干净，避免长期两套真相。
4. **开放问题**：`AGENT_MAX_ITERATIONS=20` 与 1.x 默认 recursion limit 的对齐；PostgresSaver 的 async 版本与现有 `asyncpg` session 管理的集成方式（`get_session` 工厂复用还是 saver 自建池）；Yuxi 的 thread/subagent child_thread_id 体系是否引入（涉及前端委派树回放 URL，本期建议不引入，保持 run_id 单层）。
