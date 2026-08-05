# 阶段 1：Agent 内核重构 — 设计规格

> 日期：2026-08-04 | 状态：已批准 | 来源：brainstorming 澄清结果

## 背景

EasyRAG 当前是一个"能用的 RAG 应用"——`app/graph/workflow.py` 是一条写死的 LangGraph 静态 DAG（意图识别 → 检索/工具 → 生成 → 校验），所有请求走同一条路径。本阶段将其升级为"可编排的 Agent 内核"：LLM 自主决定思考/调工具/观察的循环（ReAct），工具即插件，记忆分层。

## 已确认的关键决策（brainstorming 澄清）

1. **ReAct 实现载体**：保留 LangGraph 依赖，StateGraph 从静态 DAG 改为含循环子图。理由：现有 8 个节点都是 LangGraph 风格（收 state 返回 partial state），LangGraph 原生支持条件边循环（已有 answer_validation → answer_generation 重试循环先例），脱离重写的代价远大于收益。
2. **与现有快速路径的关系**：并存。简单问答（chitchat/tool_use/knowledge_qa）走现有快速路径；复杂任务（complex_task 或低置信度）进 ReAct 循环子图。两路径共用 AgentState、工具注册表、fallback_handler。
3. **模型策略**：先用现有 DeepSeek 主模型跑通，留好分级配置接口（fast tier），后续再优化成本。

## 架构

```
用户请求
   │
   ▼
intent_recognition（保留，新增复杂度/置信度判断）
   │
   ├─ 简单（chitchat / tool_use / knowledge_qa）
   │     → 现有快速路径（静态 DAG，不变）
   │
   └─ 复杂（complex_task 或低置信度）
         → ReAct 循环子图（新增）
               ┌─────────────────────────┐
               │ agent_reasoning (LLM)   │◄──┐
               │   思考→选工具/给答案     │   │
               └────────┬────────────────┘   │
              ┌─────────┴─────────┐          │
              ▼                   ▼          │
        tool_execution      final_answer ────┘（observation 回填继续循环）
        (复用现有节点)      → answer_validation（复用）→ END
```

**ReAct 是 StateGraph 内的循环子图，不是独立系统。**

## 三个子系统

### A. 工具插件化（`app/tools/`）

- `ToolDefinition` 新增 `check_fn: Callable[[], bool]` — 工具自检（如 web_search 检查 TAVILY_API_KEY 已配置）。`check_fn` 返回 False 的工具不出现在 `to_llm_schema()` 和 `to_react_prompt()` 里，LLM 看不到不可用工具。
- 新增 `discover_tools()` — 扫描 `app/tools/` 下所有模块，发现带 `TOOL` 全局变量（ToolDefinition 实例）的模块自动注册。替换 `_build_default_registry` 的硬编码注册。新工具 = 放一个模块进去，不改任何现有代码。
- 新增 `to_react_prompt()` — 生成 ReAct reasoning prompt 用的工具描述文本（name + description + args 说明）。
- 现有 4 个工具（calculator/datetime/text_tool/web_search）改造为插件格式：各自模块导出 `TOOL = ToolDefinition(...)` 和 `check_fn`。

### B. ReAct 循环（StateGraph 新增节点 + 循环边）

**新节点 `agent_reasoning`**：
- 输入：query + history + `state["observations"]`（过往的 action/observation 序列）+ 工具描述（`to_react_prompt()`）
- LLM 输出 JSON：`{"thought": "...", "action": {"type": "tool", "tool_name": "...", "args": {...}} 或 {"type": "final_answer", "answer": "..."}}`
- LLM 决定调工具 → 写 `state["pending_tool"]`，路由到 `tool_execution`
- LLM 决定完成 → 写 `state["draft_answer"]`，路由到 `answer_validation`
- 非法 JSON → 记为失败 observation（"你上次输出格式错误"），让 LLM 自我修正；连续 3 次 → fallback_handler
- 达 `AGENT_MAX_ITERATIONS` → 强制基于现有 observations 生成 final_answer

**复用节点 `tool_execution`**：执行后把结果追加到 `state["observations"]`（而非只写 `tool_result`），新增循环边 `tool_execution → agent_reasoning`。

**分流逻辑（`intent_recognition` 节点修改）**：complex_task 或 `intent_confidence < 0.6` 时置 `state["use_react"] = True`，router 据此进 ReAct 子图。

**AgentState 新增字段**：`observations: List[Dict]`、`pending_tool`、`use_react: bool`、`react_iterations: int`。

### C. 结构化记忆（新增 `app/memory/`）

- **工作记忆**：现有 AgentState，不动。
- **情景记忆**：`conversations` 表加 `summary` 字段（Text, nullable）。`chat_service` 新增 `maybe_update_summary(session, conv_id)`：每 10 轮消息触发一次增量摘要（旧 summary + 新增消息 → 新 summary）。`prepare_context` / `agent.run` 注入历史时，若 summary 存在则用 `summary + 最近 N 轮` 替代全部历史（避免长对话爆 context）。
- **语义记忆**：新增 `user_facts` 表（id/user_id/fact/source_conversation_id/created_at）。本期只搭存储 + 注入骨架：`get_user_facts(user_id)` 查询 + `prepare_context` 注入 system prompt；事实提取用规则触发（用户说"记住/我喜欢/我是"时 LLM 提取存入），LLM 自动判断留后续。
- DB 迁移：`ALTER TABLE conversations ADD COLUMN summary TEXT`、`CREATE TABLE user_facts (...)`，沿用现有的 `init_db()` create_all + 手动 ALTER 模式。

### D. 模型分级接口（配置层）

- `app/core/config.py` 新增 `LLM_FAST_MODEL` / `LLM_FAST_BASE_URL` / `LLM_FAST_API_KEY`（Optional，默认 None → fallback 主模型）
- `app/llm/client.py` 的 `get_llm_client()` 加 `tier: str = "main"` 参数（`"main" | "fast"`），fast tier 用 LLM_FAST_* 配置（未配置时用主模型）
- 本期所有调用点用 `main`，接口留好供后续成本优化

## 数据流（ReAct 循环内）

```
agent_reasoning (第 N 轮)
  输入: query + history + observations[0..N-1] + 工具描述
  LLM 输出: {thought, action}
       │
       ├─ action.type = "tool"
       │     → pending_tool 写入 state
       │     → tool_execution 执行
       │     → observations.append({tool, args, result, thought})
       │     → 回到 agent_reasoning (第 N+1 轮)
       │
       └─ action.type = "final_answer"
             → draft_answer = action.answer
             → answer_validation（复用现有校验节点）
             → END
```

## 错误处理

- `agent_reasoning` LLM 返回非法 JSON → 记为失败 observation，LLM 自我修正；连续 3 次 → fallback_handler
- `tool_execution` 失败 → observation 记录错误信息，LLM 决定换工具或放弃
- 达 `AGENT_MAX_ITERATIONS` → 强制 LLM 基于现有 observations 给 final_answer

## 测试策略

- 工具插件化：discover_tools 扫到新工具自动注册；check_fn=False 的工具不出现在 schema
- ReAct 循环：
  - 单工具（"1+1 等于几"）→ 1 轮 reasoning + calculator + final_answer
  - 多步（"查民法典第10条，计算条号数字和"）→ 检索 → 观察 → 计算 → 完成
  - 步数耗尽 → 强制回答
  - 非法 JSON → 自我修正
- 记忆：会话摘要 10 轮触发；user_facts 存储/注入
- 回归：现有快速路径 4 类意图（chitchat/tool_use/knowledge_qa/天气）不受影响

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| ReAct 每步调 LLM 成本高 | 本期快速路径并存，复杂任务才进 ReAct；留 fast tier 接口 |
| LangGraph 循环图调试难 | steps 审计字段记录每轮 thought/action，前端后续可展示 |
| 记忆注入改变 prompt 影响现有行为 | summary/facts 注入是可关的配置项，默认开启但可回退 |
| 非法 JSON 死循环 | 连续 3 次失败强制 fallback，单轮超时保护 |

## 不做的事（YAGNI）

- 多 Agent 协作（阶段 2）
- MCP 协议接入（阶段 2）
- 知识进化/对话沉淀（阶段 3）
- LLM 自动判断事实提取（本期用规则触发，LLM 判断留后续）
- fast tier 的实际接入（本期只留接口）
