# 阶段 1：Agent 内核重构 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 EasyRAG 从静态 DAG 升级为可编排 Agent 内核——工具插件化 + ReAct 循环子图 + 结构化记忆 + 模型分级接口。

**Architecture:** 保留 LangGraph，StateGraph 内新增 ReAct 循环子图（agent_reasoning ↔ tool_execution 循环），与现有快速路径并存；工具层加自检 + 自动发现；记忆分三层（工作/情景/语义）；模型分级留 fast tier 接口。

**Tech Stack:** Python 3.11 · LangGraph StateGraph · SQLAlchemy 2.0 async · FastAPI · PostgreSQL · 现有 DeepSeek LLM 客户端。

**Spec:** `docs/specs/2026-08-04-agent-core-react-design.md`（已批准）

## Global Constraints

- 项目**没有 pytest 测试套件**——验证用 ad-hoc 脚本（`scripts/verify_*.py`，hermes-verify 模式），不用 TDD 的 RED-GREEN 循环；每个任务的"验证"= 写/跑 verify 脚本通过。
- 跑项目脚本一律用 `D:/Anaconda3/envs/stage1-agent/python.exe`，不用 conda run/activate。
- 后端改动后重启验证：`taskkill` 旧 uvicorn → 重新起 → HTTP 实测（Windows WatchFiles reload 不可靠，不能依赖 --reload）。
- 现有 4 类意图快速路径（chitchat/tool_use/knowledge_qa/web_search 天气）必须回归不受影响。
- 每次变更同步 PROGRESS.md，完成提交 lx 分支并推送。
- ReAct 与快速路径**并存**：complex_task 或 intent_confidence < 0.6 进 ReAct，其余走现有静态 DAG，两路径共用 AgentState / 工具注册表 / fallback_handler。

---

### Task 1: 工具插件化

**Files:**
- Modify: `app/tools/registry.py`（ToolDefinition 加 check_fn、新增 discover_tools、to_react_prompt）
- Modify: `app/tools/calculator.py` `datetime_tool.py` `text_tool.py` `web_search_tool.py`（改造为插件格式，导出 TOOL + check_fn）
- Test: `scripts/verify_tool_plugin.py`

**Interfaces:**
- Produces: `ToolDefinition(check_fn: Optional[Callable[[], bool]])`；`discover_tools() -> ToolRegistry`（扫 `app/tools/` 自动注册带 `TOOL` 全局变量的模块）；`registry.to_react_prompt() -> str`；`get_tool_registry()` 内部改调 `discover_tools()`。

- [ ] **Step 1: ToolDefinition 加 check_fn + invoke 前自检**

`registry.py` 的 `ToolDefinition` dataclass 加字段：
```python
check_fn: Optional[Callable[[], bool]] = None  # 工具可用性自检, None=总是可用

def is_available(self) -> bool:
    return self.check_fn() if self.check_fn else True
```
`invoke()` 开头加：
```python
if not tool.is_available():
    raise ToolExecutionError(f"Tool '{name}' is not available (check_fn failed)")
```

- [ ] **Step 2: discover_tools 自动发现 + to_react_prompt**

`registry.py` 末尾加：
```python
def to_react_prompt(self) -> str:
    """生成 ReAct reasoning prompt 用的工具描述文本（仅含可用工具）。"""
    lines = []
    for t in self._tools.values():
        if not t.is_available():
            continue
        args = ", ".join(
            f"{k}: {v[0]}" for k, v in t.arg_schema.items()
        ) or "无参数"
        lines.append(f"- {t.name}: {t.description}（参数: {args}）")
    return "\n".join(lines) or "（无可用工具）"

def discover_tools() -> "ToolRegistry":
    """扫描 app/tools/ 下所有模块, 注册带 TOOL 全局变量的 ToolDefinition。"""
    import importlib, pkgutil, app.tools as tools_pkg
    reg = ToolRegistry()
    for info in pkgutil.iter_modules(tools_pkg.__path__):
        if info.name in ("registry", "__init__"):
            continue
        mod = importlib.import_module(f"app.tools.{info.name}")
        tool = getattr(mod, "TOOL", None)
        if isinstance(tool, ToolDefinition):
            reg.register(tool)
    return reg
```
`_build_default_registry` 改为 `return discover_tools()`（删除 4 个硬编码 register 调用）。`to_llm_schema()` 和 `list_names()` 加 `is_available()` 过滤。

- [ ] **Step 3: 4 个现有工具改造为插件格式**

以 calculator 为例（其余 3 个同构）：
```python
# app/tools/calculator.py 顶部保留现有计算函数, 文件末尾加:
from app.tools.registry import ToolDefinition

def _check() -> bool:
    return True  # calculator 无外部依赖, 总是可用

TOOL = ToolDefinition(
    name="calculator",
    description="数学计算：加减乘除、幂、开方等表达式求值",
    fn=<现有计算函数>,
    arg_schema={"expression": ("string", "要求值的数学表达式", True)},
    check_fn=_check,
)
```
web_search_tool 的 `check_fn` 检查 `cfg.TAVILY_API_KEY` 非空；datetime_tool / text_tool 恒 True。**注意**：保留各模块现有的函数签名不变（nodes.py 的 tool_execution 仍按名调用）。

- [ ] **Step 4: 验证 discover + 自检 + 现有工具回归**

写 `scripts/verify_tool_plugin.py`：
```python
# 检查: discover_tools 注册 4 个工具; web_search 在无 TAVILY_API_KEY 时 is_available=False;
# to_react_prompt 含 calculator; invoke 未配置工具抛 ToolExecutionError;
# 现有 4 工具各自 invoke 正常（calculator 1+1=2 等）
```
跑：`D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_tool_plugin.py` 全绿。

- [ ] **Step 5: Commit**

```bash
git add app/tools/ scripts/verify_tool_plugin.py
git commit -m "feat: 工具层插件化——check_fn 自检 + discover_tools 自动发现 + to_react_prompt"
```

---

### Task 2: ReAct 循环子图

**Files:**
- Modify: `app/graph/state.py`（新增 observations/pending_tool/use_react/react_iterations 字段）
- Modify: `app/graph/nodes.py`（新增 agent_reasoning 节点、intent_recognition 加分流、tool_execution 写 observations）
- Modify: `app/graph/router.py`（新增 route_after_reasoning、route_after_intent 加 ReAct 分支）
- Modify: `app/graph/workflow.py`（注册新节点 + 循环边）
- Modify: `app/prompts/templates.py`（新增 REACT_REASONING prompt）
- Test: `scripts/verify_react_loop.py`

**Interfaces:**
- Consumes: Task 1 的 `to_react_prompt()`、`AGENT_MAX_ITERATIONS`（`app/core/config.py` 已存在）
- Produces: `agent_reasoning(state) -> dict`（写 pending_tool 或 draft_answer）；`route_after_reasoning(state) -> str`；AgentState 新字段。

- [ ] **Step 1: AgentState 加 ReAct 字段**

`app/graph/state.py` 的 AgentState 加：
```python
# ── ReAct 循环 ──────────────────────────────────────────────
use_react: bool                       # 是否走 ReAct 子图
observations: List[Dict[str, Any]]    # [{thought, tool, args, result}]
pending_tool: Optional[Dict[str, Any]]  # agent_reasoning 选中的待执行工具
react_iterations: int                 # 当前 ReAct 轮数
```

- [ ] **Step 2: REACT_REASONING prompt**

`app/prompts/templates.py` 加：
```python
REACT_REASONING = PromptTemplate(
    """你是一个采用 ReAct（推理+行动）模式的智能体。根据用户问题、对话历史和过往观察，决定下一步行动。

可用工具:
{tools}

过往观察（按时间顺序）:
{observations}

用户问题: {query}

规则:
1. 先思考（thought），再决定行动（action）
2. 如果需要调用工具获取信息，action.type 设为 "tool" 并给出 tool_name 和 args
3. 如果已有足够信息回答，action.type 设为 "final_answer" 并给出完整答案
4. 只输出合法 JSON，不要任何其他文字

输出格式（二选一）:
{{"thought": "...", "action": {{"type": "tool", "tool_name": "...", "args": {{...}}}}}}
{{"thought": "...", "action": {{"type": "final_answer", "answer": "..."}}}}
"""
)
```

- [ ] **Step 3: agent_reasoning 节点**

`app/graph/nodes.py` 加：
```python
def agent_reasoning(state):
    """ReAct 推理节点: LLM 决定下一步是调工具还是给最终答案。"""
    query = state["query"]
    observations = state.get("observations") or []
    iterations = state.get("react_iterations", 0)
    max_iter = cfg.AGENT_MAX_ITERATIONS
    logger.info("[agent_reasoning] iter=%d/%d obs=%d", iterations, max_iter, len(observations))

    # 步数耗尽 → 强制基于现有观察给答案
    if iterations >= max_iter:
        obs_text = "\n".join(str(o.get("result", "")) for o in observations) or "（无有效观察）"
        return {
            "draft_answer": f"基于已有信息：{obs_text[:500]}",
            "steps": _append_step(state, "agent_reasoning -> max iterations, forced answer"),
        }

    client = get_llm_client()
    registry = get_tool_registry()
    obs_text = "\n".join(
        f"{i+1}. 思考: {o.get('thought','')} | 工具: {o.get('tool','')} | 结果: {str(o.get('result',''))[:200]}"
        for i, o in enumerate(observations)
    ) or "（暂无观察）"
    prompt = REACT_REASONING.format(
        tools=registry.to_react_prompt(),
        observations=obs_text,
        query=query,
    )
    try:
        data = client.chat_json_sync([{"role": "user", "content": prompt}])
        action = data.get("action") or {}
        thought = str(data.get("thought", ""))
        if action.get("type") == "final_answer":
            return {
                "draft_answer": str(action.get("answer", "")),
                "react_iterations": iterations + 1,
                "steps": _append_step(state, f"agent_reasoning iter{iterations} -> final_answer"),
            }
        # tool 调用
        tool_name = action.get("tool_name")
        if tool_name not in registry.list_names():
            raise ValueError(f"unknown tool {tool_name}")
        new_obs = list(observations)
        return {
            "pending_tool": {"tool_name": tool_name, "args": action.get("args") or {}, "thought": thought},
            "observations": new_obs,
            "react_iterations": iterations + 1,
            "steps": _append_step(state, f"agent_reasoning iter{iterations} -> tool:{tool_name}"),
        }
    except Exception as exc:
        logger.warning("[agent_reasoning] failed: %s", exc)
        new_obs = list(observations)
        new_obs.append({"thought": "", "tool": "_error", "args": {},
                        "result": f"推理失败: {exc}。请输出合法 JSON。"})
        # 连续 3 次推理失败 → fallback
        errors = sum(1 for o in new_obs if o.get("tool") == "_error")
        if errors >= 3:
            return {"is_fallback": True, "error_message": "ReAct 推理连续失败",
                    "steps": _append_step(state, "agent_reasoning -> 3 failures, fallback")}
        return {"observations": new_obs, "react_iterations": iterations + 1,
                "pending_tool": {"tool_name": "_retry", "args": {}},
                "steps": _append_step(state, f"agent_reasoning iter{iterations} -> retry after error")}
```
（`tool_name == "_retry"` 时 tool_execution 直接跳过、循环回 reasoning，见 Step 4）

- [ ] **Step 4: tool_execution 写 observations + 循环回 reasoning**

`tool_execution` 节点改：执行后不仅写 `tool_result`，还把本轮 thought/tool/args/result 追加到 `observations`；若 `pending_tool.tool_name == "_retry"` 则跳过执行。workflow 加循环边 `tool_execution → agent_reasoning`（仅 use_react 时）。router 的 `route_after_tool_execution` 加分支：`use_react` 为真 → 回 `agent_reasoning`，否则 → answer_generation。

- [ ] **Step 5: intent_recognition 分流 + workflow 注册**

`intent_recognition` 返回里加 `use_react: intent == "complex_task" or confidence < 0.6`。
`route_after_intent` 加分支：`use_react` → `agent_reasoning`。
workflow.py 注册 `agent_reasoning` 节点、加 `route_after_reasoning` 条件边（pending_tool 是 final_answer 已写 draft_answer → answer_validation；否则 → tool_execution）、加 `tool_execution → agent_reasoning` 循环边。

- [ ] **Step 6: 验证 ReAct 循环 + 快速路径回归**

写 `scripts/verify_react_loop.py`：
```python
# 1. 静态: agent_reasoning/route_after_reasoning 存在, 循环边注册, state 新字段
# 2. 单工具: "1+1等于几" 走 ReAct → calculator → final_answer
# 3. 多步: "查民法典第10条再算条号数字和" → retrieval + calculator 多轮
# 4. 步数耗尽: mock AGENT_MAX_ITERATIONS=1 → 强制回答
# 5. 回归: chitchat/knowledge_qa/天气 仍走快速路径（use_react=False）
```
起后端 HTTP 实测，全绿。

- [ ] **Step 7: Commit**

```bash
git add app/graph/ app/prompts/templates.py scripts/verify_react_loop.py
git commit -m "feat: ReAct 循环子图——agent_reasoning ↔ tool_execution 循环, 与快速路径并存"
```

---

### Task 3: 结构化记忆

**Files:**
- Create: `app/memory/__init__.py` `app/memory/manager.py`（summary + facts 管理）
- Modify: `backend/storage/postgres/models_conversation.py`（Conversation 加 summary 字段）
- Create: `backend/storage/postgres/models_memory.py`（UserFact 模型）
- Modify: `backend/services/chat_service.py`（maybe_update_summary、facts 存取）
- Modify: `app/services/agent_service.py`（prepare_context 注入 summary + facts）
- Test: `scripts/verify_memory_layers.py`

**Interfaces:**
- Consumes: 现有 conversations/messages 表、get_llm_client
- Produces: `maybe_update_summary(session, conv_id)`、`add_user_fact(session, user_id, fact, conv_id)`、`get_user_facts(session, user_id) -> List[str]`；`conversations.summary` 字段；`user_facts` 表。

- [ ] **Step 1: DB 模型 + 迁移**

`models_conversation.py` 的 Conversation 加 `summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)`。
新建 `models_memory.py`：
```python
class UserFact(Base):
    __tablename__ = "user_facts"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```
`init_db()` 注册新模型；手动迁移：`ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary TEXT; CREATE TABLE IF NOT EXISTS user_facts (...);`

- [ ] **Step 2: memory manager（情景摘要 + 语义 facts）**

`app/memory/manager.py`：
```python
async def maybe_update_summary(session, conv_id, llm_client) -> None:
    """每 10 轮消息触发一次增量摘要: 旧 summary + 新增消息 → 新 summary。"""

async def add_user_fact(session, user_id, fact, conv_id=None) -> None:
    """规则触发的事实存储（本期: 用户说"记住/我喜欢/我是"时调用）。"""

async def get_user_facts(session, user_id, limit=20) -> list[str]:
    """查询用户 facts, 供注入 prompt。"""
```
`chat_service.py` 在 `add_message` 后调 `maybe_update_summary`（消息数 % 10 == 0 时触发）；`prepare_context` 在注入历史前查 summary 和 facts。

- [ ] **Step 3: 注入 prompt + 规则触发事实提取**

`agent_service.prepare_context`：messages 拼装时，若 conv 有 summary 则首条插 `{"role": "system", "content": f"对话摘要: {summary}"}` + 只取最近 10 轮历史（替代全部）；facts 非空则插 `{"role": "system", "content": f"用户画像: {'; '.join(facts)}"}`。
`chat_service.add_message`：检测 query 含"记住/我喜欢/我是/叫我"等关键词时调 LLM 提取事实存 user_facts。

- [ ] **Step 4: 验证三层记忆**

写 `scripts/verify_memory_layers.py`：
```python
# 1. user_facts 表存在, add/get_user_facts 正常
# 2. 10 轮消息触发 summary 生成（mock LLM 或直接调 maybe_update_summary）
# 3. prepare_context 注入 summary + facts 到 messages
# 4. 无 summary 时行为不变（回归）
```

- [ ] **Step 5: Commit**

```bash
git add app/memory/ backend/storage/postgres/models_memory.py backend/storage/postgres/models_conversation.py backend/services/chat_service.py app/services/agent_service.py scripts/verify_memory_layers.py
git commit -m "feat: 结构化记忆——会话摘要压缩 + user_facts 语义记忆骨架"
```

---

### Task 4: 模型分级接口

**Files:**
- Modify: `app/core/config.py`（LLM_FAST_* 配置）
- Modify: `app/llm/client.py`（get_llm_client 加 tier 参数）
- Test: `scripts/verify_model_tiers.py`

**Interfaces:**
- Produces: `get_llm_client(tier: str = "main")`；配置 `LLM_FAST_MODEL/LLM_FAST_BASE_URL/LLM_FAST_API_KEY`（Optional）。

- [ ] **Step 1: 配置项 + tier 参数**

`config.py` 加 `LLM_FAST_MODEL: Optional[str] = None` 等 3 项。`get_llm_client(tier="main")`：tier="fast" 且 LLM_FAST_MODEL 已配置时用 fast 配置建 client，否则 fallback 主 client（单例缓存按 tier 分键）。

- [ ] **Step 2: 验证**

`scripts/verify_model_tiers.py`：未配置 LLM_FAST_* 时 `get_llm_client(tier="fast")` 返回主 client；配置后返回 fast client（不同 model 名）。

- [ ] **Step 3: Commit**

```bash
git add app/core/config.py app/llm/client.py scripts/verify_model_tiers.py
git commit -m "feat: 模型分级接口——LLM_FAST_* 配置 + get_llm_client(tier) 参数"
```

---

### Task 5: 整体验证 + PROGRESS.md + 提交推送

**Files:**
- Modify: `PROGRESS.md` `ARCHITECTURE.md`

- [ ] **Step 1: 全量回归**

依次跑 4 个 verify 脚本（tool_plugin / react_loop / memory_layers / model_tiers）+ 已有 `verify_message_persistence.py` + `verify_chat_stream.py` 确认无回归；HTTP 实测 4 类快速路径意图 + 1 个 ReAct 多步任务。

- [ ] **Step 2: PROGRESS.md 记录本次阶段 1 全部变更**

- [ ] **Step 3: ARCHITECTURE.md 更新（ReAct 子图 + memory/ + 工具插件化）**

- [ ] **Step 4: Commit + push**

```bash
git add PROGRESS.md ARCHITECTURE.md
git commit -m "docs: 阶段 1 Agent 内核重构——PROGRESS/ARCHITECTURE 同步"
git push origin lx
```
