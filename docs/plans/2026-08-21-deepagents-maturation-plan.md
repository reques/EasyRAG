# DeepAgents 引入成熟化规划（2026-08-21）

> 状态：**部分实现**（S1、S2、S3、S8 的 reasoning_content 部分已完成，
> 2026-08-21，见 PROGRESS.md 对应条目；其余阶段待评审后推进）。本文档记录
> **计划中**的路线图，每阶段完成后在 `PROGRESS.md` 追加变更记录（约定：大的
> 改动必须同步 PROGRESS.md）。

## 1. 背景与目标

DeepAgents 模式（`AGENT_MODE=deepagents`）是当前三条执行路径之一（单 Agent
graph / Orchestrator-Worker 多智能体 / DeepAgents 主 Agent+SubAgent），当前
引入程度"可用但不够成熟"：核心骨架（主 Agent + task 委派 + 配置化 SubAgent +
SSE 透传 + mock 测试）已具备，但存在**知识库检索缺失**、**配置不可达**、
**子 Agent 黑盒**等影响可用性与正确性的缺口。

**目标**：让 DeepAgents 成为一条与单 Agent / 多智能体同等成熟度的路径——
知识库问答可检索、委派过程可观测、异常可降级、配置可发现、行为可验证。

## 2. 现状架构（代码确认，2026-08-21）

```
AGENT_MODE=deepagents
  └─ agent_service.run() ──> _run_deep()          # 同步；SSE 走 chat_router 专用分支
       ├─ 消息组装：skill_prompt + knowledge_catalog(仅目录) + 用户事实 + history + query
       ├─ 主 Agent：create_react_agent            # langgraph prebuilt
       │    ├─ 全量工具（registry → StructuredTool，技能白名单运行时生效）
       │    └─ task 委派工具 → run_subagent()     # 同步，独立 state，上下文隔离
       └─ 返回 final_answer + steps + artifacts
```

关键文件：
- `app/agents/deep/agent.py` — 主 Agent 构建（进程级缓存）
- `app/agents/deep/task_tool.py` — `task(description, subagent_type)` 委派工具
- `app/agents/deep/subagents.py` — SubAgent 配置/构建/运行（`DEFAULT_SUBAGENTS` 2 个）
- `app/agents/deep/tools.py` — ToolRegistry → StructuredTool 转换
- `app/agents/deep/llm.py` — ChatOpenAI 适配（与 LLMClient 共用配置）
- `app/services/agent_service.py::_run_deep` — 编排
- `backend/server/routers/chat_router.py` — `/chat/stream` 的 `use_deep` 分支
- `tests/test_deep_agents.py` — 11 个 mock 测试

已具备：工具转换层（可用性检查运行时生效）、SubAgent 配置化（内置+外部文件）、
历史与用户事实注入、主 Agent 层 SSE 状态/artifact 透传、进程级缓存、mock 测试。

## 3. 现状缺口（按严重度排序，含代码证据）

| # | 缺口 | 证据 |
|---|---|---|
| ~~S1~~ | ~~知识库检索缺失~~ **已修复（2026-08-21）** | 见 PROGRESS.md 对应条目；`kb_search` 工具 + `_run_deep` 前置检索注入 |
| ~~S2~~ | ~~配置不可达~~ **已修复（2026-08-21）** | AGENT_MODE Literal 加 deepagents；DEEP_SUBAGENTS_FILE 等声明（外部覆盖失效修复）；`.env.template` 文档化；启动日志 |
| **S2** | 配置不可达 | `config.py:169` `AGENT_MODE: Literal["auto","single","multi"]` **缺 "deepagents"**；`DEEP_SUBAGENTS_FILE` 未声明（`subagents.py:92` 用 `getattr(cfg, "DEEP_SUBAGENTS_FILE", None)` 兜底 → pydantic-settings 未声明字段读不到 env → 外部覆盖**实际失效**）；`.env.template` 无 AGENT_MODE/DEEP_* 文档 |
| ~~S3~~ | ~~子 Agent 黑盒~~ **已修复（2026-08-21）** | 请求级观察者透传（`observe.py`）；`run_subagent` 改 stream 循环；委派过程前端 SSE 可见 |
| **S4** | 超限无降级 | 主 Agent `GraphRecursionError`（recursion_limit 耗尽）→ `_run_deep` 捕获后直接返回错误响应；对比单 Agent graph 有 "max iterations, forced answer" 收尾 |
| **S5** | task 无熔断 | 子 Agent 失败仅返回错误文本（`task_tool.py:70`），主 Agent 可能对同一任务反复委派，无失败计数/上限 |
| **S6** | 工具集静态化 | `build_main_agent`/`build_subagent` 构建时固定工具列表（进程级缓存）；请求级技能激活新增的工具不会反映（可用性检查运行时生效，但列表静态） |
| **S7** | 前端仅标签级 | 只有 intent 标签"智能体"（`ChatView.vue:572`），无委派可视化；对比 orchestrator 有 sub_tasks 面板 |
| **S8** | 验证薄弱/适配粗糙 | **reasoning_content 回传已修复（2026-08-21，见 PROGRESS.md）**；真实 LLM 冒烟已有（S1/S8 冒烟均通过）。剩余：无 SSE 分支集成测试；`llm.py` `max_tokens=cfg.LLM_MAX_TOKENS` 全局固定（reasoning 模型 max_tokens 下限问题未处理，见 `generate_conversation_title` 注释） |

## 4. 分阶段规划

### 阶段 0（P0，可达性）— 配置与入口补全 ✅（2026-08-21 完成）

目标：DeepAgents 模式"可发现、可配置、可切换"，外部 SubAgent 覆盖真正生效。

任务（全部完成，见 PROGRESS.md "DeepAgents 成熟化 S2" 条目）：
1. ✅ `config.py`：`AGENT_MODE` Literal 增加 `"deepagents"`；声明
   `DEEP_SUBAGENTS_FILE: str = ""`、`DEEP_MAIN_RECURSION_LIMIT: int = 20`、
   `DEEP_SUBAGENT_RECURSION_LIMIT: int = 20`
2. ✅ 修复 `DEEP_SUBAGENTS_FILE` 失效（声明字段后生效；文件不存在/坏格式回退内置并告警）
3. ✅ `.env.template`：文档化 `AGENT_MODE` 四种取值与 `DEEP_*` 配置
4. ✅ 启动日志：服务启动时打印当前 AGENT_MODE 与子 Agent 名册（可发现性）
5. ✅ 冒烟：`AGENT_MODE=deepagents` 配置可实例化；真实 env 读取验证

### 阶段 1（P1，核心能力）— 知识库接入 + 子 Agent 可观测性 + 健壮性

目标：消除 S1（幻觉风险）、S3/S4/S5（黑盒/无降级/无熔断）。

任务（✅ = 已完成 2026-08-21）：
1. ✅ **主 Agent 检索接入（S1）**：`_run_deep` 生成前执行知识库检索（增强检索，
   注入 system 上下文 + sources 收集；失败不阻塞）
2. ✅ **kb 检索工具（S1）**：`kb_search` 注册表工具（请求级授权 ContextVar），
   主 Agent 全量可见 + research-agent 白名单已加（子 Agent 也能查知识库）
3. ✅ **子 Agent 步骤透传（S3）**：请求级观察者（`observe.py`）→ task 工具
   转发 → `run_subagent` stream 循环透传（`{subagent_name}/step` 前缀），
   复用现有 SSE artifact 通道，前端可见委派过程
4. **超限优雅收尾（S4）**：捕获 `GraphRecursionError`，基于已有 messages 生成
   收尾答案（对齐单 Agent graph 的 forced-answer）
5. **task 熔断（S5）**：按 (会话, subagent_type) 连续失败计数，超限后返回
   "建议停止委派"提示主 Agent，避免死循环

验收（任务 1-3 已达成：DeepAgents 下知识库问答引用检索内容并有来源；
SSE 委派过程可见；任务 4/5 待完成）。

### 阶段 2（P2，体验与一致性）

目标：与单 Agent / 多智能体体验对齐，消除 S6/S7。

任务：
1. 前端 DeepAgents 面板（S7）：委派树/子 Agent 状态，复用 orchestrator 的
   sub_tasks 组件模式
2. 委派记录持久化：复用/扩展 `multi_agent_runs` 表，历史可回放
3. 工具集按请求动态构建（S6）：技能激活时失效缓存/重建 agent（或按请求构建，
   权衡缓存收益）
4. 与 orchestrator 统一路线评估：deepagents 与 Orchestrator-Worker 功能重叠
   （子 Agent vs Worker），输出评估结论（合并/共存/淘汰其一），作为独立决策项
5. （可选）query_rewrite 对齐：deepagents 是否也做指代消解（当前靠 history
   注入让模型自解，可接受）

### 阶段 3（P3，验证与发布）

目标：行为可验证、文档齐全、覆盖 S8。

任务：
1. 真实 LLM 端到端冒烟脚本（标记 slow，不入常规 CI）+ SSE 分支集成测试
2. `llm.py` max_tokens 策略对齐（reasoning 模型下限，参考
   `generate_conversation_title` 的 100+ 注释）
3. 文档：README/DeepAgents 章节（开启方式、SubAgent 配置、与另两条路径的关系）
4. `PROGRESS.md` 按阶段记录

## 5. 决策点（需要评审时明确）

- D1：**kb 检索工具形态**——`kb_search` 工具（模型自主决定何时检索）还是
  `_run_deep` 前置强制检索（每次必检）+ 可选工具？前者灵活但可能漏检，
  后者稳定但多一次调用。倾向：前置强制检索（消除幻觉是 P1 目标）+ 工具可选。
- D2：**DeepAgents 与 Orchestrator 的关系**（阶段 2）——两者都是"多智能体"，
  未来是否以 deepagents 为统一实现（子 Agent 可继承 orchestrator 的 Worker
  能力/黑板）？
- D3：**工具集动态化代价**——技能激活时重建 agent 缓存（首次调用多构建一次）
  还是按请求构建（放弃缓存）？需要实测构建耗时。

## 6. 相关文件索引

- 主 Agent：`app/agents/deep/agent.py`
- 委派：`app/agents/deep/task_tool.py`；SubAgent：`app/agents/deep/subagents.py`
- 工具转换：`app/agents/deep/tools.py`；模型适配：`app/agents/deep/llm.py`
- 编排：`app/services/agent_service.py::_run_deep`
- SSE 分支：`backend/server/routers/chat_router.py`（`use_deep`，L749 起）
- 配置：`app/core/config.py`（AGENT_MODE L169）、`.env.template`
- 测试：`tests/test_deep_agents.py`（mock，11 项）
- 前端：`frontend/src/views/ChatView.vue`（intent 标签）、
  `frontend/src/components/AgentActivity.vue`
