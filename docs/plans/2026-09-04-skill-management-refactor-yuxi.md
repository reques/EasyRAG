# Skill 管理机制重构规划 — 参照 Yuxi

> 日期：2026-09-04 ｜ 状态：**已实施**（阶段 1-5 完成，见 §7 实施记录）｜ 前置阅读：`2026-08-15-skill-configuration-design.md`、`2026-09-02-agentstate-refactor-yuxi.md`
>
> 参照文档：https://xerrors.github.io/Yuxi/agents/skills-management.html

## 1. 现状与差距

现在的 Skill 是**用户显式选择型**：`SkillProfile` dataclass（`app/skills/catalog.py`）+ PG 行，前端勾选 ≤3 个 → `skill_ids` 进请求 → 路由层 `_resolve_request_skills` 解析 → `use_skill_context` 写 ContextVar → 三处手工拼 system message + registry 三处白名单校验。模型看不到未勾选的 Skill，也无法自主启用。

| Yuxi 机制 | EasyRAG 现状 | 差距 |
|---|---|---|
| `SKILL.md` + YAML frontmatter + `tools/` `prompts/` 目录 | `catalog.py` 元组常量 / PG `instructions` 文本字段 | Skill 不可移植、不能带脚本与参考资料、改内置要改代码 |
| 三来源（内置 / 共享 / 个人），文件为真相 + DB 索引 | 内置=代码常量，自定义=PG 全文 | 无统一来源模型 |
| 渐进式披露：首轮只给 name+description，读 `SKILL.md` 后进 `activated_skills`，**下一轮**才解锁其工具 | 勾选即全文注入 + 工具立刻可用 | prompt 常驻膨胀；工具权限与"模型是否真的理解该 Skill"脱钩 |
| `SkillsMiddleware` 挂 `create_agent` | `get_active_skill_prompt()` 在 3 处手工拼 | 功能耦合，无法按 Agent 组装 |
| `preload_skills` 首轮展开依赖闭包 | 无 | — |
| `skill_dependencies` 依赖闭包 | 无 | — |

**关键前置条件已满足**：环境 langchain 1.3.18，`AgentMiddleware` / `dynamic_prompt` / `wrap_tool_call` / `ModelRequest.override(tools=..., system_message=...)` / `ToolCallRequest` 全部可用（已实测），三个 `create_agent` 调用点（`deep/agent.py:98`、`deep/subagents.py:242`、`dynamic.py:94`）已是 1.x 写法。这是上次阶段 1 升级留下的红利。

### 1.1 本期范围（已确认决策）

| 维度 | 决策 |
|---|---|
| 范围 | **核心三块**：SKILL.md 文件格式 + 渐进式披露 + SkillsMiddleware。不做远程安装（GitHub/ModelScope）、不做 ZIP 上传、不做 read_scope/manage_scope 共享范围 |
| 存储 | **文件为真相，DB 降级为索引**：`custom_skill_configs` 移除 `instructions` / `tool_names_json`，保留 slug/owner/enabled/source 等索引字段；一次性迁移脚本导出存量行为 SKILL.md |
| 个人目录 | `volumes/user-skills/<user_id>/<slug>/`（volumes/ 已在 .gitignore 内） |
| 前端 | 勾选定**有效集合**（上限从 3 放宽到 10），模型自主激活 |
| 激活方式 | 专用 `read_skill(slug)` 工具，副作用写 `activated_skills` |
| 工具门控 | **双层强制**：middleware `wrap_tool_call` + `ToolRegistry.invoke` ContextVar 检查 |
| 子 Agent | 继承主 Agent 的激活集，自身不可再激活新 Skill |
| 旧代码 | **彻底替换**：`catalog.py` 与 `get_active_skill_prompt()` 删除，三处手工拼接改由 middleware 接管 |

非目标（预留扩展点）：远程安装与白名单策略、共享范围与多角色授权、`install_skill` 工具、Skill 目录内 `tools/` 脚本的沙箱执行（等 `2026-09-02-tool-sandbox-design.md` 的 SandboxRunner 落地）。

## 2. 目标架构

```
skills/                                  # 内置 Skill（随代码发布，只读）
  knowledge-research/SKILL.md
  web-research/SKILL.md
  data-analysis/SKILL.md
  professional-writing/SKILL.md
  legal-analysis/SKILL.md
volumes/user-skills/<user_id>/<slug>/    # 个人 Skill（文件为真相）
  SKILL.md
  prompts/            # 可选：参考资料，由 SKILL.md 正文引用
  tools/              # 可选：脚本，本期只允许存放不允许执行

app/skills/
  loader.py      # SKILL.md 解析 + frontmatter 校验 → SkillDefinition
  registry.py    # 两来源索引（builtin / personal）+ 磁盘扫描 + mtime 缓存
  runtime.py     # SkillRuntimeContext：有效集合 / activated_skills / 依赖闭包 / 工具门控
  middleware.py  # SkillsMiddleware：dynamic system prompt + tools 过滤 + wrap_tool_call
  read_tool.py   # read_skill 工具（激活入口）
  __init__.py    # 对外导出
```

### 2.1 SKILL.md 格式

```markdown
---
name: 联网研究
slug: web-research
description: 使用联网搜索获取时效信息，并对来源进行交叉核验。
category: 研究
icon: globe
tool_dependencies:
  - web_search
skill_dependencies: []
---

## 何时使用

当任务涉及最新动态、公开资料或外部事实时使用。

## 工作方式

优先选择权威一手来源，区分发布日期与事件发生日期。

## 不要做什么

不要把搜索摘要当作已核实结论；不确定时明确说明。
```

对齐 Yuxi 的字段语义，有三处按项目现状调整：

- **必填 `name` + `description`**，与 Yuxi 一致。`slug` 省略时用 `name` 当标识，此时 `name` 必须匹配 `^[a-z0-9]+(-[a-z0-9]+)*$`；内置 Skill 全部显式写 slug（`name` 是中文展示名）。上限 128 字符。
- **`mcp_dependencies` 本期不引入**。项目 MCP 工具在注册表里就是普通工具名（`mcp_<server>_<tool>`，`app/tools/mcp/manager.py:34`），写进 `tool_dependencies` 即可，不需要单独的依赖类别。等 MCP 有了服务器级启停授权再拆。
- **`tool_dependencies` 校验策略分来源**：个人 Skill 保存时校验工具名已注册（沿用 `validate_tool_names` 的语义，≤8 个）；内置 Skill 启动时校验，未注册的工具名只 warn 跳过，不阻断启动（MCP 工具是运行时动态注册的，启动时必然查不到）。

`SkillDefinition`（`loader.py`）替代旧 `SkillProfile`：

```python
@dataclass(frozen=True)
class SkillDefinition:
    slug: str
    name: str
    description: str
    body: str                              # frontmatter 之后的正文（激活后才注入）
    tool_dependencies: tuple[str, ...] = ()
    skill_dependencies: tuple[str, ...] = ()
    category: str = "通用"
    icon: str = "sparkles"
    source: str = "builtin"                # builtin | personal
    owner_id: str | None = None            # personal 才有
    path: str = ""                         # SKILL.md 绝对路径（read_skill 用）

    @property
    def can_edit(self) -> bool: return self.source == "personal"
    def summary_line(self) -> str: ...     # 首轮 prompt 用的一行摘要
    def to_public_dict(self) -> dict: ...  # 前端契约（不含 body）
```

`body` 与 `to_public_dict()` 的分离是渐进式披露的落点：列表接口和首轮 prompt 都拿不到 `body`。

### 2.2 三层集合模型（runtime.py）

```
available_skills    用户可访问的全部 Skill（内置 + 自己的个人 Skill）
      │  用户勾选（前端，≤10）
      ▼
effective_skills    本次请求的有效集合 = 勾选 ∪ 其 skill_dependencies 闭包
      │  模型 read_skill(slug)
      ▼
activated_skills    已激活：正文进 prompt，tool_dependencies 解锁（下一轮生效）
```

依赖闭包只进入**描述范围**（`effective_skills`），不等于工具立刻暴露——与 Yuxi 一致。闭包展开做环检测与深度上限（默认 5），超限告警截断。

`preload_skills`：`effective_skills` 的子集，Run 开始时直接进 `activated_skills`（首轮就展开正文与工具）。本期通过 `ChatContext.preload_skill_slugs` 传入，默认空；根文件缺失或不可读 → 构建期抛错，不静默退回渐进加载（对齐 Yuxi）。

工具门控规则（`SkillRuntimeContext.allows_tool`）：

```
无勾选任何 Skill              → 全部工具放行（保持现有自动工具行为，与今天一致）
勾选了 Skill                  → 白名单 = read_skill ∪ 所有 activated_skills 的 tool_dependencies 并集
                                ∪ 无 Skill 声明的"公共工具"
```

**"公共工具"是本次新增的概念，必须明确**：今天 `allows_tool` 的实现是"勾选后只放行勾选 Skill 声明的工具"，意味着勾选「专业写作」（只声明 `text_tool`）时 `kb_search` 被挡住。渐进式披露把这个问题放大了——首轮 `activated_skills` 为空，若严格执行则连 `kb_search` 都不可用，模型无法回答任何知识库问题。

解决办法：给 `ToolDefinition.metadata` 增加 `"public": True` 标记，`kb_search` / `datetime_tool` / `calculator` 三个无副作用的基础工具标为公共，不受 Skill 门控。`web_search`（出网）、`text_tool`（大文本处理）、MCP 工具保持受控。这是**行为变更**，需要在验收时确认：勾选「专业写作」后 `kb_search` 从"被挡"变为"可用"。

### 2.3 SkillsMiddleware

```python
class SkillsMiddleware(AgentMiddleware):
    state_schema = SkillAwareState          # AgentState + activated_skills: Annotated[list[str], merge_slugs]
    tools = [read_skill_tool]               # middleware 自带工具注册

    def wrap_model_call(self, request, handler):
        rt = get_active_skill_context()
        activated = request.state.get("activated_skills", ())
        # 1. system_message：追加 Skill 区块（未激活=摘要行，已激活=正文全文）
        # 2. tools：过滤掉未激活 Skill 的 tool_dependencies
        return handler(request.override(system_message=..., tools=...))

    def wrap_tool_call(self, request, handler):
        # 未激活 Skill 的工具 → 直接返回 ToolMessage 错误，不进 handler
        ...
```

用 `wrap_model_call` 而非 `dynamic_prompt`，因为要同时改 `system_message` 和 `tools`——`dynamic_prompt` 只能改前者。`ModelRequest.override()` 的不可变模式（实测支持 `system_message` / `tools` 键）保证不污染原请求。

`activated_skills` 放 L1 State（逐轮变化、需 reducer 合并、随 checkpoint 持久化），符合 `2026-09-02-agentstate-refactor-yuxi.md` §2 的判据。reducer 语义：去重并集，只增不减。

**渐进式披露的"下一轮才生效"由数据流自然保证**：`read_skill` 通过 `Command(update={"activated_skills": [slug]})` 写 State，本轮模型调用的 `tools` 已经算完，下一轮 `wrap_model_call` 才读到新的 `activated_skills`。不需要额外的延迟机制。

### 2.4 双层门控与子 Agent

```
第一层  SkillsMiddleware.wrap_tool_call   → 主 Agent / dynamic Agent 的工具调用
第二层  ToolRegistry.invoke ContextVar    → 子 Agent 线程、graph 节点、MCP 桥接
```

第二层保留今天的实现位置（`registry.py:175/192/236`），但检查对象从 `SkillRuntimeContext.allowed_tool_names`（静态白名单）换成"已激活集合的并集"。ContextVar 里存的 `SkillRuntimeContext` 变为可变激活状态的持有者——`read_skill` 同时写 State 和 ContextVar，前者给 middleware，后者给子 Agent 线程。

子 Agent 继承主 Agent 的激活集：`snapshot_request_context()` 抓的 contextvars 快照里已经含最新激活状态（`app/agents/events.py:156`），子 Agent 侧不挂 `SkillsMiddleware`、不给 `read_skill` 工具，因此只能用继承来的集合。与 Yuxi 的"子智能体不可用 install_skill"同构。

`resolve_tool_spec`（`deep/subagents.py:41`）的候选池 `registry.list_all()` 已经过 ContextVar 门控，因此 SubAgent 的 `"*"` / `"@tag"` 声明自动收窄——这条链不用改。

## 3. 数据库与迁移

`custom_skill_configs` 表变更：

```
移除  instructions       TEXT           → 内容移到 SKILL.md
移除  tool_names_json    TEXT           → 移到 frontmatter tool_dependencies
新增  slug               VARCHAR(128)   → 文件目录名，与 owner 联合唯一
新增  source_type        VARCHAR(16)    → 'personal'（预留 'shared'）
保留  id/owner_id/name/description/category/icon/is_active/created_at/updated_at
```

项目没有 Alembic（`init_db()` 用 `create_all` + 手写幂等 `_migrate_*` 函数，`backend/storage/postgres/manager.py:109`）。沿用同一模式加 `_migrate_skill_config_to_files(conn)`：

1. 检查 `slug` 列是否存在，存在则直接返回（幂等）；
2. `ADD COLUMN slug` / `source_type`，回填 `slug = 'skill-' || substr(id::text, 1, 8)`（存量 name 多为中文，不能直接 slugify），建联合唯一索引；
3. 逐行导出 `instructions` + `tool_names_json` 到 `volumes/user-skills/<owner_id>/<slug>/SKILL.md`；
4. **导出全部成功**后才 `DROP COLUMN instructions, tool_names_json`；任一行失败则保留旧列并 `ERROR` 日志，下次启动重试。

导出与删列的顺序是关键：先删后写会在中途失败时丢数据。

## 4. API 与前端变更

| 端点 | 变更 |
|---|---|
| `GET /chat/skills` | 响应加 `body_available: true`（不含 body 全文）；`max_selected` 3 → 10；`source` 值 `builtin`\|`custom` → `builtin`\|`personal` |
| `POST/PUT /chat/skills` | 请求体 `instructions` → 写 SKILL.md 正文，`tool_names` → frontmatter；后端负责渲染文件。请求契约保持不变，前端表单不用改 |
| `DELETE /chat/skills/{slug}` | 删 DB 行 + 删目录（先 DB 后文件；文件删除失败只 warn，孤儿目录由启动扫描清理） |
| `GET /chat/skills/{slug}/content` | **新增**：返回 SKILL.md 全文（配置弹窗编辑用；`read_skill` 工具走内部调用不经此端点） |
| `ChatRequest.skill_ids` | `max_length` 3 → 10；语义从"注入这些"变为"本次可用范围" |

前端改动集中在三处：`maxSelectedSkills` 上限来自接口（已经是 `data.max_selected || 3`，自动生效）；下拉面板文案"最多选择 N 个"改为"选择本次对话可用的 Skill"；已激活 Skill 在任务状态栏显示一个标记（SSE 事件 `skill_activated`，让用户看到模型激活了哪个）。

## 5. 分阶段实施

| 阶段 | 内容 | 验收 | 状态 |
|---|---|---|---|
| 1 | `loader.py` + `SkillDefinition` + 5 个内置 SKILL.md 文件 + `registry.py` 两来源扫描 | 单测：frontmatter 校验（缺 name/description、非法 slug、依赖环）；5 个内置 Skill 解析结果与旧 `BUILTIN_SKILLS` 语义等价 | ✅ |
| 2 | `runtime.py` 三层集合 + 依赖闭包 + `allows_tool` 新语义 + `public` 工具标记 | 单测：闭包展开与环检测、激活前后工具集变化、无勾选时全放行 | ✅ 判据顺序有修正，见 §7.1① |
| 3 | `middleware.py` + `read_skill` 工具 + 三个 `create_agent` 挂载，删除三处手工拼接与 `catalog.py` | 集成测：首轮 prompt 只含摘要行；`read_skill` 后下一轮出现该 Skill 工具；未激活工具调用被 `wrap_tool_call` 拒绝 | ✅ `test_skill_progressive_disclosure.py` 7 个用例 |
| 4 | DB 迁移 + 路由层改造（`_resolve_request_skills` 改查文件索引）+ 新增 content 端点 | 存量库启动迁移成功，导出的 SKILL.md 可被 loader 解析；owner 隔离回归测试通过 | ⚠️ 导出逻辑已测；SQL 未跑真实 Postgres，见 §7.2 |
| 5 | 前端文案 + `skill_activated` SSE 事件 + 任务状态栏标记 | 浏览器验证：勾选 5 个 → 发送 → 状态栏出现激活标记 → 该 Skill 工具被调用 | ⚠️ 代码完成 + 构建通过；浏览器端到端未验证（依赖后端服务） |

依赖：1 → 2 → 3 → 4 → 5。阶段 3 完成时新机制已可端到端跑通（内置 Skill），阶段 4 才动存量数据——这个顺序让"迁移出错"和"机制有 bug"两类问题不会同时出现。

## 6. 风险与已知取舍

1. **公共工具标记是行为变更**（§2.2）。勾选「专业写作」后 `kb_search` 从被挡变为可用。这是渐进式披露的必要代价：首轮 `activated_skills` 为空，若不放行基础工具则模型无法回答任何问题。需要在阶段 2 验收时确认这个变更可接受。
2. **首轮 token 变化方向不确定**。摘要行比全文短（省），但勾选上限从 3 放宽到 10（增）。10 个 Skill 的摘要行约 400-600 token，对比今天 3 个 Skill 全文约 600-900 token——大致持平或略省。真正的收益是模型只在需要时才读全文，而不是无条件承担全部指令。
3. **多轮对话的激活状态跨轮次持久性**。`activated_skills` 进 checkpoint，同一 thread 的下一条消息会继承上一轮的激活集。这符合直觉（用户追问时不该让模型重新激活），但意味着"某轮误激活的 Skill 会一直留着"。本期接受；若需要按消息重置，把 reducer 换成 `EphemeralValue` 语义即可。
4. **文件系统成为新的故障源**。磁盘不可读、目录被误删、`volumes/` 挂载丢失都会让 Skill 消失。`registry.py` 的扫描对单个坏文件只 warn 跳过（沿用 `_load_subagents_file` 的健壮性策略，`deep/subagents.py:109`），但整个目录消失时只能报空列表。
5. **`tools/` 目录本期只存不跑**。SKILL.md 正文可以引用脚本路径，但没有执行通道。等 `SandboxRunner`（`2026-09-02-tool-sandbox-design.md`）落地再接，届时脚本执行的写入目标要限制在工作目录，不能写 Skill 目录本身（Yuxi 明确提醒过这点）。

## 7. 实施记录（2026-09-04）

阶段 1-5 全部落地，测试 410 passed（重构前 404），5 个失败与本次无关（`test_custom_model_config` 的 `supports_vision` stub、`test_graph_build_acceleration` 的 `GRAPH_EXTRACT_CONCURRENCY`、`test_graph_extraction` 的 jieba 断言——重构前后同样失败，已用 `git stash` 前后对照确认）。

### 7.1 实现与规划的三处偏差

**① 公共工具标记优先于 Skill 声明**（规划 §2.2 未考虑到的冲突）。

规划把 `allows_tool` 的判据写成"先查已激活、再查被门控、最后查 public"。实现时被 `test_dependency_closure_is_visible_but_locked` 打出来一个矛盾：`kb_search` 既标了 `public: True`，又被 `knowledge-research` 声明在 `tool_dependencies` 里，于是它先命中"被门控"分支，勾选 legal-analysis 的请求首轮连知识库都查不了——正是 `public` 要解决的问题本身。

改为 **public 优先于门控**：`public` 是工具自身的绝对属性（"平台基础能力，不受 Skill 选择限制"），Skill 在 `tool_dependencies` 里写公共工具只是**说明它会用到**，不构成加锁。连带修正两处提示文本：`gated_tool_names` 与 `summary_line` 都剔除公共工具，否则 prompt 会说"激活后可用：kb_search"而它一直可用。测试固化在 `test_public_flag_wins_over_skill_declaration`。

**② 非 Agent 路径需要 eager 渲染**（规划完全没覆盖）。

`prepare_context`（固定编排 + 单次生成）、`graph/nodes.py` 的 `intent_recognition`、`chat_router` 的直连 LLM 兜底这三条路径不走 `create_agent`，middleware 无从挂载，也没有 `read_skill` 工具和多轮循环——渐进式披露在那里无从发生，只给摘要行等于什么约束都没给。

加了 `render_prompt(eager=True)`：把有效集合全部当作已激活渲染。这不是妥协，而是把"哪条路径能渐进、哪条不能"显式化——`eager` 出现在代码里就是一个可查的标记。

**③ `mcp_dependencies` 未引入**。项目的 MCP 工具在注册表里就是普通工具名（`mcp_<server>_<tool>`），写进 `tool_dependencies` 即可，单独一个依赖类别没有信息量。等 MCP 有了服务器级启停授权再拆。

### 7.2 迁移安全性

`render_skill_markdown` 的往返保证是硬要求，`test_export_handles_empty_optional_fields` 抓到一个真实的数据丢失路径：旧表 `description` 列默认 `''`，这类行会导出成 loader 拒绝解析的文件，而迁移随后就 DROP 掉旧列——Skill 永久丢失。修法是在渲染层兜底（`description` 空→回落 `name`，`body` 空→回落 `description`），而不是让每个调用方各自记得。

迁移的顺序保证（先导出全部行、全部成功才 DROP）由 `test_migration_drops_legacy_columns_only_after_full_export` 用源码断言固化——它检查失败守卫的 `return` 出现在 `DROP COLUMN` 之前。

**未验证项**：迁移的 SQL 部分没有跑过真实 Postgres（本机 Docker 未启动）。首次带存量数据启动前建议先备份 `custom_skill_configs`。导出逻辑（数据丢失风险所在）已有 5 个单测覆盖。

### 7.3 公开标识变更

Skill 的对外 id 从 `builtin:<slug>` / `custom:<uuid>` 统一为 **slug**，让"前端选中的 id"、"磁盘目录名"、"`read_skill` 的参数"三者一致（旧形态需要在三处之间来回映射）。

前端 localStorage 里的旧 id 会在 `loadSkills` 的 `validIds.has(id)` 过滤时被静默丢弃——用户表现为"上次勾选的 Skill 没了"，重新勾一次即可，不报错。历史消息的 `metadata_json` 里存的是旧 id + 名称快照，只用于展示，不受影响。

