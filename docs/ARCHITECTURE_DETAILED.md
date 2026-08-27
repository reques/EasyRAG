# EasyRAG 整体架构详解

> 版本：基于当前 `y` 分支代码（2026-08-25 核对）
> 本文是对 [ARCHITECTURE.md](./ARCHITECTURE.md)（宏观总览）的深度补充，逐模块讲「怎么实现、为什么这么实现」。

---

## 1. 项目定位

EasyRAG 是一个面向真实业务场景的**企业知识库智能问答平台**：多策略 RAG + Agent 工具调用 + 知识图谱 + 多智能体编排，开箱即用的全栈应用（Vue 3 + FastAPI + LangGraph + Milvus）。

与"跑通 demo 即止"的玩具项目的区别：多用户 JWT 认证、文档管理、SSE 流式对话、知识图谱可视化、检索评估（确定性指标 + 可选 Ragas）、可配置 Skill 系统、MCP 外部工具接入、旁路部署的 MinerU 文档解析服务。

**一句话概括全链路**：用户提问 → FastAPI 路由（JWT 鉴权）→ AgentService 分流（LangGraph 单 Agent / DeepAgents 多智能体）→ 意图识别分流 → 检索（增强五步流水线或传统向量检索）→ LLM 生成 → 校验 → SSE 逐 token 推给前端。

---

## 2. 技术栈全景

| 层 | 选型 |
|----|------|
| 前端 | Vue 3.5 · Vite 6 · Pinia · Axios · lucide 图标 · ECharts 5（图谱） |
| 后端 | FastAPI（async）· SQLAlchemy 2.0 async · LangGraph 工作流 |
| Agent | LangGraph StateGraph（意图分流 / ReAct 循环 / 校验重试）· DeepAgents 统一多智能体（主 Agent + SubAgent + DAG 委派 + 结构化黑板） |
| 存储 | PostgreSQL（pgvector 镜像，业务数据 + 图谱 + Skill 配置）· Redis · MinIO |
| 向量 | Milvus 2.5（etcd + MinIO 依赖）· BGE-M3 embedding（本地 / Ollama / API） |
| LLM | DeepSeek / MiniMax / Qwen(DashScope) / GLM / 任意 OpenAI 兼容 API（自定义 base_url + 加密 API Key） |
| 文档解析 | 本地解析器 + 旁路部署 MinerU Pipeline API（Docker，GPU） |
| 评估 | 本地确定性指标（HitRate / MRR / avg_score）+ 可选 Ragas（独立 venv） |
| 部署 | Docker Compose 7 服务一键编排 |

---

## 3. 总体架构分层

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Vue 3 SPA)                         │
│   ChatView(SSE流式) · KnowledgeView(文件/检索测试/图谱) · 评估页      │
│   Pinia(auth/chat) · api/index.js(axios + fetch SSE 封装)            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ REST / SSE (/api/v1, Vite 代理 :8001)
┌───────────────────────────────▼─────────────────────────────────────┐
│                    Backend (FastAPI async, :8001)                    │
│   routers: auth / chat / knowledge / evaluation / mcp (+旧路由)      │
│   services: chat · knowledge · graph · evaluation · agent_run ...    │
│   repositories: 数据访问层（泛型 BaseRepository[T]）                  │
└───────┬────────────────┬────────────────┬────────────────┬───────────┘
        │                │                │                │
┌───────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼──────────┐
│   Agent 内核   │ │   RAG 管线    │ │  工具系统     │ │  外部服务        │
│ app/agents     │ │ app/rag      │ │ app/tools    │ │                 │
│ app/graph      │ │ (chunker/    │ │ registry     │ │ Postgres(pgvector│
│ (LangGraph)    │ │  embedding/  │ │ + MCP 桥接   │ │  +图谱+业务)     │
│ DeepAgents     │ │  retriever/  │ │ app/skills   │ │ Milvus(向量)     │
│ 委派协同        │ │  bm25/rerank/│ │ app/memory   │ │ Redis · MinIO    │
│                │ │  ocr/parsers)│ │              │ │ Ollama(embedding)│
│                │ │  graph_cache)│ │              │ │ MinerU(旁路解析) │
└────────────────┘ └──────────────┘ └──────────────┘ │  Tavily(联网搜索) │
                                                     └──────────────────┘
```

关键分层原则：**`app/` 是纯逻辑层（Agent/RAG/工具，与业务解耦），`backend/` 是业务后端（HTTP 契约 + 业务逻辑 + 数据访问 + 存储客户端）**。`app/api/` 是旧版无鉴权路由（保留兼容）。

---

## 4. 目录结构逐层讲解

```
EasyRAG/
├── app/                          # Agent 内核（纯逻辑层）
│   ├── agents/                   # 多智能体层（DeepAgents 统一，2026-08-26 退役 Orchestrator）
│   │   ├── events.py             #   统一事件流（请求级 trace + span + emit + sink）
│   │   ├── progress.py           #   进度投影器（SSE 进度摘要）
│   │   └── deep/                 #   DeepAgents：主 Agent + task/spawn_tasks 委派 +
│   │                              #   结构化黑板 + 子智能体名册 + 动态重规划
│   ├── graph/                    # LangGraph 工作流
│   │   ├── state.py              #   AgentState TypedDict（total=False）
│   │   ├── nodes.py              #   10 个节点实现（966 行）
│   │   ├── router.py             #   7 个条件路由函数
│   │   ├── workflow.py           #   StateGraph 装配与编译（单例 get_graph()）
│   │   └── fast_intent.py        #   规则快速意图（简单问题零成本分流）
│   ├── llm/                      # LLM 客户端
│   │   ├── client.py             #   LLMClient: sync/async/JSON/stream 四接口 + 分级(tier)
│   │   └── models.py             #   模型目录（DeepSeek/MiniMax/Qwen/GLM + 自定义）
│   ├── rag/                      # RAG 管线（核心）
│   │   ├── chunker.py            #   文本提取 + 5 种分块策略（含 legal）
│   │   ├── embeddings.py         #   BGE-M3 embedding（local/ollama/api 三实现）
│   │   ├── vector_store.py       #   向量库三后端（Milvus/Chroma/Memory）
│   │   ├── retriever.py          #   检索器（MilvusRetriever 等，父块回填）
│   │   ├── enhanced_retriever.py #   ★ 增强检索引擎（1962 行，五步流水线）
│   │   ├── bm25.py               #   Okapi BM25 + jieba 分词
│   │   ├── reranker.py           #   交叉编码器精排（local/api/disabled）
│   │   ├── graph_cache.py        #   进程级图谱内存缓存（避免 PG 线程冲突）
│   │   ├── ocr.py                #   扫描件 OCR（PaddleOCR 式引擎封装）
│   │   ├── parsers/              #   解析器：router(分流)/local_parser/mineru_*
│   │   └── extractors/           #   [Neo4j 分支遗留，用户明确不用，勿启用]
│   │   └── graph_retriever.py / graph_vector_index.py / rrf.py   #   [同上，遗留]
│   ├── tools/                    # 工具系统
│   │   ├── registry.py           #   ToolRegistry 单例（RLock 线程安全，invoke 锁外执行）
│   │   ├── web_search_tool.py / calculator.py / datetime_tool.py / text_tool.py
│   │   ├── kb_search.py          #   知识库搜索工具
│   │   └── mcp/                  #   MCP 客户端桥接（config/manager/demo_server）
│   ├── skills/                   # Skill 系统（catalog 内置 + context 注入）
│   ├── memory/                   # 分层记忆管理
│   ├── prompts/                  # Prompt 模板（意图/规划/ReAct/生成/校验/重写）
│   ├── services/                 # agent_service(核心编排入口) / knowledge_catalog
│   ├── core/                     # config / exceptions / logger
│   └── api/                      # [遗留] 旧版路由 /health /chat /kb/*
├── backend/                      # 业务后端（分层架构）
│   ├── server/
│   │   ├── main.py               #   FastAPI 装配 + lifespan（建表/增量迁移/种子/MCP 启动）
│   │   ├── routers/              #   auth / chat / knowledge / evaluation / mcp
│   │   └── seed.py
│   ├── services/                 # chat / knowledge / graph / evaluation / model_config
│   │                             # skill_config / ragas_evaluator / ragas_worker / agent_run
│   ├── repositories/             # BaseRepository[T] 泛型 + 各实体仓库
│   └── storage/                  # postgres(models_*.py) / redis / minio 客户端
├── frontend/                     # Vue 3 SPA
│   ├── src/views/                # ChatView / KnowledgeView / EvaluationView / Login / Register / Layout
│   ├── src/components/           # AgentActivity(任务面板) / ProgressJournal
│   ├── src/stores/               # Pinia（auth / chat）
│   ├── src/api/index.js          # axios 封装 + fetch SSE 流式封装
│   └── src/router/               # 路由 + 登录守卫
├── deploy/mineru/                # MinerU 解析服务（已合并进主 compose）
├── verify/                       # 人工验证脚本
├── scripts/                      # 迁移/验证脚本
├── tests/                        # pytest（36+ 测试文件）
├── docs/                         # 本架构文档、plans/、specs/
├── docker-compose.yml            # 7 服务编排（etcd/minio-s3/milvus/postgres/redis/minio/mineru-api）
├── .env / .env.template          # 配置（.env 优先级高于代码默认值！）
└── PROGRESS.md                   # 逐次迭代的演进记录（93KB 历史）
```

> 注意：`app/rag/` 下的 `graph_retriever.py`、`graph_vector_index.py`、`rrf.py`、`extractors/`、`app/agents/deep/` 来自一次 Neo4j 分支合并。**用户已明确不要 Neo4j GraphRAG**，这些文件保留但不应启用。

---

## 5. 请求生命周期（核心章节）

### 5.1 两条对话路径

| 路径 | 端点 | 特点 |
|------|------|------|
| 同步 | `POST /api/v1/chat/send` | 完整 LangGraph 工作流，一次性返回 |
| 流式 | `POST /api/v1/chat/stream` | SSE 逐 token 推送，边生成边渲染 |

### 5.2 同步路径 /chat/send

```
send_message (chat_router.py)
  ├─ 1. _resolve_request_model：按 req.model_id 解析模型（内置目录 or 用户自定义）
  ├─ 2. _resolve_request_skills：解析 Skill 组合（最多 3 个）→ SkillRuntimeContext
  ├─ 3. 会话：复用或创建 conversations 行；保存用户消息（metadata 带 skills 快照）
  ├─ 4. get_compressed_history：情景记忆压缩（有 summary = 摘要 + 最近 N 轮）
  ├─ 5. _load_knowledge_scope：查用户全部知识库授权范围
  ├─ 6. 多智能体判定：use_multi = AGENT_MODE=="multi" 或 (auto 且 _should_use_multi)
  │     → 命中则先创建 agent_runs 记录（前端任务面板数据）
  ├─ 7. agent.run(...)  ← AgentService 三层路由（见 5.4）
  └─ 8. 答案落库（assistant 消息 + 引用 sources JSON）+ run 收尾
```

### 5.3 流式路径 /chat/stream（SSE）

事件序列：`conversation_id` → `delta`（多次）→ `done`（sources/intent/elapsed）→ 出错时 `error`。

```
send_message_stream
  ├─ 同样的模型/Skill/会话/历史/知识范围解析
  ├─ use_deep = AGENT_MODE 属 {deepagents, multi} 或 req.deep_research=true
  │     或 auto 且 _should_use_multi 命中（multi 为 deepagents 兼容别名）
  ├─ agent_mode = deepagents | single（首个事件就返回，前端徽标用）
  ├─ DeepAgents：线程里跑主 Agent，状态队列(Queue) 逐条投影为 progress_summary；
  │     统一事件流经桥接映射为任务面板协议（sub_tasks/status/worker_output）
  ├─ 单 Agent：prepare_context（手动执行意图→工具→检索节点，不跑完整图）
  │     → llm.chat_stream 逐 token yield delta
  └─ 结束：done 事件携带 sources + intent + run_id（委派持久化）；答案/引用落库
```

流式路径不完整跑 LangGraph 图（避免 stream 与图执行状态纠缠），而是手动执行「意图识别 → 工具 → 检索 → 拼装 messages」后直接调 `chat_stream`。**两条路径的 context 注入逻辑保持一致**（`prepare_context` 与图内节点共用同一套模板与截断守卫）。

### 5.4 AgentService 三层路由（agent_service.py）

```
agent.run()
  ├─ AGENT_MODE == "deepagents"        → _run_deep()    （主 Agent + SubAgent 委派）
  ├─ AGENT_MODE == "multi"             → _run_deep()（deepagents 兼容别名，仅告警提示）
  ├─ AGENT_MODE == "auto":
  │     _should_use_multi(query) 命中 → _run_deep()（主 Agent 自行拆解/委派）
  │     不命中                          → self._graph.invoke()（LangGraph 单 Agent）
  └─ AGENT_MODE == "single"            → self._graph.invoke()
```

`_should_use_multi` 轻量规则（不调 LLM）：
1. 查询命中 **≥2 个不同领域**的关键词（12 个领域字典：法律/代码/金融/医疗/教育/生活…）→ multi
2. 查询 >80 字符且含「然后/并且/同时」等连词 → multi
3. 其余 → single

命中规则只是把请求交给 DeepAgents 主 Agent，是否真正拆解/委派由模型自行决定（task / spawn_tasks 工具），无需独立拆解器。

### 5.5 会话记忆与摘要折叠

- 无摘要时：注入最近 `HISTORY_CONTEXT_MAX_MESSAGES=100` 条
- 有摘要时：`摘要 + 最近 20 条`，`conversations.last_summarized_message_id` 记录折叠断点
- 摘要长期生成失败时：注入原始消息并截断 + 记日志

---

## 6. LangGraph 工作流详解（app/graph）

### 6.1 拓扑（10 节点 + 7 条件路由）

```
query → [query_rewrite]  ← 追问指代消解（短句/指代词 + 有历史才触发 LLM 改写）
           ↓
    [intent_recognition]  ← 快速规则路径(fast_intent) 优先，否则 LLM JSON 分类
           │
           ├─ use_react(complex_task 或 confidence<0.6) → [agent_reasoning] ⇄ [tool_execution]  (ReAct 循环)
           │                                               └─ final_answer → [answer_validation]
           ├─ tool_use ──→ [tool_selection] → [tool_execution] ──→ [answer_generation]
           ├─ knowledge_qa ──→ [knowledge_retrieval] ──→ [answer_generation]
           └─ chitchat ──→ [answer_generation]
                                        │
                              [answer_generation] → [answer_validation]
                                                      ├─ passed → END
                                                      └─ failed(≤1次重试) → answer_generation
任何节点出错 → [fallback_handler] → END
```

### 6.2 节点职责

| 节点 | 实现要点 |
|------|---------|
| **query_rewrite** | 短句（≤12字）或含指代词（呢/那/它/这/还有…）且 ≤30 字 → 用 fast tier LLM 结合历史改写为自包含问题；失败/无历史原样返回。与 SSE 快速路径统一（P1 修复，两条路径行为一致） |
| **intent_recognition** | ① `FAST_INTENT_ENABLED` 且未启用 Skill → `fast_intent_detect` 规则预判（常识/问候/计算/时间直接出意图，零 LLM）；② 否则 LLM JSON 分类输出 {intent, confidence, requires_retrieval, requires_tool, tool_name, tool_args}，工具列表动态注入（含 MCP）；③ `use_react = intent=="complex_task" or confidence<0.6`；④ 失败降级 knowledge_qa |
| **task_planning** | 复杂任务 LLM 拆解 {sub_tasks[], needs_retrieval, needs_tool}，最多 `MAX_PLAN_STEPS=5`；失败退化单任务 |
| **knowledge_retrieval** | 双路径：`ENHANCED_RETRIEVAL_ENABLED=true` → 增强五步流水线；否则传统向量检索；增强失败自动 fallback 传统路径。检索后做 `(kb_id, source)→file_id` 反查，前端引用可点击跳文档 |
| **tool_selection / tool_execution** | 优先用意图识别的 tool_name，否则按关键词推断；`registry.invoke()`；web_search 结果嵌 `<!--SOURCES:[...]-->` 机器可读块 |
| **agent_reasoning** | ReAct 循环推理节点：LLM 每轮输出 action（tool→写 pending_tool 路由到 tool_execution；final_answer→draft 路由到 answer_validation）；非法 JSON/未知工具记为 `_error` observation 让 LLM 自我修正，连续 3 次 → fallback；达 `AGENT_MAX_ITERATIONS=20` 强制基于已有观察生成 |
| **answer_generation** | 拼装 messages = 历史 + 检索上下文（3 种模板：增强块/传统文本/无上下文）+ 工具结果；上下文 8000 字截断守卫；空响应 → 平铺格式降级重试 |
| **answer_validation** | 硬规则（长度 <20 判 fail）+ LLM 自评 {passed, score, feedback}；regen<2 时重生成一次；LLM 崩溃直接接受 |
| **fallback_handler** | error_message 有值 → 礼貌道歉 → END |

### 6.3 关键工程细节

- **线程连接池隔离**：LangGraph 节点是同步函数，被 FastAPI `run_in_executor` 丢到 worker 线程。线程内 DB 查询一律走**随用随建的独立 engine + 用完 dispose**（`_run_in_thread_isolated`），与主事件循环的全局 async 连接池完全隔离，避免 Future 跨事件循环污染（症状：首个事务静默失效 → FK violation）。
- **技能上下文注入**：Skill 指令在意图识别、ReAct 推理、生成节点统一前置注入。

---

## 7. 增强检索五步流水线（★ 系统核心，app/rag/enhanced_retriever.py）

### 7.1 总览

```
retrieve(query, history, knowledge_base_ids)
  │
  ├─ 0. BM25 可用性检查（wait 0.5s，未就绪跳过 Path D——绝不阻塞主链路）
  ├─ 1. 查询结构分解（LLM，带 1h LRU 缓存 + 规则回退）
  ├─ 2. 四路并行检索（ThreadPoolExecutor 持久线程池）
  ├─ 3. 图谱感知融合重排（α×向量 + β×图谱距离 + γ×跨路共识 + δ×时效）
  ├─ 3.5 交叉编码器精排（可选，RERANKER_TYPE≠disabled）
  ├─ 3.6 answerability 评估（乘法否决「能回答压过主题像」）
  ├─ 3.7 逐子问题覆盖检查 coverage gate（缺口子问题针对性补充）
  ├─ 3.8 每子问题硬配额（弱子问题证据不被强子问题挤掉）
  ├─ 4. 知识块聚类（Union-Find 按共享图谱实体连通分量）
  └─ 5. 迭代缺口检测与补充（complex 且未全覆盖时，最多 2 轮 LLM）
```

### 7.2 第 1 步：查询结构分解

LLM 输出 `QueryDecomposition`：

```
{
  query_type: "factual"|"comparative"|"causal"|"multi_hop"|...,
  complexity: "low"|"medium"|"high",
  entities: [规范化概念（口语归一化、量化词不作实体、概念具体到可检索粒度）],
  themes: [...],
  relation_patterns: [...],
  sub_questions: [原子化精准子问题],
  sub_question_keywords: [与子问题一一对应的 2-3 个规范表述/近义词检索关键词]
}
```

要点：
- **LRU 缓存**：`_decomp_cache`（128 条，TTL 由 `ENHANCED_DECOMPOSITION_CACHE_TTL` 控制，默认 3600s）。同一 query 5 分钟/1 小时内结果完全一致；**带对话历史时不缓存**；**规则回退结果不缓存**（`_last_decomp_was_fallback` 标记，避免 LLM 恢复后长期吃劣质分解）
- **规则回退**：LLM 失败 → `_rule_based_decompose`（按标点/连词拆 1-2 个子问题）
- **不限制 max_tokens**：分解输出长度随查询复杂度变化（4 子问题+关键词可超 1500 token），限制会导致 JSON 截断 → 解析失败 → 意外退化成单子问题
- **拆分规则指向「分隔符」**：问号/分号/换行分隔的并列独立问题必须逐一拆成独立子问题（写「连接词」会让 LLM 漏拆）

### 7.3 第 2 步：四路并行检索

| 路径 | 方法 | 机制 |
|------|------|------|
| A 实体 | `_retrieve_entity_path` | 查询实体 → GraphCache 精确/子串匹配 → 一跳邻居，`GRAPH_QUERY_TOP_ENTITIES=3` 展开 |
| B 语义 | `_retrieve_semantic_path` | 逐子问题（+关键词扩展）→ Milvus 向量检索 → 反向追溯图谱实体（过滤 `_STRUCTURAL_NODE_TYPES` 结构节点） |
| C 关系 | `_retrieve_relation_path` | 关系模式 → GraphCache 关系链多跳（引导式） |
| D BM25 | `_retrieve_bm25_multi_path` | 遍历 `[query] + 子问题` 独立 BM25 检索合并去重（jieba 分词 + Okapi） |

关键规则：
- **BM25 命中必须向量二次确认**：BM25 分数（稀疏检索分，内部 ×0.5）不能冒充余弦相似度混入融合——`_vector_rerank_candidates` 对 BM25 命中重新算 embedding 余弦，低于 `RAG_SCORE_THRESHOLD` 的直接剔除，embedding 失败才回退 BM25 分
- 子问题混在一起检索会互相淹没 → 每个子问题独立检索
- 语义路径逐子问题取 `top_k_per_path` 且按 `RAG_SCORE_THRESHOLD` 过滤（宁缺毋滥不凑数）；**BM25 路径不加该阈值**（量纲不同）
- 多路命中同一文档 → `cross_path_hits` 计数 +1

### 7.4 第 3 步：四维融合重排

```
score = α×0.35 向量相似度 + β×0.25 图谱距离 + γ×0.25 跨路共识 + δ×0.15 时效性
```

- **跨路共识**：1 路命中 = 0.3 分，4 路全中 = 1.0（线性插值）——但**多次命中 ≠ 更正确**，这是后面 answerability 乘法否决要压制的对象
- **图谱距离**：精确相等 1.0；子串命中按重叠度降权 `0.6 + 0.3×min/max`；无命中 0.5 中性——防止宽泛概念子串命中拿满分把「主题像但答不了」的候选推上 Top1
- **时效性**：metadata 时间字段新鲜度

### 7.5 第 3.5 步：交叉编码器精排（可选）

`RERANKER_TYPE=local`（本地 bge-reranker-v2-m3 cross-encoder）/ `openai_compatible`（API）/ `disabled`。只对 top 候选精排，失败降级保持原序。

### 7.6 第 3.6 步：answerability 评估（验证层核心）

**「能回答」压过「主题像」——乘法否决**：

```
final_score = fusion_score × answerability   (answerability 下限 max(0.1, ans))
```

- LLM 批量判断每个候选对每个子问题的可答性，输出多标签 `{"answerability": 0.9, "sub_questions": [1,3]}`（空列表也要替换，清掉假阳性标注）
- 判定标准（领域无关）：**适用前提匹配**（主体/对象/条件/类型一致）、**排除性条件**（「除…外/不适用于」且事实落在排除范围 → 0）、**否定信号**（答非所问/主体不同/范畴不匹配 → 0）
- 下限 0.1：防 LLM 随机误判把正确候选清零（乘法下 ans=0 整体归零）
- `top_k` 必须 ≥ `final_top_k + 子问题数×min_per_sub`：所有可能进最终 Top-K 的候选（含保底候选）都必须过验证，否则第 top_k+1 名之后的伪相关保留原高分逃过验证
- LLM 失败降级返回原排序

### 7.7 第 3.7 步：coverage gate + 第 3.8 步：每子问题配额

- **coverage gate**：收集所有候选标注的 sub_questions 并集 → 找出 uncovered 子问题 → 用该子问题文本做针对性 semantic+BM25 补充检索（每路径 top-3 限流防候选爆炸）→ 补充候选同样过 answerability 乘法否决。返回 `(ranked, covered_all)`
- **全覆盖短路 gap fill**：`covered_all=True` 时跳过第 5 步的 1-2 次 LLM 调用（实测 multi_hop high 86.8s → 47s，gap_rounds 2→0）
- **每子问题硬配额**：`min_per_sub = max(1, final_top_k // 子问题数)`，每个子问题至少保底 N 条进最终 Top-K（原始 query 级也保底），剩余名额按融合分补满，输出 ≤ final_top_k。替代全局 `[:final_top_k]` 截断（全局截断会打破保底：min_per_sub×子问题数 > final_top_k 时弱子问题证据被截在窗外）

### 7.8 第 4 步：知识块聚类

- Union-Find 按**共享图谱实体**聚类成连通分量 → `KnowledgeBlock`
- `_collect_block_relations`：从 GraphCache 回捞块内实体间的边（`(source, relation, target)` 去重，上限 12 条）——曾因漏填导致前端「关系」区和 prompt 关系注入恒空
- 块标注多个子问题（并集），块 score 由内部 doc 融合分加权

### 7.9 第 5 步：迭代缺口检测与补充

- LLM 判断现有知识块对每个子问题的覆盖：`sufficient / insufficient / missing`
- gap_queries 必须**逐一对应** status=insufficient/missing 的子问题，用该子问题核心概念构造精准查询（不是重复原始 query，防 query drift）
- 线程池并行补搜 → 融合 + answerability → 合并；补完仍不满足标记「证据不足」
- 最多 `ENHANCED_MAX_GAP_ROUNDS=2` 轮

### 7.10 性能治理（重要设计约束）

| 手段 | 作用 |
|------|------|
| 持久线程池（6 worker） | 省每次创建销毁开销 |
| 查询分解 LRU 缓存 | 跳过重复 LLM 调用 |
| BM25 后台预构建（0.5s 等待 + 15s 超时） | Milvus 挂了也不阻塞主链路 |
| answerability 候选数上限 + max_tokens=1500 | 输出固定 JSON，限制延迟 |
| coverage gate 全覆盖短路 gap fill | 省 1-2 次 LLM 调用 |
| `LLM_TIMEOUT=30` | 检索路径串行 3-4 次调用，30×3+10=100s < 前端 axios 120s |

**超时预算公式：串行 LLM 链总超时 = 单次超时 × 调用次数，必须 < 上游超时阈值（前端 120s）**。

---

## 8. 基础 RAG 管线（app/rag 其他模块）

### 8.1 文档解析（parsers/）

- `parsers/router.py`：按文件类型 + 扩展名 + `preferred_parser` 分流：**本地解析器**（txt/md 直接读、pdf 用 pdfplumber 类库、docx 用 python-docx、图片走 OCR）vs **MinerU Pipeline API**（`MINERU_ENABLED`，Docker 旁路部署，GPU，返回结构化 JSON：段落/表格/标题层级）
- `parsers/models.py`：`ParsedDocument`（text/结构块/provenance/warnings）统一返回模型
- 解析失败且 `MINERU_FALLBACK_TO_LOCAL=True` 时降级本地解析
- OCR：`app/rag/ocr.py` 扫描件识别（图片 OCR + PDF 按页渲染再 OCR）

### 8.2 分块（chunker.py）——5 种策略

| 策略 | 机制 | 适用 |
|------|------|------|
| fixed | 固定窗口滑窗 | 通用 |
| recursive | 递归分隔符切分（段落→句子→词，语义边界断开） | 默认 |
| markdown | 按标题层级聚合，代码块不拆 | Markdown 文档 |
| parent_child | 小块（500）索引，命中回填父块（1500）上下文 | 长文档 |
| **legal** | 按「第X条」正则切分 + 章节标题前缀 `[第十二章 借款合同]` | 法律文本 |

- **法律文本自动检测**：`_looks_like_legal`（≥20 个「第X条」判定，避免误判普通文本），`CHUNK_STRATEGY` 未指定时自动走 legal
- 效果：民法典 272 chunks → 1313 chunks（平均 100 字/条），第 675 条独立成块且带章节上下文
- **改分块策略后必须重新索引才生效**（Milvus 里是旧 chunk），已提供 `POST /files/{id}/reindex` 入口

### 8.3 向量化（embeddings.py）

- BGE-M3（1024 维）：`local`（SentenceTransformer 本地加载，`EMBEDDING_MODEL_PATH=./models/bge-m3`）/ `ollama`（本地 Ollama）/ `openai_compatible`（远程 API）
- **并发安全**：`embed_texts` 用 `threading.Lock` 包住 encode——本地 embedding 单例 + `asyncio.to_thread` 并发调用必须加锁串行化（PyTorch CPU 推理竞争），锁后实测 8 线程并发 encode 峰值并发=1
- 批大小 64（减少 to_thread 次数与锁竞争）

### 8.4 向量存储与检索（vector_store.py / retriever.py）

- 主后端 **Milvus 2.5**：collection `rag_docs`，IVF_FLAT + IP（归一化余弦），metadata 带 `knowledge_base_id`/`source`/`chunk_id`；Chroma/Memory 供测试
- `MilvusRetriever.retrieve(query, top_k, score_threshold, knowledge_base_ids)`：先取 top_k 再按阈值过滤；`score_threshold=0` 转 `-1.0`（IP 分数可为负，0 表示不过滤）
- parent_child 策略下命中 child 块 → `_unwrap_parent` 回填父块上下文
- 知识库隔离：`knowledge_base_ids` 是硬过滤条件（metadata 精确匹配），空范围拒绝检索

### 8.5 BM25（bm25.py）

- jieba 分词（首次 import 延迟加载）+ Okapi BM25，进程级单例
- 数据源同步自 Milvus chunk（`sync_bm25_from_vector_store`）或直接喂 chunks
- **定位是锦上添花**：永远不阻塞主链路（后台构建 + 0.5s 等待 + 失败静默跳过）

### 8.6 上传-索引异步管线（knowledge_router.py `_run_ingestion`）

```
POST /bases/{kb_id}/upload → 202 + file_id（立刻返回）
  │
  ├─ 10% parsing     记录解析器选择 → 调 parser_router.parse()
  ├─ 30% chunking    存全文 text_content（供预览）+ chunk_parsed_document()
  ├─ 80% embedding   批量向量化 + Milvus upsert（每批 64）
  ├─ 90% graph       若 GRAPH_ENABLED：双层图谱抽取入库
  └─ 100% completed  前端 2s 轮询 GET /files 直到 status 非 processing
```

每阶段独立 session 提交 `progress/progress_message/processing_stage`，前端两阶段进度条（传输 0-100% + 索引 0-100%）。**uvicorn 必须 `--reload-exclude tests`**，否则改测试文件触发 reload 打断后台任务（文件卡 processing，前端按钮永久 disabled）。

---

## 9. 知识图谱子系统

### 9.1 双层抽取（graph_service.py）

`extract_graph_from_chunks` **双层都执行、结果合并**（不是降级替换）：

1. **LLM 语义抽取**（`_extract_graph_llm`，图谱可视化主体）：抽概念实体（person/organization/location/event/concept/method/artifact/data 8 类）+ 语义关系（属于/使用/导致/依赖…）。参考 LightRAG prompt 风格，强调命名一致性 + 输出语言跟随输入。并发控制 `GRAPH_LLM_CONCURRENCY=6`（asyncio.Semaphore + as_completed，**入库必须串行**——AsyncSession 非并发安全）
2. **结构抽取**（保留给检索用）：法律文本走 `_extract_graph_rule_based`（正则抽「第X条」+章节，**description 含正文**——检索 `match_entities` 靠它命中正文）；其余走 `_extract_graph_generic`（jieba NER，词性映射排除 eng 词性避免英文停用词淹没）

- **实体跨 chunk 去重按 `name`**（不是 `(name, type)`）：同名实体在不同 chunk 被抽成不同 type 时只保留第一个，否则 PG 同名多 type 实体会让前端 ECharts 渲染报 `dataIndex` 错误
- 判定用 `_looks_like_legal_chunks`（前 10 chunk ≥3 个「第X条」）
- 入库 PG `knowledge_entities` / `knowledge_relations` 表（kb_id 隔离）

### 9.2 进程级缓存（graph_cache.py）

- 检索路径 A/C 不直接访问 PG（避免 asyncio 事件循环冲突），而是把图谱加载进进程级 `GraphCache` 内存（实体→邻居、name 匹配）
- `match_entities`：name 精确命中优先 + **description 子串匹配兜底**（标题不含关键词的实体也能命中）

### 9.3 图谱 API

| 端点 | 说明 |
|------|------|
| `GET /bases/{kb_id}/graph` | 实体+关系。`limit`（默认 300）按**连接度 degree 降序**取 top-N（只返回选中节点间的边，全量 1500+ 会拖垮前端）；`semantic_only` 过滤结构编号实体（article/chapter）；孤立节点（degree=0）始终过滤 |
| `GET /bases/{kb_id}/graph/neighbors?entity=` | 点实体 → 右侧栏罗列邻居（方向箭头 out/in + 关系类型 + 描述），不受 top-N 截断影响 |
| `GET .../graph/config` / `status` / `entities` / `search` | 图谱构建配置查询、构建状态、实体列表、语义搜索 |
| `POST .../graph`（构建） / `DELETE .../graph`（重置） | 手动触发全库图谱构建 / 清空 |
| `POST /files/{id}/reindex` | 单文件重新索引（清旧向量+图谱 → 重新分块 → 重新抽取） |

### 9.4 前端可视化（KnowledgeView.vue）

- **最终方案：ECharts 5 的 2D `graph` 系列**（历经 cytoscape → 3d-force-graph → echarts-gl graphGL → 回退 2D；真 3D 依赖 WebGL，远程桌面/虚拟机不可用 → 整页空白）
- 伪 3D 立体感：节点 `shadowBlur/shadowOffsetY` 投影 + 边 `curveness` 弧线 + 节点大小按 degree 分级
- 交互：force 布局（`layoutAnimation:false`）、roam 缩放平移、拖节点、`focus: 'adjacency'` 点击高亮邻居、边 label 显示关系类型
- **图例/颜色动态生成**：从实际返回数据的 entity_type 去重后按出现顺序从 PALETTE 取色——绝不硬编码领域类型（用户硬性要求）
- 节点点击 → 右侧详情栏（label/type/description + 关联关系列表，点击邻居可继续跳转高亮）

---

## 10. 多智能体（DeepAgents 统一实现，app/agents）

### 10.1 设计思想

2026-08-26 起多智能体统一到 LangGraph 原生 DeepAgents：不再设独立拆解器，
**由主 Agent（create_react_agent）根据工具描述自主决定委派**——拆解/派发/汇总从
固定编排流程变成模型行为。原 Orchestrator-Worker（orchestrator.py / workers/ /
旧黑板）已退役删除；`AGENT_MODE=multi` 保留为 `deepagents` 的兼容别名（仅告警）。

### 10.2 组件

| 组件 | 职责 |
|------|------|
| **主 Agent**（`deep/agent.py`） | `create_react_agent` + checkpointer 会话记忆；持有 task / spawn_tasks / revise_plan 委派工具 |
| **task 工具**（`deep/task_tool.py`） | 单任务委派：SubAgent 内联执行，含熔断与降级（连续失败熔断后续委派） |
| **spawn_tasks**（`deep/planner.py`） | DAG 并行委派：`depends_on` 拓扑分层 + 线程池，依赖产出摘要自动注入；请求上下文快照重放 |
| **revise_plan** | 运行中动态重规划（结构化尾部：增/改/取消任务） |
| **SubAgent**（`deep/subagents.py`） | 内置 research-agent / coding-agent + 外部 JSON/YAML 配置；动态工具绑定（`*` / `except:` / `@tag`）+ 越权错误附可用清单 |
| **结构化黑板**（`deep/blackboard.py`） | 任务产出物 `{key, producer, summary, data, tags, version}` 摘要/全量两级共享 + 依赖订阅 |
| **统一事件流**（`agents/events.py`） | 请求级 trace/span + emit + 事件汇聚；跨线程调度用 `snapshot_request_context` 重放 |
| **委派持久化**（`delegation_service`） | 解析事件流 → Run/Task/AgentRun 三表落库（best-effort，复用既有表结构） |
| **可观测性**（`observability/tracing.py`） | OTel 可选 span（`tool.invoke.<name>` / `subagent.<type>` / `spawn_tasks`）+ 工具进度回调接入事件流 |

### 10.3 前端消费与历史演进

- 统一事件流经 `delegation_service.bridge_delegation_event` 桥接为既有任务面板协议（sub_tasks / status / worker_output / progress_summary），前端任务面板与 AgentActivity 直接复用，无新组件；委派树/工具时间线由事件 span（`spawn/<key>`、`subagent/<name>`）关联。
- 汇总阶段的 token 治理：旧 `_synthesize` 的 fast-model 压缩机制随 Orchestrator 退役，改由主 Agent 整合输出与委派结果截断规则承接。
- 历史：Orchestrator-Worker（LLM 拆解 → Worker 派发 → 汇总，含并行单例竞态等已知问题）于 2026-08-26 阶段 5 删除，相关测试迁移为 deep 路径等价物。


---

## 11. DeepAgents（app/agents/deep/）

- **主 Agent**（`build_main_agent`）：LangGraph 主图，可调用 task 委派工具把子任务交给 SubAgent
- **SubAgent**（`subagents.py`）：外部配置文件（`DEEP_SUBAGENTS_FILE`，JSON/YAML）定义，内置默认 research-agent / coding-agent；`run_subagent` 在独立 recursion_limit 内执行
- **深度研究开关**：`req.deep_research=true` 或 `AGENT_MODE=deepagents` 强制走此路径；前端「深度研究」开关
- **进度投影**（`progress.py` `DeepResearchProgressProjector`）：主 Agent 内部 step → 用户可读的 phase/text/status 摘要（如「委派给研究 SubAgent」「工具调用中」），通过 SSE `progress_summary` 事件流式推给前端；**原始思考、工具参数、结果正文不出服务端**（隐私与 token 治理）
- 对话终止：前端 AbortController → fetch abort → 服务端停止生成

---

## 12. Skill 系统、记忆与 MCP

### 12.1 Skill（app/skills + backend skill_config）

- **内置 Skill**：知识库研究、联网研究、数据分析、专业写作、法律分析（`catalog.py` 的 `SkillProfile`）
- **自定义 Skill**：CRUD（名称/用途/详细指令 + **最小权限工具白名单**），存 PG `skill_configs`，敏感配置加密
- **注入机制**：`context.py` 把选中 Skill 指令作为系统上下文注入（意图识别/ReAct/生成节点统一前置）；工具白名单在**后端执行层强制校验**——取消工具权限不是前端展示变化
- 最多组合 3 个 Skill；历史消息保留 Skill 名称快照
- 用户启用 Skill 时跳过 fast_intent 规则路径（尊重 Skill 的工具约束）

### 12.2 记忆（app/memory）

分层记忆管理：用户事实（facts）提取与注入（`prepare_context` 里 `_fetch_facts` 注入用户档案），服务于个性化回答。

### 12.3 MCP（app/tools/mcp）

- `mcp_router.py`：服务器启停 API（`GET/POST /mcp/servers`、`start/stop`）
- 常驻事件循环线程 + 同步桥接（避免 async 生命周期与工具执行层纠缠），支持 stdio / HTTP 双传输
- AI 调用需把 `mcp_<server>_<tool>` 全名加进 Worker 的 tool_names 白名单
- lifespan 启动时自动拉起 enabled 服务器（失败不阻塞应用启动）

---

## 13. 数据层与部署

### 13.1 PostgreSQL（pgvector/pg17，业务数据 + 图谱）

| 表（models_*.py） | 内容 |
|------|------|
| users | 用户（密码哈希） |
| conversations / messages | 会话与消息（metadata JSON 存 skills 快照、sources 引用） |
| knowledge_bases / knowledge_files | 知识库 + 文件（text_content 全文、进度、解析 provenance、parser 信息） |
| knowledge_entities / knowledge_relations | 图谱实体/关系（kb_id 隔离，description 含正文） |
| model_configs | 用户自定义模型（API Key AES 加密） |
| skill_configs | 用户自定义 Skill |
| agent_runs / agent_tasks | 多智能体 run 与任务状态 |
| memories | 用户事实记忆 |

启动时 `init_db()` 建表 + **增量列迁移**（`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`，开发用；生产应换 Alembic）。

### 13.2 Milvus 2.5

- collection `rag_docs`：向量 + metadata（knowledge_base_id / source / chunk_id / parent_id）
- etcd（元数据）+ 内部 MinIO（存储）依赖；Docker 重建时版本变化会导致 volume 数据不兼容 → 检查 `col.num_entities`

### 13.3 Redis / MinIO / 外部服务

- Redis：缓存/会话（当前主要为生命周期预留，核心状态在 PG）
- MinIO（easyrag-minio :9090 Console / 9091 API）：上传文件原始二进制对象存储
- Tavily：web_search 工具
- Ollama：本地 embedding（可选）

### 13.4 Docker Compose 拓扑（7 服务）

```
etcd:2379 ──→ milvus-standalone:19530 ──→ minio-s3:9000/9001
postgres(pgvector/pg17):5432     redis:7:6379
minio:9090(Console)/9091(API)    mineru-api:18000(→容器8000, GPU, 仅 127.0.0.1)
```

数据卷在 `volumes/`（Docker Desktop WSL2 下注意 C 盘写爆，可 junction 迁移到 D 盘）。MinerU 已合并进主 compose（`MINERU_VERSION=3.4.4`，清华源拉模型，CUDA_VISIBLE_DEVICES=0）。

---

## 14. 前端架构（frontend/src）

### 14.1 页面与状态

| 视图 | 职责 |
|------|------|
| ChatView.vue | SSE 逐 token 渲染、引用来源可点击跳转、模型切换、Skill 标签、任务状态栏（可收起）、停止生成、深度研究开关 |
| KnowledgeView.vue | 知识库 CRUD、两阶段上传进度条、文件预览/删除/重新索引、**检索测试工作台**（basic/enhanced 双模式 + 子问题 tab + 图谱可视化 + 邻居详情） |
| EvaluationView.vue | 检索评估运行列表/明细（HitRate/MRR/avg_score） |
| LoginView / RegisterView / LayoutView | 认证与布局 |
| AgentActivity.vue / ProgressJournal.vue | 多智能体任务面板 / 深度研究进度日志 |

- Pinia：`auth.js`（token/user）+ `chat.js`（会话列表、活跃会话）
- `api/index.js`：axios 封装（JWT 拦截器、401 跳登录、`get` 已解包 `r.data`）+ `streamChat()`（fetch + ReadableStream 手写 SSE 解析，`AbortController` 支持停止生成）
- 路由守卫：未登录跳 /login

### 14.2 检索测试工作台（KnowledgeView）

- `mode=basic`：纯向量检索（top_k + score_threshold 生效）
- `mode=enhanced`：五步流水线，返回 `query_decomposition`（子问题卡片）+ `knowledge_blocks`（块内容/实体/关系/来源）+ `gap_rounds`
- **子问题 tab 切换**：`selectedSubQuestion`（-1=全部）+ `filteredBlocks` computed（按块标注的 sub_questions 过滤）——「全部」数量 ≠ 子问题数量之和是设计如此（一块可回答多个子问题 + 无标注块只在「全部」出现），不是 bug
- 图谱页：ECharts 2D graph（见 9.4）

---

## 15. 配置体系

`app/core/config.py`（pydantic-settings，`Settings` 单例 `get_settings()`）：

- **`.env` 优先级高于代码默认值**（`SettingsConfigDict(env_file=".env")`）——改 config.py 默认值后必须 grep .env 确认无同名覆盖（`LLM_TIMEOUT`/`LLM_MAX_TOKENS` 都踩过）
- 关键开关：
  - `ENHANCED_RETRIEVAL_ENABLED=true`：增强检索总开关（默认 false 走传统路径）
  - `GRAPH_ENABLED=true`：上传时抽取图谱
  - `RERANKER_TYPE=disabled|local|openai_compatible`：精排
  - `AGENT_MODE=auto|single|multi|deepagents`：执行路径
  - `FAST_INTENT_ENABLED=true`：规则快速意图
  - `CHUNK_STRATEGY=recursive|fixed|markdown|parent_child|legal`
  - `EMBEDDING_TYPE=local|ollama|openai_compatible`
- 模型目录：DeepSeek / MiniMax / Qwen / GLM 内置配置（浏览器只拿到 public ID，endpoint/API key 永远留在服务端）
- 启动时 `no_proxy` 注入：把 127.0.0.1/localhost 加入 no_proxy（否则 grpcio 经 Windows 系统代理连 Milvus 会握手超时；外部 API 仍走代理）

---

## 16. 评测体系

### 16.1 本地确定性指标（evaluation_service + retrieval_metrics）

- `POST /evaluation/runs` 创建命名评估运行：数据集（query + 期望命中的 reference 文件/子串）→ 逐条执行 basic/enhanced 检索 → 计算 **HitRate / MRR / avg_score** → 逐条命中明细（哪些期望命中了、分数多少）
- 前端 EvaluationView 展示运行列表与明细，可对比不同检索配置的质量

### 16.2 可选 Ragas（ragas_evaluator + ragas_worker）

- `RAGAS_ENABLED=false` 默认关闭——API 进程永远不 import Ragas（避免升级主服务依赖）
- `RAGAS_EXECUTION_MODE=process`：独立 venv 子进程执行（`RAGAS_PYTHON_EXECUTABLE`），指标 `id_context_precision,id_context_recall`

---

## 17. 关键设计决策（含工程教训）

1. **意图识别分流**：先分类再干活，避免无差别检索；fast_intent 规则路径把简单问题零成本分流（省 LLM 调用）
2. **ReAct 子图**：复杂任务/低置信度走推理-工具循环，普通任务直通管道，兼顾质量与延迟
3. **增强检索的验证层**：语义相关 ≠ 能回答 → answerability **乘法否决**（多次命中奖励放大错误，乘法让一个维度为 0 整体归零），配合 coverage gate + 每子问题保底 + 查询关键词扩展，把「主题像但答不了」的伪相关压出 Top-K
4. **BM25 是锦上添花**：永不阻塞主链路（后台构建 + 0.5s 等待 + 超时静默跳过）；BM25 分数必须向量二次确认才能进融合
5. **SSE 与同步路径 context 逻辑必须一致**：改一处必须改另一处（截断守卫、空答案兜底、query_rewrite 下沉到图入口）
6. **空答案五层兜底**：max_regen=2 + 增强 context 8000 字截断 + 空→平铺格式降级 + chat_sync 空响应重试 + 流式 0 token→同步兜底（根因是代理端点对法律内容间歇性返回空 HTTP 200）
7. **异步索引**：202 + 阶段进度轮询；每阶段独立 session 提交；uvicorn --reload-exclude tests
8. **线程连接池隔离**：executor 线程内 DB 查询用独立 engine 用完 dispose
9. **图谱可视化限流**：后端按连接度 top-N + 只返回选中节点边；前端 2D echarts（真 3D 在远程桌面 WebGL 不可用）
10. **图例/实体类型动态化**：不硬编码领域类型（用户红线），任何领域的知识库都能自适应
11. **本地 embedding 加锁串行**：单例 + to_thread 并发 = 必须锁
12. **图谱实体按 name 去重**：同名多 type 会让 ECharts 报 dataIndex 错误
13. **MinerU 旁路部署**：解析服务 Docker 独立运行（GPU），不污染主 Python 环境；失败降级本地解析
14. **Skill 执行层强制**：工具白名单在后端校验，指令注入系统上下文——自定义 Skill 无需改代码/重启

---

## 18. 一次完整对话的端到端时序（含检索+图谱）

```
用户: "公司未给员工买社保，员工受伤了怎么办？"
 │
 ▼ ChatView (前端)
 │  POST /api/v1/chat/stream  {query, model_id, skill_ids, conversation_id}
 ▼ chat_router.send_message_stream
 │  JWT 鉴权 → 解析模型/Skill → 会话/消息落库 → 压缩历史 → 知识库范围
 │  use_deep 判定(命中"劳动"+"赔偿"两领域) → agent_mode="deepagents"
 ▼ AgentService._run_deep（executor 线程）→ 主 Agent（create_react_agent）
 │  模型自主委派：spawn_tasks [task-1 工伤认定条件, task-2 赔偿标准]
 │  depends_on 拓扑分层 + 线程池并行（请求上下文快照重放）
 │    ├─ research-agent → kb_search（授权范围取自请求级 ContextVar）
 │    │    增强流水线：查询分解 → 四路并行 → 融合重排 → answerability
 │    │    → sources 带 kb_id+file_id
 │    └─ research-agent → LLM 条文引用分析（结构化输出）
 │  结构化黑板收集产出 → 主 Agent 整合最终回答；
 │  委派事件流 → 桥接任务面板协议 + Run/Task/AgentRun 落库（best-effort）
 ▼ SSE 事件流
 │  conversation_id → sub_tasks/worker_output(委派面板) → delta(逐token) → done(sources+intent+run_id)
 ▼ 前端
 │  回答逐 token 渲染，引用来源可点击 → 跳转文档预览
 │  任务面板展示委派进度（done/total）与各子智能体产出
```

---

## 19. 相关文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 宏观架构总览（技术栈/目录/API 概览/演进路线） |
| [../PROGRESS.md](../PROGRESS.md) | 逐次迭代的演进记录（含历史踩坑） |
| [plans/](./plans/) · [specs/](./specs/) | 设计稿与规格说明 |
| [../README.md](../README.md) | 快速开始 |
| [ragas-evaluator.md](./ragas-evaluator.md) | Ragas 评估部署 |
| [../deploy/mineru/README.md](../deploy/mineru/README.md) | MinerU 解析服务运维 |
