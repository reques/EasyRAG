# EasyRAG 规范化 RAG 评测体系

> 目的：把「检索评估」从临时脚本升级为一套**数据、指标、执行、报告四层标准化**的体系。
> 评测结果可复现、可对比、可解释，能回答「这次检索配置改动到底变好还是变坏、差在哪」。

---

## 1. 为什么需要一套体系，而不是一个脚本

RAG 系统的质量由两个独立环节决定，必须分开评估：

| 环节 | 回答的问题 | 典型故障 |
|---|---|---|
| 检索（Retrieval） | 相关的知识片段有没有被召回、排得够不够靠前 | 漏召回、相关片段排在很后面、召回无关内容 |
| 生成（Generation） | 基于召回的上下文，答案是否忠实、相关、完整 | 幻觉、答非所问、信息缺失 |

把两个环节混在一起打分，出了问题无法定位是「没检索到」还是「模型没用上」。所以规范化的第一步是**分层**：检索阶段用确定性指标 + RAGAs 语义指标，生成阶段用 Faithfulness / Answer Relevancy（本仓库生成阶段为预留项）。

评测体系的四层：

1. **评测数据层（Golden Set）**：可复用、可版本化、可导入导出的评测数据集；
2. **指标层**：分层指标体系，确定性指标与 LLM 语义指标互补；
3. **执行层**：逐条真实检索 + 环境快照，保证可复现、可 A/B；
4. **报告层**：聚合指标 + 逐条明细 + 失败分析 + Markdown 报告。

---

## 2. 评测数据层：Golden Set 怎么构造（最关键）

### 2.1 一条用例长什么样

```json
{
  "question": "消费者享有公平交易的权利依据哪一条？",
  "expected_file_id": "uuid",
  "expected_chunk_ids": ["sha256-..."],
  "reference_answer": "《消费者权益保护法》第十条……",
  "expect_miss": false
}
```

### 2.2 为什么 reference 必须是「与问题相关的 chunk 集」，而不是整份文件

RAGAs 官方的 `reference_contexts` 语义是：**生成该问题答案所依据的证据片段**（通常 1~5 条 chunk）。如果偷懒把整份文件的所有 chunk 都当成相关集，会出现两个系统性失真：

- **Recall 被压扁**：分母变成整份文件的 chunk 数。一份 179 条的法规文件、K=5 时，即使命中了正确的条文，Recall 上限也只有 `5/179 ≈ 0.028`；
- **Precision 被抬高**：只要检索到该文件的**任意** chunk 都算命中，Precision 不再反映「是否命中真正相关的条文」。

这就是「检索『民法典第10条』结果不相关，但 Recall 却接近 0」这种反直觉结果的根因——不是指标错了，是喂给指标的 reference 口径错了。

### 2.3 Golden Set 的三种样本

| 类型 | 构造方式 | 用途 |
|---|---|---|
| 正样本 | 问题 + 与该问题真正相关的 1~5 条 chunk + 参考答案 | 度量召回/排序能力 |
| 负样本（`expect_miss=true`） | 问题与目标文件无关，期望不命中 | 度量误报率（false positive rate） |
| 难例（Hard Negative） | 语义相似但答案不同的文件/chunk | 检验检索的判别力 |

### 2.4 标注辅助（本仓库实现）

`POST /evaluation/chunk-candidates`：给定「问题 + 目标文件」，先真实检索，把该文件内被召回的候选 chunk 连同片段和分数返回。人工从候选中勾选真正相关的几条，作为 `expected_chunk_ids`。这比手抄 chunk id 现实得多，也是业界 golden set 构建的常见做法（先检索、再人工确认）。

### 2.5 数据集管理

- `evaluation_datasets` 表持久化；同名保存自动递增 `version`，天然支持「v1 / v2 迭代」；
- 可导出 JSON 离线评审，也可导入重建；
- 一次运行记录 `dataset_id`，同一数据集可反复跑不同配置做对比。

---

## 3. 指标层：指标体系与取舍

### 3.1 确定性检索指标（无 LLM，可回归）

基于二值相关性（相关 chunk 命中即 1），在 `backend/services/retrieval_metrics.py`：

| 指标 | 公式 / 含义 | 测什么 | 陷阱 |
|---|---|---|---|
| HitRate@K | 相关结果是否出现在 Top-K | 有没有召回 | 不看排序 |
| MRR@K | 第一个相关结果的倒数排名 | 最快多久命中 | 只关心第一个 |
| Recall@K | 召回的相关数 / 总相关数 | 有没有漏 | 依赖 reference 口径（见 2.2） |
| Precision@K | 召回的相关数 / K | 有多少噪声 | K 固定时数值天然偏低 |
| nDCG@K | 排序加权折扣收益 / 理想排序 | 排序质量 | 二值相关性下退化为位置加权 |

### 3.2 RAGAs 指标

| 指标 | 类型 | 测什么 |
|---|---|---|
| `IDBasedContextPrecision` | 无 LLM | 检索结果中相关 chunk 的占比（集合匹配，不看排序） |
| `IDBasedContextRecall` | 无 LLM | 相关 chunk 被召回的占比（集合匹配） |
| `ContextPrecision` | LLM | **排序敏感**的平均精确率：相关 chunk 越靠前分越高 |
| `ContextRecall` | LLM | 以参考答案为代理，拆 claims 判断被检索上下文支持的比例 |

**取舍**：ID 指标便宜、确定，但**完全不惩罚排序**（相关 chunk 排在最后一名也是满分），也无法处理同义改写；LLM 指标贵、有轻微非确定性，但能评估排序和语义相关性。规范做法是：日常回归跑 ID 指标，发布前/关键实验跑 LLM 指标。

### 3.3 指标集怎么选（本仓库实现）

`.env` 的 `RAGAS_METRICS` 控制指标集；单次运行可用 `ragas_metrics` 覆盖全局配置，同一数据集分别跑 ID 版和 LLM 版做对比。

---

## 4. 执行层：可复现与 A/B 实验

### 4.1 环境快照

每次运行把以下配置写进 `metrics_json.run_metadata`，保证「这个分数是在什么环境下跑出来的」：

- `k`（Top-K）、`chunk_strategy`（分块策略）、`embedding_type` + `embedding_model`
- `score_threshold`、`enhanced_retrieval`、`graph_enabled`
- RAGAs 版本与指标集

### 4.2 典型 A/B 实验流程

1. 上传/重建索引：换 `CHUNK_STRATEGY` 或 `EMBEDDING_TYPE`；
2. 对同一 Golden Set 跑一次命名运行（`POST /evaluation/runs`）；
3. 对比两次运行的聚合指标与逐条明细，定位变化来源；
4. 用失败分析（见 5）确认是「召回变少」还是「排序变差」。

面试时可举例：`recursive` 分块 vs `legal` 按条分块，同一数据集下 Recall@K 从 0.5 → 0.9，同时 Precision 不变，说明收益来自分块粒度匹配法律条文结构。

---

## 5. 报告层：失败分析怎么解读

报告（`GET /evaluation/runs/{id}/report`）自动把逐条结果归类为三类失败：

| 失败模式 | 判定 | 说明 |
|---|---|---|
| Missed（未命中） | 相关 chunk 一条都没召回 | 大概率是 embedding 语义偏差或 query 表述问题 |
| Low Recall（低召回） | 召回率 > 0 且 < 50% | 相关片段被召回但不全，可能是 top-k 太小或分块截断 |
| False Positive（误报） | 负样本命中了目标文件 | 检索判别力不足，需要 hard negative 或阈值 |

**解读原则**：先看 Missed 有没有、再看排序（MRR/nDCG）、最后看误报。指标只告诉你「哪里差」，逐条明细和失败分析告诉你「差在哪条、为什么」。

---

## 6. 代码映射

| 环节 | 文件 |
|---|---|
| 确定性指标 | `backend/services/retrieval_metrics.py` |
| 评测执行引擎 | `backend/services/evaluation_service.py` |
| RAGAs 适配器 / worker | `backend/services/ragas_evaluator.py`、`backend/services/ragas_worker.py` |
| Golden Set 数据层 | `backend/services/evaluation_datasets.py` |
| Markdown 报告 | `backend/services/evaluation_report.py` |
| API（数据集/运行/报告/候选） | `backend/server/routers/evaluation_router.py` |
| 前端页面 | `frontend/src/views/EvaluationView.vue` |

---

## 7. 面试讲解话术（90 秒版本）

1. **一句话**：「我把检索评估做成了四层标准化的体系——数据层用带精确 chunk 标注的 Golden Set，指标层区分确定性指标和 RAGAs 语义指标，执行层记录环境快照保证可复现，报告层自动做失败分析。」
2. **讲一个踩坑**：「最早我把整份文件当相关集，Recall 永远上不去，后来发现是 reference 口径不符合 RAGAs 官方语义——reference 应该是生成答案所用的那几条证据 chunk，不是整份文档。」
3. **讲取舍**：「ID 指标便宜但不看排序，LLM 指标准但贵且有非确定性，所以日常回归用 ID 版，关键实验跑 LLM 版，用同一个数据集做 A/B。」
4. **讲落地**：「前端可以在知识库『RAG 评估』页维护用例、跑运行、下报告；后端全部有 API 和单测。」

---

## 8. 后续规划（端到端评测）

- 生成阶段指标：`Faithfulness` / `AnswerRelevancy` / `AnswerCorrectness`（需要接入生成链路，输出 `response` 后评估）；
- 评测集自动构建：用 LLM 从文档生成 question-reference 对，再人工抽检（半自动 golden set）；
- 评测集回归门禁：CI 里对同一数据集跑回归，指标低于阈值即失败。