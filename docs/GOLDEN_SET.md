# Golden Set 规范化构建指南

> 目的：把「评测集从哪来、长什么样、怎么标注、怎么验证」固定成可复现的流水线，
> 使任何一篇语料都能产出**可直接导入、可 A/B、可回归**的 Golden Set。
>
> 配套文档：`docs/RAG_EVALUATION.md`（四层评测体系总览）。本文只讲**数据层怎么造**。

---

## 1. 现状诊断：为什么现在"不标准"

对当前仓库与已落库数据做了一次完整核查，结论如下。

### 1.1 已有评测集的问题

`eval/aventro/`（221 条）与 `eval/zx_bank/`（321 条）是用一次性脚本攒出来的，存在四类系统性缺陷：

| 缺陷 | 具体表现 | 后果 |
|---|---|---|
| **无 chunk 级标注** | 全部 `expected_chunk_ids: []` | 评分退化为「整份文件兜底」（`reference_mode=file`）。这正是 `RAG_EVALUATION.md` §2.2 警告的口径错误：Recall 分母变成整份文件的 chunk 数，Precision 虚高 |
| **无负样本** | 全部 `expect_miss: false` | 误报率（false positive rate）无法度量 |
| **标识符不稳定** | 用 `expected_filename` 而非 `expected_file_id` | 文件名一旦有前导空格/全半角差异就整批导入失败（见 §4） |
| **两份集子格式不一致** | aventro 用 `.md`，zx_bank 用 `.docx` | 同一套指标无法横向对比 |
| **未去重 / 类型未归一** | zx_bank 有 5 条 `duplicate_question=true` 仍在集内；frozenset 分类存在 `Supporting`/`Eligibility`/`Safety` 等单例类型 | 重复样本给同一能力重复计分；类型维度聚合出现单例桶 |

### 1.2 更关键的问题：这两份评测集**现在跑不了**

直连运行中的 Postgres 核查：

```
knowledge_bases            -> 仅 1 条：CloudWay-24 (9108e914-…)，37 个文件
evaluation_datasets        -> 0 条
evaluation_runs            -> 0 条
```

即：**已入库的知识库只有 CloudWay-24，而 aventro / zx_bank 的 KB 根本不存在**，
两份评测集无法执行。而 CloudWay-24 恰恰是唯一**没有**评测集的——因为上游
RAG-Multi-Corpus 的 benchmark 口径文件（`Dataset categories - queries.csv`，1088 条）
**整篇剔除了 Cloudway**（master 文件 1252 条 = 该 1088 条 + Cloudway 164 条）。

> 换言之：语料里唯一被索引的企业，没有评测数据；有评测数据的企业，没有索引。
> 这是本次规范化要解决的第一个实际问题。

### 1.3 上游语料的固有缺陷（必须在构建期修掉）

| 缺陷 | 数量 | 处理方式 |
|---|---|---|
| 文件名脏数据（前导空格 / 双空格 / `:`↔`_`） | 5 处 | 归一化折叠匹配（§3.2） |
| 引用的证据文件在语料中完全不存在（`Account Close Guide.md`） | 1 个文件 / 6 条 query / 19 条 fact | 计入 `*_excluded.csv`，不静默丢弃 |
| 问题重复 | 1252 行 → 1071 条唯一问题（181 条重复） | 构建期检测并在 review 表标注 |
| Query Type 非规范标签 | 4 条（`Supporting`/`Eligibility Inquiry`/`Eligibility`/`Safety`） | 折叠到 7 类规范类型，原值保留在 `query_type_raw` |
| 企业名不一致 | `Cloudway 24`（CSV） vs `CloudWay-24`（目录） vs `CloudWay24`（正文） | `normalize_enterprise()` 折叠 |
| 上游 benchmark 结果被子集化 | 1088 条只评了 **787 / 786** 条（按企业截断到 200） | 不可作为逐条对比基线；记录在案 |
| **证据事实是 LLM 复述，非原文** | 仅 **25.9%** 可在源 md 中原样命中 | **决定了标注必须是语义匹配，不能字符串匹配**（§3.3） |

最后一条是整个方案的设计前提，值得强调：`Supporting Facts` 里的 `text` 是
LLM 对原文某一段的**改写**（例如原文 `- **Flat 15% Discount** on base airfare…`，
事实写成 `Flat 15% Discount on base airfare…`），因此不存在"把 fact 原文抠出来
当 chunk 标注"这条捷径。

---

## 2. 目标契约：一条用例到底长什么样

以下取自运行中后端的 OpenAPI（`http://127.0.0.1:8000/openapi.json`）与
`backend/server/routers/evaluation_router.py`，是**唯一的权威口径**。

### 2.1 用例字段

```jsonc
{
  "question": "…",                  // 必填，非空，<=4096
  "expected_file_id": "<uuid>",     // 必填（即便是负样本，它是误报目标）
  "expected_chunk_ids": ["<64hex>"],// 可选，<=32 条；留空则退化为整文件兜底
  "reference_answer": "…",          // 可选，<=100000；context_recall 需要
  "expect_miss": false              // 可选，默认 false
}
```

### 2.2 数据集级

```jsonc
{ "name": "…", "description": "…", "cases": [ /* 1..1000 条 */ ] }
```

- 身份 = `(name, knowledge_base_id)`；同名重复导入**原地更新并 version+1**；
- 导入文件：`.json` / `.csv`，UTF-8，<=5 MiB；
- **导入是全或无**：任何一行报错即 422，整批失败（`evaluation_router.py:1062`）；
- 导入时可用 `expected_filename` 代替 `expected_file_id`，由后端按**精确字符串**
  在目标 KB 内解析，重名或找不到即报错。

> 结论：**线上产物一律写 `expected_file_id`**，`expected_filename` 只作人类可读冗余。
> 前者不受文件名脏数据影响。

### 2.3 chunk_id 的生成方式（决定了离线标注可行性）

`app/rag/retriever.py:28`：

```python
sha256(f"{kb_uuid}\x1f{source}\x1f{chunk_index or ''}\x1f{content}")   # \x1f = U+001F
```

关键事实：**Milvus 集合只持久化 `content / source / knowledge_base_id / vector`**
（`app/rag/retriever.py:366-395`），既没有 `file_id` 也没有 `chunk_index`。
因此在该部署下 `chunk_index` 恒为空串，chunk_id 完全由
`(kb_id, source, content)` 决定 —— **可以离线精确复算**。

本项目据此实现了复算并与容器内真实函数逐条比对，**374/374 完全一致**。

> 注：`docs/RAG_EVALUATION.md` 早期把 chunk id 写成 `"sha256-..."`，实际输出**不带前缀**。
> 另需注意：memory / chroma 后端会在 metadata 里带 `chunk_index`，其 chunk_id 与
> 「整文件兜底」路径算出的 id **不一致**，会导致留空 `expected_chunk_ids` 的用例
> chunk 指标恒为 0。这也是必须显式标注 chunk 的原因之一。

---

## 3. 构建流水线：三段式

```
  ┌─ Stage 1  prepare ────────────┐   ┌─ Stage 2  build ─────────────┐   ┌─ Stage 3  verify ──────┐
  │ 容器内执行                     │   │ 宿主机执行                    │   │ 容器内执行              │
  │ • 导出 KB 全部 chunk + 向量     │   │ • 解析 master query CSV       │   │ • 真实 importer 解析    │
  │ • 用索引同一 embedder 嵌入      │──▶│ • 折叠解析文件名 → file_id     │──▶│ • 真实检索跑一遍         │
  │   questions + facts            │   │ • 证据↔chunk 语义匹配          │   │ • 结构 + chunk 完整性    │
  │ • chunk_id 用真实函数生成       │   │ • 生成 golden set + review    │   │                        │
  └────────────────────────────────┘   └───────────────────────────────┘   └────────────────────────┘
     eval/golden_prepare.py                eval/build_golden_set.py            eval/verify_import.py
                                                                               eval/run_golden_baseline.py
                                                                               eval/validate_golden_set.py
```

### 3.1 Stage 1 — `eval/golden_prepare.py`（容器内）

必须在容器内跑，理由有二：

1. **chunk_id 必须用真实函数生成**，杜绝公式漂移；
2. **必须用建索引时的同一个 embedder**（本部署为 ollama `bge-m3:latest`）嵌入
   问题与证据事实，否则相似度不可比。

产出：`golden_chunks.json`（source / content / chunk_id / vector）、
`golden_embeddings.json`（1071 条唯一问题 + 1336 条唯一事实的向量）。

### 3.2 Stage 2 — `eval/build_golden_set.py`（宿主机）

1. **企业筛选**：`normalize_enterprise()` 折叠 `Cloudway 24` / `CloudWay-24` / `cloudway24`。
2. **文件名折叠解析**（`_fold()`）：NFKC → 去 markdown 标记/链接 → 删标点 →
   `_`/`-`/空白统一折叠 → casefold。已实测可修复全部 5 处脏文件名，
   且不会把不同文件误合并。
3. **证据 ↔ chunk 语义匹配**：对每条 fact 取该文件内的 chunk，
   按余弦相似度排序：
   - `sim >= --threshold (0.70)` → 接受；
   - 若一条都没过阈值但最高分 `>= --floor (0.60)` → **兜底保留最佳 chunk**，
     以保证用例不退化为整文件兜底；
   - 每例最多保留 `--max-chunks (5)` 条（合同上限 32）。
4. **reference_answer** = 该问题全部证据事实原文去重拼接（保持顺序）。
5. **负样本**：从**其他企业**的问题池取题，在本 KB 内挑选
   「与该问题所有 chunk 相似度最大值最小」的文件作为 `expected_file_id`，
   且要求该最大值 `<= --negative-max-sim (0.55)` 才接受 —— 即**可证明**该文件
   不是该问题的证据。取题按「企业→问题类型」两级轮转，避免负样本被单一来源主导。
6. **类型归一**：非规范标签折叠进 7 类，原值写入 `query_type_raw`。
7. **排除**：解析不到文件的问题写入 `*_excluded.csv`（含原因），不静默丢。

### 3.3 为什么是语义匹配而不是字符串匹配

实测（全语料 1577 条事实）：

| 判定方式 | 命中率 |
|---|---|
| 事实原文是源 md 的子串 | 25.9% |
| 事实原文是 chunk 文本的子串（含 markdown 标记归一后） | ~29% |
| **证据 ↔ chunk 余弦相似度 top-1** | 中位数 **0.859**，92.4% ≥ 0.70 |

事实是改写，字符串匹配必然大量漏标；而语义相似度稳定可判别。
这与上游 benchmark 自己的做法一致（其 readme：*"Supporting facts are matched
with results for relevancy using LLM gpt 4.1"*），也是 EasyRAG 文档推荐的
「先检索、再人工确认」（`POST /evaluation/chunk-candidates`）的离线等价物。

---

## 4. 已验证产出：CloudWay-24

当前唯一已入库的 KB，因此是唯一能端到端跑通的集子。

```
知识库      CloudWay-24  9108e914-0294-4d92-99f9-01ae195ef284   37 个文件 / 374 个 chunk
语料 query  164 条（master CSV 中 Cloudway 24 全部）
产出        eval/cloudway24/cloudway24_golden_set.json   204 条
            ├─ 正样本 164 条（100% 带 chunk 级标注）
            └─ 负样本  40 条（可证明与该文件无关，max_sim 0.214–0.352）
排除        0 条
待人工复核  35 条（最佳相似度 < 0.80）
```

标注质量：`sim_max` 中位数 **0.8775**、最小 **0.6336**、平均每例 **2.89** 个 chunk。

### 4.1 校验结果

**结构校验**（`eval/validate_golden_set.py`，11 项全过）：字段仅含合同键、
`expected_chunk_ids` 无重复且 ≤32、负样本不带 chunk/答案、
`description` ≤512、`cases` ≤1000、问题与答案长度合规。

**chunk 完整性**：474 个引用 chunk_id（252 个去重）**全部存在于线上索引**，
且均为 64 位 sha256。

**真实 importer 校验**（`eval/verify_import.py`，在容器内调用
`backend.services.evaluation_import.parse_dataset_import`）：
**204/204 条解析通过，0 error**。

**真实检索跑通**（`eval/run_golden_baseline.py`，调用线上
`run_evaluation`，k=6）：

| 指标 | 值 |
|---|---|
| `reference_mode` 分布 | `chunk_ids` 164 / `negative` 40 —— **无一条退化为 file 兜底** |
| chunk HitRate@6 | 0.6127 |
| chunk MRR@6 | 0.5152 |
| chunk Recall@6 | 0.4306 |
| chunk Precision@6 | 0.2067 |
| chunk nDCG@6 | 0.4167 |
| file HitRate@6 | 0.7794 |
| 失败分析 | missed 5 / low_recall 27 / **false_positive 0** |
| 有 chunk 级命中的正样本 | 125 / 164 |
| 误命中目标文件的负样本 | **0 / 40** |

基线快照存于 `eval/cloudway24/cloudway24_baseline_run.json`（含 `run_metadata`
环境快照，可用于横向对比）。

> 与上游 benchmark 的 `Recall@6 ≈ 0.93` 不可直接比较：上游用的是 agentic chunk
> + LLM 相关性判定（口径更粗），且只在 4 个企业上评了 787 条、排除了 Cloudway；
> 本集是 EasyRAG 自身 `recursive` 500/50 分块下的**严格 chunk 级**口径，并含负样本。

---

## 5. 怎么用

### 5.1 复现 CloudWay-24（三条命令）

```bash
mkdir -p eval/_work          # 中间件目录（~40MB，可随时重建，不必入库）

# Stage 1 — 容器内：导出索引 + 嵌入（约 7 分钟，产出 ~40MB 中间件）
docker cp eval/golden_prepare.py easyrag-backend:/tmp/
docker cp "<语料>/datasets/Dataset categories - queries_01122025.csv" easyrag-backend:/tmp/master_queries.csv
docker exec easyrag-backend python /tmp/golden_prepare.py \
    --kb-id 9108e914-0294-4d92-99f9-01ae195ef284 --queries /tmp/master_queries.csv
docker cp easyrag-backend:/tmp/golden_chunks.json     eval/_work/
docker cp easyrag-backend:/tmp/golden_embeddings.json eval/_work/

# Stage 2 — 宿主机：生成
python eval/build_golden_set.py \
    --enterprise "Cloudway 24" --kb-id 9108e914-0294-4d92-99f9-01ae195ef284 \
    --kb-files eval/cloudway24/cloudway24_kb_files.json \
    --chunks eval/_work/golden_chunks.json \
    --embeddings eval/_work/golden_embeddings.json \
    --out-dir eval/cloudway24 --negatives 40

# Stage 3 — 校验
python eval/validate_golden_set.py --golden eval/cloudway24/cloudway24_golden_set.json
docker cp eval/verify_import.py easyrag-backend:/tmp/ && \
docker exec -w /app easyrag-backend python /tmp/verify_import.py /tmp/gs.json
```

### 5.2 导入到系统

前端：知识库 → **RAG 评估** → 导入评测集，选择 `cloudway24_golden_set.json`。
或走 API：`POST /api/v1/evaluation/datasets/import`（`file` + `kb_id`）。
导入后用 `POST /api/v1/evaluation/runs`（`dataset_id` + `top_k`）跑运行、取报告。

### 5.3 人工复核闭环（剩余工作）

自动标注只保证「**证据落在正确的文件里**」且「**最佳 chunk 达到阈值**」，
不保证「**选中的每一条 chunk 都恰好是该问题的证据**」。
因此 `cloudway24_qa_review.csv` 里 35 条 `needs_review=yes` 需要人工过一遍：

1. 按 `needs_review=yes` 过滤（已按 `sim_max` 升序可排序）；
2. 对每条用例，用 `POST /evaluation/chunk-candidates`（需带 `kb_id`/`file_id`/`question`）
   拿回目标文件内的候选 chunk（`chunk_id` + `snippet` + `score`）；
3. 在前端「候选 chunk 标注面板」勾选真正相关的几条，覆盖 `expected_chunk_ids`；
4. 存回（同名保存自动 version+1），旧版本仍可对比。

35 条 ≈ 全量 21%，是可控的人工工作量；**复核完这 35 条即完成 v1 定稿**。

---

## 6. 给新企业建集子的检查清单

接入 aventro / zx_bank / cendara / velvera 时按此顺序执行：

- [ ] **先把文档入库并索引完成**（没有 KB 就没有 `file_id` 与 chunk_id，无法标注）；
- [ ] 确认上传格式（`.md` 还是 `.docx`），并在 `--kb-files` 中使用**实际入库文件名**；
- [ ] 导出该 KB 的 `file_id → filename` 映射，落盘为 `*_kb_files.json`（版本化）；
- [ ] 跑 Stage 1 / 2 / 3；
- [ ] 检查 `*_excluded.csv`：应为空或仅有已知语料缺陷（如 ZX Bank 的
      `Account Close Guide.md`）；
- [ ] 检查 `*_build_summary.json` 的 `chunk_label_coverage` 与
      `cases_below_review_threshold`；
- [ ] 人工复核 `needs_review=yes` 的用例；
- [ ] 跑一次基线并存档 `*_baseline_run.json`，作为后续 A/B 的对照。

### 已知需要特殊处理的语料缺陷

| 企业 | 语料中的文件名 | 实际文件名 | 说明 |
|---|---|---|---|
| Aventro Motors | `Aventro Awards & Recognitions.md` | `Aventro  Awards & Recognitions.md` | 双空格，折叠可解 |
| Aventro Motors | `Aventro Connectivity.md` | `Aventro  Connectivity.md` | 双空格，折叠可解 |
| ZX Bank | `ASK Zia – Your 24:7 Banking Assistant.md` | `ASK Zia – Your 24_7 Banking Assistant.md` | `:` ↔ `_`，折叠可解 |
| ZX Bank | `Account Close Guide.md` | **不存在** | 唯一真缺失文档，须排除 |
| CloudWay-24 | `Fine-Dining Menus.md` | ` Fine-Dining Menus.md`（前导空格） | 折叠可解；该缺陷来自语料 chunk 元数据，已传入 KB |

---

## 7. 设计取舍备忘

- **为什么不让整文件兜底**：见 `RAG_EVALUATION.md` §2.2。一文件 179 条法规、
  K=5 时 Recall 上限 0.028，指标失去意义。故本流水线用 `--floor` 兜底
  「至少 1 条 chunk」，确保 `reference_mode` 恒为 `chunk_ids`，
  代价是最弱用例的标注需要人工确认。
- **为什么用 `expected_file_id` 而非文件名**：文件名是脏数据重灾区，且后端按
  精确字符串解析，一处不匹配即整批 422。UUID 一次绑定、永久稳定。
- **为什么负样本要「可证明无关」**：随机配文件的负样本可能恰好语义相关，
  会把"判别力不足"错记成"误报"，污染 false_positive 指标。本流水线要求
  问题对该文件所有 chunk 的最大相似度 ≤0.55 才收录（实测落在 0.214–0.352）。
- **为什么保留 `expected_filename` 冗余字段**：导入器忽略未知键，但对人排错
  极有价值；`validate_golden_set.py` 也据此交叉核对。
- **为什么不直接复用上游 `parsed-chunks`**：该目录是 JSONL、且**不含任何向量**
  （上游 readme 所称的 bge-m3/1024 只是散文描述），chunk 粒度也与 EasyRAG
  自身的分块无关。评测必须测**你自己的分块**，否则指标不对应线上行为。
