"""
增强检索引擎 — 查询分解 × 多粒度图-向量融合检索 × 迭代补充。

基于 LightRAG 的图谱KV索引 + 向量检索双层架构进一步优化，实现：
  ① 查询结构分解（实体+主题+关系模式+子问题）
  ② 四路并行检索（精准实体 + 语义向量 + 关系链 + 全文精确）
  ③ 图谱感知融合重排序（四维评分：向量相似度×图谱距离×跨路共识×时效性）
  ④ 知识块聚类（按图谱连通分量打包，替代平铺chunk列表）
  ⑤ 迭代缺口检测与补充检索（最多2轮）

用法:
    from app.rag.enhanced_retriever import EnhancedRetriever

    retriever = EnhancedRetriever()
    result = retriever.retrieve(query="电动汽车如何影响城市空气质量？")
    # result = {
    #     "knowledge_blocks": [...],
    #     "raw_docs": [...],
    #     "sources": [...],
    #     "query_decomposition": {...},
    #     "gap_rounds": 0,
    # }
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.embeddings import get_embedder

logger = get_logger(__name__)
cfg = get_settings()

# ═══════════════════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QueryDecomposition:
    """查询结构分解结果。"""

    explicit_entities: List[Dict[str, str]] = field(default_factory=list)
    # [{"name": "电动汽车", "type": "technology", "constraints": "..."}]

    themes: List[Dict[str, str]] = field(default_factory=list)
    # [{"theme": "城市空气质量改善路径", "scope": "broad"}]

    relation_patterns: List[Dict[str, str]] = field(default_factory=list)
    # [{"subject": "电动汽车", "predicate": "减少", "object": "尾气排放"}]

    sub_questions: List[str] = field(default_factory=list)
    # ["电动汽车推广如何直接减少尾气排放？", ...]

    query_type: str = "factual"  # factual | causal | comparative | summary | multi_hop
    complexity: str = "medium"   # low | medium | high

    def is_complex(self) -> bool:
        return self.complexity == "high" or len(self.sub_questions) > 1

    def to_dict(self) -> dict:
        return {
            "explicit_entities": self.explicit_entities,
            "themes": self.themes,
            "relation_patterns": self.relation_patterns,
            "sub_questions": self.sub_questions,
            "query_type": self.query_type,
            "complexity": self.complexity,
        }


@dataclass
class CandidateDoc:
    """统一检索候选文档。"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    retrieval_path: str = ""           # entity / semantic / relation / bm25
    graph_entities: List[str] = field(default_factory=list)  # 关联的图谱实体名
    graph_distance: float = float("inf")  # 到查询实体的图谱距离
    cross_path_hits: int = 0             # 被几条路径命中

    def __hash__(self):
        return hash(self.content[:200])


@dataclass
class KnowledgeBlock:
    """知识块 — 按图谱连通性聚类后的结果单元。"""

    block_id: str
    entities: List[str] = field(default_factory=list)
    relations: List[Dict[str, str]] = field(default_factory=list)
    docs: List[CandidateDoc] = field(default_factory=list)
    summary: str = ""
    block_score: float = 0.0


@dataclass
class RetrievalResult:
    """增强检索的完整结果。"""

    query_decomposition: QueryDecomposition = field(default_factory=QueryDecomposition)
    knowledge_blocks: List[KnowledgeBlock] = field(default_factory=list)
    raw_docs: List[CandidateDoc] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    gap_rounds: int = 0
    gap_details: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    retrieval_summary: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# 查询分解 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

_QUERY_DECOMPOSITION_PROMPT = """你是一个查询分析专家。将用户查询分解为结构化信息，用于多路径检索。

用户查询：{query}
历史上下文（如有）：{history}

返回严格 JSON（不要其他内容）：
{{
  "explicit_entities": [
    {{"name": "实体名（原文表述）", "type": "technology/person/organization/concept/location/event/product", "constraints": "限定条件，如'中国市场的'"}}
  ],
  "themes": [
    {{"theme": "主题描述（一句话）", "scope": "broad或specific"}}
  ],
  "relation_patterns": [
    {{"subject": "主体", "predicate": "谓语（如:导致/减少/推动/依赖/替代/属于/影响/使用/对比）", "object": "客体"}}
  ],
  "sub_questions": ["将复杂查询拆解为2-4个原子子问题，每个只问一件事"],
  "query_type": "factual/causal/comparative/summary/multi_hop",
  "complexity": "low/medium/high"
}}

规则：
- explicit_entities: 提取查询中明确提到的实体，type从7类中选
- themes: 1-3个主题，每个一句话，标记scope
- relation_patterns: 如果查询含因果/影响/对比关系，提取为(subject, predicate, object)三元组
- sub_questions: 复杂查询拆成原子问题；简单查询可只含原问题
- query_type: factual=事实类, causal=因果类, comparative=对比类, summary=总结类, multi_hop=多跳推理
- complexity: low=简单事实查询, medium=需要综合多条信息, high=多跳推理/跨文档分析

不要遗漏查询中的任何重要信息。实体名用查询中的原文表述。"""

# ═══════════════════════════════════════════════════════════════════════════════
# 缺口检测 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

_GAP_DETECTION_PROMPT = """你是一个检索质量评估专家。检查当前检索结果是否充分覆盖用户查询的所有方面。

用户原始查询：{query}

子问题清单：
{sub_questions}

已检索到的信息概要（前500字符）：
{retrieved_summary}

请逐一检查每个子问题：
- ✓ 信息充分：可以直接生成高质量回答
- △ 信息不足：有线索但不完整，需要补充
- ✗ 完全缺失：检索结果中没有任何相关信息

返回严格 JSON（不要其他内容）：
{{
  "overall_sufficient": true或false,
  "sub_question_checks": [
    {{"sub_question": "子问题原文", "status": "sufficient/insufficient/missing", "reason": "一句话说明"}}
  ],
  "gap_queries": ["需要补充检索的具体查询1", "需要补充检索的具体查询2"],
  "gap_explanation": "如果overall_sufficient为false，一句话说明缺什么"
}}

如果所有子问题都充分覆盖，overall_sufficient=true，gap_queries为空数组。"""

# ═══════════════════════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════════════════════


def _run_async_in_thread(coro):
    """在独立线程中运行 async 协程，避免与 uvicorn 事件循环冲突。

    LangGraph 节点是同步函数，但 PG/图谱查询必须走 async。
    直接调用 asyncio 的 run 方法在 uvicorn 内部会创建冲突的事件循环。
    此函数在新线程中创建专用事件循环，安全执行 async 操作。
    """
    import threading
    result = None
    error = None

    def _runner():
        nonlocal result, error
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
        except Exception as e:
            error = e
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=30)
    if error:
        raise error
    return result


class EnhancedRetriever:
    """增强检索引擎。

    整合了查询分解、四路并行检索、图谱融合重排序、知识块聚类和迭代缺口补充。
    设计为在 LangGraph 节点中调用，暴露同步接口。

    参数
    ----
    fusion_weights : (α, β, γ, δ)
        α = 向量语义相似度权重 (默认 0.35)
        β = 图谱距离权重       (默认 0.25)
        γ = 跨路共识权重       (默认 0.25)
        δ = 来源时效性权重     (默认 0.15)
    """

    def __init__(
        self,
        fusion_weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15),
        max_gap_rounds: int = 2,
        top_k_per_path: int = 6,
        final_top_k: int = 8,
    ):
        self.fusion_alpha, self.fusion_beta, self.fusion_gamma, self.fusion_delta = fusion_weights
        self.max_gap_rounds = max_gap_rounds
        self.top_k_per_path = top_k_per_path
        self.final_top_k = final_top_k
        self._llm = None
        self._embedder = None
        self._bm25 = None

    # ── 属性（延迟加载）───────────────────────────────────────────────────────

    @property
    def llm(self):
        if self._llm is None:
            from app.llm.client import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def bm25(self):
        if self._bm25 is None:
            from app.rag.bm25 import get_bm25
            self._bm25 = get_bm25()
        return self._bm25

    # ═══════════════════════════════════════════════════════════════════════════
    # 公开接口
    # ═══════════════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        kb_sources_filter: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """主入口：执行完整的增强检索流程。

        参数
        ----
        query : 用户原始查询
        history : 对话历史 [{"role":"user","content":"..."}, ...]
        kb_sources_filter : 限定检索的知识库文件 source 列表（None=全部）

        返回
        ----
        RetrievalResult 包含 knowledge_blocks / raw_docs / sources 等
        """
        t0 = time.perf_counter()

        # ── 第 0 步：构建 BM25 索引（如果尚未构建）──────────────────────────
        self._ensure_bm25_index()

        # ── 第 1 步：查询结构分解 ──────────────────────────────────────────
        decomposition = self._decompose_query(query, history)

        # ── 第 2 步：四路并行检索 ──────────────────────────────────────────
        all_candidates = self._parallel_retrieve(
            query, decomposition, kb_sources_filter
        )

        # ── 第 3 步：图谱感知融合重排序 ─────────────────────────────────────
        ranked = self._fusion_rerank(all_candidates, decomposition)

        # ── 第 3.5 步：交叉编码器精排（可选，RERANKER_TYPE != disabled）────
        ranked = self._apply_reranker(query, ranked)

        # ── 第 4 步：知识块聚类 ────────────────────────────────────────────
        blocks = self._cluster_into_blocks(ranked[: self.final_top_k], decomposition)

        # ── 第 5 步：迭代缺口检测与补充 ────────────────────────────────────
        result = RetrievalResult(
            query_decomposition=decomposition,
            knowledge_blocks=blocks,
            raw_docs=ranked[: self.final_top_k],
            sources=self._extract_sources(ranked[: self.final_top_k]),
        )

        if decomposition.is_complex():
            result = self._iterative_gap_fill(query, result, decomposition)

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[enhanced] query=%r | paths=4 | candidates=%d | blocks=%d | gaps=%d | %.0fms",
            query[:60], len(all_candidates), len(blocks),
            result.gap_rounds, result.elapsed_ms,
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # ① 查询结构分解
    # ═══════════════════════════════════════════════════════════════════════════

    def _decompose_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryDecomposition:
        """调用 LLM 将查询分解为结构化信息。失败时回退到规则解析。"""
        history_str = ""
        if history:
            recent = history[-6:]  # 最近 3 轮
            history_str = "\n".join(
                f"[{h['role']}]: {h['content'][:200]}" for h in recent
            )

        try:
            data = self.llm.chat_json_sync([{
                "role": "user",
                "content": _QUERY_DECOMPOSITION_PROMPT.format(
                    query=query, history=history_str
                ),
            }])
            logger.info("[enhanced] query decomposition: type=%s complexity=%s",
                        data.get("query_type"), data.get("complexity"))
            return QueryDecomposition(
                explicit_entities=data.get("explicit_entities") or [],
                themes=data.get("themes") or [],
                relation_patterns=data.get("relation_patterns") or [],
                sub_questions=data.get("sub_questions") or [query],
                query_type=data.get("query_type", "factual"),
                complexity=data.get("complexity", "medium"),
            )
        except Exception as exc:
            logger.warning("[enhanced] LLM query decomposition failed: %s, fallback to rule-based", exc)
            return self._rule_based_decompose(query)

    def _rule_based_decompose(self, query: str) -> QueryDecomposition:
        """规则回退：简单关键词拆分。"""
        import re

        # 提取中文实体（连续名词性片段）
        cn_entities = re.findall(r"[\u4e00-\u9fff]{2,8}(?:系统|模型|方法|技术|政策|问题|数据|影响|市场|行业|公司|基金|股票|指数)?", query)
        entities = []
        seen = set()
        for e in cn_entities:
            if e not in seen and len(e) >= 2:
                seen.add(e)
                entities.append({"name": e, "type": "concept", "constraints": ""})

        # 检测关系模式
        relation_words = {
            "导致": "导致", "造成": "导致", "引起": "导致",
            "减少": "减少", "降低": "减少", "下降": "减少",
            "推动": "推动", "促进": "推动", "驱动": "推动",
            "影响": "影响", "改变": "影响",
            "对比": "对比", "比较": "对比", "相比": "对比",
        }
        relations = []
        for word, predicate in relation_words.items():
            if word in query:
                idx = query.index(word)
                subj = query[max(0, idx - 10):idx].strip()
                obj = query[idx + len(word):idx + len(word) + 10].strip()
                if subj and obj:
                    relations.append({"subject": subj, "predicate": predicate, "object": obj})

        # 推断复杂度
        complexity = "low"
        if len(entities) >= 3 or relations:
            complexity = "medium"
        if relations and len(entities) >= 3:
            complexity = "high"

        q_type = "factual"
        if relations:
            q_type = "causal"
        if any(w in query for w in ["对比", "比较", "区别", "vs"]):
            q_type = "comparative"

        return QueryDecomposition(
            explicit_entities=entities[:5],
            themes=[{"theme": query[:80], "scope": "broad"}],
            relation_patterns=relations[:3],
            sub_questions=[query],
            query_type=q_type,
            complexity=complexity,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ② 四路并行检索
    # ═══════════════════════════════════════════════════════════════════════════

    def _parallel_retrieve(
        self,
        query: str,
        decomposition: QueryDecomposition,
        kb_sources_filter: Optional[List[str]] = None,
    ) -> List[CandidateDoc]:
        """四路真正的并行检索，合并去重。

        使用 ThreadPoolExecutor 并发执行四路径，每路独立。
        对于 I/O 密集型操作（向量检索、图谱查询、BM25），线程并发能显著降低总延迟。
        """
        all_docs: List[CandidateDoc] = []
        seen_content: Set[str] = set()

        # 四路并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_entity = executor.submit(self._retrieve_entity_path, decomposition)
            future_semantic = executor.submit(
                self._retrieve_semantic_path, query, decomposition, kb_sources_filter
            )
            future_relation = executor.submit(
                self._retrieve_relation_path, decomposition
            )
            future_bm25 = executor.submit(self._retrieve_bm25_path, query)

            # 按完成顺序处理（谁先回来谁先处理，减少等待）
            path_results = [
                ("entity", future_entity),
                ("semantic", future_semantic),
                ("relation", future_relation),
                ("bm25", future_bm25),
            ]

        for path_name, future in path_results:
            try:
                docs = future.result(timeout=30)
            except Exception as exc:
                logger.warning("[enhanced] Path %s failed: %s", path_name, exc)
                docs = []

            for d in docs:
                key = self._dedup_key(d)
                if key in seen_content:
                    # 多路命中：增加已有文档的 cross_path_hits
                    for existing in all_docs:
                        if self._dedup_key(existing) == key:
                            existing.cross_path_hits += 1
                            existing.retrieval_path += "+" + path_name
                            break
                else:
                    seen_content.add(key)
                    if d.cross_path_hits == 0:
                        d.cross_path_hits = 1
                    all_docs.append(d)

        entity_count = sum(1 for d in all_docs if "entity" in d.retrieval_path)
        semantic_count = sum(1 for d in all_docs if "semantic" in d.retrieval_path)
        relation_count = sum(1 for d in all_docs if "relation" in d.retrieval_path)
        bm25_count = sum(1 for d in all_docs if "bm25" in d.retrieval_path)

        logger.info(
            "[enhanced] 4-path parallel retrieval: entity=%d semantic=%d relation=%d bm25=%d → merged=%d",
            entity_count, semantic_count, relation_count, bm25_count, len(all_docs),
        )
        return all_docs

    @staticmethod
    def _dedup_key(doc: CandidateDoc) -> str:
        """生成去重键：内容前200字符的哈希。"""
        return hashlib.md5(doc.content[:200].encode()).hexdigest()

    # ── Path A: 精准实体路径 ──────────────────────────────────────────────────

    def _retrieve_entity_path(self, decomposition: QueryDecomposition) -> List[CandidateDoc]:
        """Path A: 显式实体 → 图谱内存缓存匹配 → 一跳邻居 → 向量精排。"""
        entity_names = [e["name"] for e in decomposition.explicit_entities]
        if not entity_names:
            return []

        try:
            from app.rag.graph_cache import graph_cache
            matched = graph_cache.match_entities(entity_names, top_n=cfg.GRAPH_QUERY_TOP_ENTITIES)
        except Exception as exc:
            logger.warning("[enhanced] Path A graph cache lookup failed: %s", exc)
            return []

        docs: List[CandidateDoc] = []
        for ent in matched:
            desc = ent.get("description", "")
            name = ent.get("name", "")
            if desc:
                docs.append(CandidateDoc(
                    content=f"[实体] {name} ({ent.get('type','')}): {desc}",
                    metadata={"source": "knowledge_graph", "entity": name, "score": 1.0},
                    score=1.0,
                    retrieval_path="entity",
                    graph_entities=[name],
                ))
            # 邻居关系
            for rel in graph_cache.get_neighbor_relations(name)[:6]:
                rel_text = (
                    f"[关系] {rel['source']} --[{rel['relation']}]--> "
                    f"{rel['target']}: {rel.get('description','')}"
                )
                docs.append(CandidateDoc(
                    content=rel_text,
                    metadata={"source": "knowledge_graph", "entity": name, "relation": rel["relation"]},
                    score=0.85,
                    retrieval_path="entity",
                    graph_entities=[rel.get("source", ""), rel.get("target", "")],
                ))

        if len(docs) > self.top_k_per_path:
            docs = self._vector_rerank_candidates(" ".join(entity_names), docs, self.top_k_per_path)
            for d in docs:
                if d.metadata.get("score", 0) > 0:
                    d.score = d.metadata["score"]

        return docs

    # ── Path B: 语义向量路径（含反向追溯）─────────────────────────────────────

    def _retrieve_semantic_path(
        self,
        query: str,
        decomposition: QueryDecomposition,
        kb_sources_filter: Optional[List[str]] = None,
    ) -> List[CandidateDoc]:
        """Path B: 语义向量检索 + 反向追溯图谱实体以扩展上下文。

        1. 对原始查询和所有主题做向量检索
        2. 命中的 chunk 反向查找它关联了哪些图谱实体
        3. 如果有关联实体，把该实体的描述也作为候选加入
        """
        from app.rag.retriever import get_retriever

        retriever = get_retriever()
        docs: List[CandidateDoc] = []

        # 构建查询向量：原始查询 + 主题描述拼接
        theme_texts = [t["theme"] for t in decomposition.themes] if decomposition.themes else []
        queries_to_search = [query] + theme_texts[:2]  # 最多3个向量查询

        seen_keys = set()
        for q in queries_to_search:
            try:
                raw_docs = retriever.retrieve(q, top_k=self.top_k_per_path)
            except Exception:
                continue

            for rd in raw_docs:
                meta = rd.get("metadata", {})
                content = rd.get("content", "")
                score = float(meta.get("score", 0.0))

                doc = CandidateDoc(
                    content=content,
                    metadata=meta,
                    score=score,
                    retrieval_path="semantic",
                )

                # 仅保留未重复的
                key = self._dedup_key(doc)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                docs.append(doc)

        # 反向追溯：从命中的 chunk 溯源图谱实体（内存缓存，无 PG 调用）
        if docs and cfg.GRAPH_ENABLED:
            try:
                import re
                from app.rag.graph_cache import graph_cache
                graph_extensions = []
                for doc in docs[:3]:
                    keywords = re.findall(r"[\u4e00-\u9fff]{2,10}|[A-Za-z]{3,20}", doc.content[:500])
                    if not keywords:
                        continue
                    matched = graph_cache.match_entities(list(set(keywords))[:5], top_n=3)
                    for ent in matched:
                        graph_extensions.append({
                            "entity": ent["name"],
                            "content": f"[实体] {ent['name']} ({ent['type']}): {ent.get('description','')}",
                            "score": 0.75,
                        })
                for ext in graph_extensions:
                    ext_doc = CandidateDoc(
                        content=ext["content"],
                        metadata={"source": "knowledge_graph_reverse", "entity": ext["entity"]},
                        score=ext.get("score", 0.7),
                        retrieval_path="semantic+graph_reverse",
                        graph_entities=[ext["entity"]],
                    )
                    key = self._dedup_key(ext_doc)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        docs.append(ext_doc)
            except Exception as exc:
                logger.debug("[enhanced] Path B reverse lookup failed: %s", exc)

        return docs

    # ── Path C: 关系链路径 ───────────────────────────────────────────────────

    def _retrieve_relation_path(self, decomposition: QueryDecomposition) -> List[CandidateDoc]:
        """Path C: 关系模式 → 内存图谱缓存搜索 → 引导式多跳。"""
        patterns = decomposition.relation_patterns
        if not patterns:
            return []

        docs: List[CandidateDoc] = []
        try:
            from app.rag.graph_cache import graph_cache
            all_subjects = set(p["subject"] for p in patterns)
            all_objects = set(p["object"] for p in patterns)
            all_predicates = [p["predicate"] for p in patterns]
            search_entities = list(all_subjects | all_objects)

            chains = graph_cache.get_relations_by_predicate(search_entities, all_predicates)

            for chain in chains:
                chain_text_parts = []
                for step in chain.get("steps", []):
                    chain_text_parts.append(
                        f"{step['source']} --[{step['relation']}]--> "
                        f"{step['target']}: {step.get('description','')}"
                    )
                docs.append(CandidateDoc(
                    content="\n".join(chain_text_parts),
                    metadata={"source": "knowledge_graph_chain", "chain_length": len(chain.get("steps", []))},
                    score=0.9,
                    retrieval_path="relation",
                    graph_entities=[s.get("source", "") for s in chain.get("steps", [])]
                    + [s.get("target", "") for s in chain.get("steps", [])],
                ))

        except Exception as exc:
            logger.warning("[enhanced] Path C relation search failed: %s", exc)

        return docs

    # ── Path D: 全文精确路径 (BM25) ──────────────────────────────────────────

    def _retrieve_bm25_path(self, query: str) -> List[CandidateDoc]:
        """Path D: BM25 稀疏检索 — 精确匹配数字、代码、专有名词。"""
        try:
            results = self.bm25.search(query, top_k=self.top_k_per_path)
        except Exception as exc:
            logger.debug("[enhanced] Path D BM25 failed: %s", exc)
            return []

        return [
            CandidateDoc(
                content=r["content"],
                metadata=r.get("metadata", {}),
                score=r["score"] * 0.5,  # BM25 分数归一化（大致映射到 [0,1] 区间）
                retrieval_path="bm25",
            )
            for r in results
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # ③ 图谱感知融合重排序
    # ═══════════════════════════════════════════════════════════════════════════

    def _fusion_rerank(
        self,
        candidates: List[CandidateDoc],
        decomposition: QueryDecomposition,
    ) -> List[CandidateDoc]:
        """四维评分函数重排序。

        final_score = α × vector_sim + β × graph_proximity + γ × cross_consensus + δ × freshness
        """
        if not candidates:
            return []

        query_entities = [e["name"] for e in decomposition.explicit_entities]

        # 计算每种分数的归一化范围
        for doc in candidates:
            # ① 向量相似度（取原始score或metadata.score的max）
            vector_sim = max(doc.score, float(doc.metadata.get("score", 0.0)))
            if vector_sim > 1.0:
                vector_sim = min(vector_sim / 10.0, 1.0)  # 归一化 Milvus IP 分数

            # ② 图谱接近度：文档关联实体到查询实体的距离
            graph_dist = self._compute_graph_proximity(doc, query_entities)

            # ③ 跨路共识：多路命中加成
            cross_consensus = self._compute_cross_consensus(doc)

            # ④ 来源时效性（从 metadata 提取日期信息）
            freshness = self._compute_freshness(doc)

            # 综合评分
            doc.score = (
                self.fusion_alpha * vector_sim
                + self.fusion_beta * graph_dist
                + self.fusion_gamma * cross_consensus
                + self.fusion_delta * freshness
            )

            # 保存各维度分数到 metadata
            doc.metadata["fusion_vector_sim"] = round(vector_sim, 4)
            doc.metadata["fusion_graph_prox"] = round(graph_dist, 4)
            doc.metadata["fusion_cross"] = round(cross_consensus, 4)
            doc.metadata["fusion_freshness"] = round(freshness, 4)
            doc.metadata["fusion_final"] = round(doc.score, 4)

        # 降序排列
        ranked = sorted(candidates, key=lambda d: d.score, reverse=True)
        logger.info(
            "[enhanced] fusion rerank: %d → top score=%.4f",
            len(ranked), ranked[0].score if ranked else 0,
        )
        return ranked

    def _compute_graph_proximity(
        self, doc: CandidateDoc, query_entities: List[str]
    ) -> float:
        """计算文档关联实体到查询实体的图谱接近度。

        返回值 [0, 1]：1.0 表示直接命中查询实体，0.0 表示无关。
        """
        if not query_entities or not doc.graph_entities:
            # 无图谱信息 → 中性分数
            return 0.5

        # 直接命中：文档关联的实体就是查询实体
        for qe in query_entities:
            for ge in doc.graph_entities:
                if qe in ge or ge in qe:
                    return 1.0

        # 间接关联：有图谱实体但未直接命中 → 中等分数
        return 0.6

    @staticmethod
    def _compute_cross_consensus(doc: CandidateDoc) -> float:
        """跨路共识：多路命中 = 高置信度。

        1 路命中 = 0.3, 2 路 = 0.65, 3 路 = 0.85, 4 路 = 1.0
        """
        hits = max(1, doc.cross_path_hits + 1)  # +1 因为至少被自己的路径命中
        consensus_map = {1: 0.3, 2: 0.65, 3: 0.85, 4: 1.0}
        return consensus_map.get(hits, 1.0)

    def _apply_reranker(
        self, query: str, docs: List[CandidateDoc]
    ) -> List[CandidateDoc]:
        """交叉编码器精排：在融合评分后用 reranker 进一步过滤。

        只对 top_n 个候选做精排（控制延迟），精排后按 reranker 分数重排。
        RERANKER_TYPE=disabled 时直接返回原列表。
        """
        if cfg.RERANKER_TYPE == "disabled":
            return docs
        if not docs:
            return docs

        re_top_n = min(len(docs), cfg.RERANKER_TOP_K * 2)
        candidates = docs[:re_top_n]

        try:
            from app.rag.reranker import get_reranker
            reranker = get_reranker()
            texts = [d.content[: cfg.RERANKER_MAX_LENGTH] for d in candidates]
            ranked = reranker.rerank(query, texts, top_k=cfg.RERANKER_TOP_K)

            # 按精排分数重建顺序
            reranked = []
            remaining = set(range(len(candidates)))
            for idx, score in ranked:
                doc = candidates[idx]
                doc.score = score  # 用精排分数替代融合分数
                doc.metadata["rerank_score"] = round(score, 4)
                reranked.append(doc)
                remaining.discard(idx)

            # 未被精排覆盖的候选保持原顺序
            for i in sorted(remaining):
                reranked.append(candidates[i])

            # 追加被截断的原始候选
            reranked.extend(docs[re_top_n:])

            logger.info(
                "[enhanced] reranker: %d candidates → %d reranked, top score=%.4f",
                re_top_n, len(ranked), ranked[0][1] if ranked else 0,
            )
            return reranked

        except Exception as exc:
            logger.warning("[enhanced] reranker failed: %s, keeping fusion order", exc)
            return docs

    @staticmethod
    def _compute_freshness(doc: CandidateDoc) -> float:
        """从 metadata 推断来源时效性。

        检测年份信息：2024+ = 1.0, 2023 = 0.9, ..., 2020 以下 = 0.5。
        无日期信息 = 0.7（中性）。
        """
        import re
        all_text = doc.content + " " + str(doc.metadata)
        years = re.findall(r"(20\d{2})", all_text)
        if not years:
            return 0.7
        latest = max(int(y) for y in years)
        if latest >= 2024:
            return 1.0
        if latest >= 2022:
            return 0.85
        if latest >= 2020:
            return 0.7
        return 0.5

    # ═══════════════════════════════════════════════════════════════════════════
    # ④ 知识块聚类
    # ═══════════════════════════════════════════════════════════════════════════

    def _cluster_into_blocks(
        self,
        docs: List[CandidateDoc],
        decomposition: QueryDecomposition,
    ) -> List[KnowledgeBlock]:
        """按图谱连通性将文档聚类为知识块。

        同一连通分量内的实体/chunk 打包为一个 KnowledgeBlock，
        替代平铺的 [1][2][3] 列表。
        """
        if not docs:
            return []

        # 构建连通分量
        # 策略：共享图谱实体的文档属于同一分量
        entity_to_docs: Dict[str, List[int]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for ent in doc.graph_entities:
                entity_to_docs[ent].append(i)

        # Union-Find
        parent = list(range(len(docs)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for indices in entity_to_docs.values():
            for i in range(1, len(indices)):
                union(indices[0], indices[i])

        # 分组
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(docs)):
            groups[find(i)].append(i)

        # 构建知识块
        blocks: List[KnowledgeBlock] = []
        for root, indices in groups.items():
            block_docs = [docs[i] for i in indices]
            block_entities = list(set(
                ent for d in block_docs for ent in d.graph_entities
            ))
            avg_score = sum(d.score for d in block_docs) / len(block_docs)

            block = KnowledgeBlock(
                block_id=f"block_{root}",
                entities=block_entities[:10],
                docs=block_docs,
                block_score=round(avg_score, 4),
            )
            # 生成摘要
            block.summary = self._summarize_block(block)
            blocks.append(block)

        # 按分数排序
        blocks.sort(key=lambda b: b.block_score, reverse=True)

        # 补充：没有图谱关联的文档各自成块
        orphan_indices = [
            i for i in range(len(docs))
            if not docs[i].graph_entities
        ]
        existing_indices = {i for indices in groups.values() for i in indices}
        for i in orphan_indices:
            if i not in existing_indices:
                blocks.append(KnowledgeBlock(
                    block_id=f"block_orphan_{i}",
                    docs=[docs[i]],
                    block_score=docs[i].score,
                    summary=f"[独立片段] {docs[i].content[:100]}...",
                ))

        logger.info("[enhanced] clustered %d docs → %d blocks", len(docs), len(blocks))
        return blocks

    @staticmethod
    def _summarize_block(block: KnowledgeBlock) -> str:
        """生成知识块的一句话摘要。"""
        if block.entities:
            entity_str = "、".join(block.entities[:5])
            return f"知识块涉及实体: {entity_str}（{len(block.docs)} 条信息）"
        if block.docs:
            return block.docs[0].content[:100] + "..."
        return "空知识块"

    # ═══════════════════════════════════════════════════════════════════════════
    # ⑤ 迭代缺口检测与补充检索
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterative_gap_fill(
        self,
        query: str,
        result: RetrievalResult,
        decomposition: QueryDecomposition,
    ) -> RetrievalResult:
        """检测检索缺口并执行补充检索（最多 max_gap_rounds 轮）。"""
        for round_num in range(1, self.max_gap_rounds + 1):
            gap_result = self._detect_gaps(query, result, decomposition)

            if gap_result.get("overall_sufficient", True):
                logger.info("[enhanced] gap detection round %d: sufficient, stopping", round_num)
                break

            gap_queries = gap_result.get("gap_queries", [])
            if not gap_queries:
                break

            logger.info(
                "[enhanced] gap round %d: %d gap queries → supplementing",
                round_num, len(gap_queries),
            )
            result.gap_rounds = round_num
            result.gap_details.append(gap_result)

            # 对每个缺口查询做补充检索（仅 Path B 语义路径 + Path D BM25，不重复建图查询）
            for gq in gap_queries[:3]:
                try:
                    from app.rag.retriever import get_retriever
                    retriever = get_retriever()
                    raw = retriever.retrieve(gq, top_k=3)
                except Exception:
                    continue

                seen_keys = {
                    self._dedup_key(d) for d in result.raw_docs
                }
                for rd in raw:
                    doc = CandidateDoc(
                        content=rd.get("content", ""),
                        metadata=rd.get("metadata", {}),
                        score=float(rd.get("metadata", {}).get("score", 0.5)),
                        retrieval_path=f"gap_round_{round_num}",
                    )
                    key = self._dedup_key(doc)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        result.raw_docs.append(doc)
                        result.sources.extend(self._extract_sources([doc]))

        return result

    def _detect_gaps(
        self,
        query: str,
        result: RetrievalResult,
        decomposition: QueryDecomposition,
    ) -> Dict[str, Any]:
        """用 LLM 评估检索结果是否充分覆盖所有子问题。"""
        # 构建已检索内容的摘要
        summary_parts = []
        for i, block in enumerate(result.knowledge_blocks[:5]):
            summary_parts.append(f"[知识块{i+1}] {block.summary}")
        for i, doc in enumerate(result.raw_docs[:5]):
            if doc.content not in "".join(summary_parts):
                summary_parts.append(f"[片段{i+1}] {doc.content[:150]}")

        retrieved_summary = "\n".join(summary_parts)[:500]
        sub_questions_str = "\n".join(
            f"  {i+1}. {sq}" for i, sq in enumerate(decomposition.sub_questions)
        )

        try:
            data = self.llm.chat_json_sync([{
                "role": "user",
                "content": _GAP_DETECTION_PROMPT.format(
                    query=query,
                    sub_questions=sub_questions_str,
                    retrieved_summary=retrieved_summary,
                ),
            }])
            return data
        except Exception as exc:
            logger.warning("[enhanced] gap detection LLM call failed: %s", exc)
            return {"overall_sufficient": True, "gap_queries": []}

    # ═══════════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════════════

    def _ensure_bm25_index(self):
        """确保 BM25 索引已从向量库同步。首次调用时自动构建。"""
        if self.bm25.doc_count > 0:
            return

        self.sync_bm25_from_vector_store()

    def sync_bm25_from_vector_store(self):
        """从向量库（Milvus/Memory/Chroma）拉取所有 chunk 并构建 BM25 索引。

        应在每次文档入库后调用，或在检索触发时懒加载。
        对于 Milvus，会分页扫描所有实体。
        """
        try:
            from app.rag.retriever import get_retriever
            retriever = get_retriever()
            file_infos = retriever.list_documents()
            if not file_infos:
                logger.info("[enhanced] BM25: no documents in vector store, skipping")
                return

            all_chunks: List[dict] = []
            # 尝试从向量库获取所有 chunk（分页扫描）
            try:
                # Milvus 分页扫描
                if hasattr(retriever, "_col") and retriever._col is not None:
                    total = retriever._col.num_entities
                    page_size = 1000
                    offset = 0
                    while offset < total:
                        res = retriever._col.query(
                            expr="id != ''",
                            output_fields=["content", "source"],
                            offset=offset,
                            limit=page_size,
                        )
                        for row in res:
                            all_chunks.append({
                                "id": row.get("id", ""),
                                "content": row.get("content", ""),
                                "metadata": {"source": row.get("source", "unknown")},
                            })
                        offset += page_size
                        if len(res) < page_size:
                            break
                elif hasattr(retriever, "_texts"):
                    # Memory retriever
                    for i, text in enumerate(retriever._texts):
                        meta = retriever._metas[i] if retriever._metas else {}
                        all_chunks.append({
                            "id": str(i),
                            "content": text,
                            "metadata": meta,
                        })
            except Exception as scan_err:
                logger.warning("[enhanced] BM25 scan fallback: %s", scan_err)

            if all_chunks:
                self.bm25.index(all_chunks)
                logger.info(
                    "[enhanced] BM25: indexed %d chunks from vector store", len(all_chunks)
                )
            else:
                logger.info("[enhanced] BM25: %d files registered, no chunks scanned", len(file_infos))

        except Exception as exc:
            logger.warning("[enhanced] BM25 sync failed: %s", exc)

    def sync_bm25_from_chunks(self, chunks: List[Tuple[str, dict]]):
        """从 chunk 列表构建 BM25 索引。

        参数 chunks: [(text, metadata), ...]，metadata 需含 source。
        """
        documents = [
            {
                "id": meta.get("source", f"chunk_{i}"),
                "content": text,
                "metadata": meta,
            }
            for i, (text, meta) in enumerate(chunks)
        ]
        self.bm25.index(documents)
        logger.info("[enhanced] BM25 index built from %d chunks", len(documents))

    @staticmethod
    def _vector_rerank_candidates(
        query_text: str,
        docs: List[CandidateDoc],
        top_k: int,
    ) -> List[CandidateDoc]:
        """用向量相似度对候选列表精排。"""
        if not docs:
            return docs
        embedder = get_embedder()
        try:
            q_vec = embedder.embed_query(query_text)
        except Exception:
            return docs[:top_k]

        import numpy as np
        q = np.array(q_vec, dtype=float)
        q_norm = np.linalg.norm(q) + 1e-9
        q = q / q_norm

        scored: List[Tuple[float, CandidateDoc]] = []
        for doc in docs:
            try:
                d_vec = np.array(
                    embedder.embed_query(doc.content[:500]), dtype=float
                )
                d_norm = np.linalg.norm(d_vec) + 1e-9
                sim = float(np.dot(q, d_vec / d_norm))
            except Exception:
                sim = doc.score
            scored.append((sim, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [doc for _, doc in scored[:top_k]]
        for sim, doc in scored[:top_k]:
            doc.metadata["rerank_score"] = round(sim, 4)

        return top

    @staticmethod
    def _extract_sources(docs: List[CandidateDoc]) -> List[Dict[str, Any]]:
        """从候选文档提取去重的来源列表。"""
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for doc in docs:
            src = (doc.metadata.get("source") or "").strip()
            if not src or src in seen:
                continue
            seen.add(src)
            sources.append({
                "title": src,
                "url": doc.metadata.get("url", ""),
                "type": doc.metadata.get("type", "kb"),
                "score": round(doc.score, 4),
                "retrieval_path": doc.retrieval_path,
            })
        return sources


# ═══════════════════════════════════════════════════════════════════════════════
# 上下文组装工具
# ═══════════════════════════════════════════════════════════════════════════════


def format_blocks_for_prompt(blocks: List[KnowledgeBlock]) -> str:
    """将知识块列表格式化为 LLM prompt 可用的结构化上下文。

    与传统的平铺 [1][2][3] 列表不同，此函数按知识块分组组织，
    每个块包含实体关系链 + 支撑原文。
    """
    parts = []
    for i, block in enumerate(blocks):
        block_header = f"## 知识块 {i + 1}: {block.summary}"
        lines = [block_header]

        # 核心实体
        if block.entities:
            lines.append(f"### 核心实体: {', '.join(block.entities[:8])}")

        # 关系链
        if block.relations:
            lines.append("### 关系链:")
            for r in block.relations:
                lines.append(
                    f"  {r.get('source','')} --[{r.get('relation','')}]--> "
                    f"{r.get('target','')}"
                )

        # 支撑原文
        if block.docs:
            lines.append("### 支撑原文:")
            for j, doc in enumerate(block.docs[:5]):
                src = doc.metadata.get("source", "unknown")
                lines.append(f"[{j + 1}] (来源: {src}, 评分: {doc.score:.3f})")
                lines.append(f"    {doc.content[:500]}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def format_flat_for_prompt(docs: List[CandidateDoc], max_chars: int = 4000) -> str:
    """传统平铺格式（回退用）：编号列表。

    参数
    ----
    max_chars : 总字符上限，超出截断
    """
    parts = []
    total = 0
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        entry = f"[{i + 1}] (来源: {src}) {doc.content[:400]}"
        parts.append(entry)
        total += len(entry)
        if total > max_chars:
            parts.append(f"... (共 {len(docs)} 条，已截断到 {i + 1} 条)")
            break

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 单例
# ═══════════════════════════════════════════════════════════════════════════════

_enhanced: Optional[EnhancedRetriever] = None


def get_enhanced_retriever() -> EnhancedRetriever:
    global _enhanced
    if _enhanced is None:
        _enhanced = EnhancedRetriever()
    return _enhanced
