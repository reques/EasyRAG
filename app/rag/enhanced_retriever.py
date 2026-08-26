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
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.embeddings import get_embedder

logger = get_logger(__name__)
cfg = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# 查询分解缓存 (LRU + TTL)
# ═══════════════════════════════════════════════════════════════════════════════

class _DecompositionCache:
    """LRU 缓存查询分解结果，减少重复 LLM 调用。"""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 300.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Tuple[float, QueryDecomposition]] = OrderedDict()

    def _make_key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[QueryDecomposition]:
        key = self._make_key(query)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, val = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            # Move to end (LRU)
            self._store.move_to_end(key)
            return val

    def put(self, query: str, decomposition: QueryDecomposition):
        key = self._make_key(query)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                self._store[key] = (time.monotonic(), decomposition)
                while len(self._store) > self._max_size:
                    self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()


_decomp_cache = _DecompositionCache(
    max_size=128,
    ttl_seconds=float(cfg.ENHANCED_DECOMPOSITION_CACHE_TTL),
)

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

    sub_question_keywords: List[List[str]] = field(default_factory=list)
    # 每个子问题对应的检索关键词（规范表述/同义替换），帮助从口语映射到规范，如
    # [["尾气排放", "空气污染", "减排"], ...]

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
            "sub_question_keywords": self.sub_question_keywords,
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
    sub_questions: List[str] = field(default_factory=list)  # 命中的来源子问题（空=原始 query）

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
    sub_questions: List[str] = field(default_factory=list)  # 该知识块覆盖的子问题


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
    {{"name": "检索概念（规范化表述，非口语原文）", "type": "person/organization/concept/location/event/product/other", "constraints": "限定条件"}}
  ],
  "themes": [
    {{"theme": "主题描述（一句话）", "scope": "broad或specific"}}
  ],
  "relation_patterns": [
    {{"subject": "主体", "predicate": "谓语（如:导致/减少/推动/依赖/替代/属于/影响/使用/对比）", "object": "客体"}}
  ],
  "sub_questions": ["将复杂查询拆解为2-4个原子子问题，每个只问一件事"],
  "sub_question_keywords": [["子问题1的检索关键词", "同义/规范表述"], ["子问题2的检索关键词", "..."]],
  "query_type": "factual/causal/comparative/summary/multi_hop",
  "complexity": "low/medium/high"
}}

规则：
- explicit_entities: 提取用于检索的核心概念（规范化表述），type从上述类中选
- themes: 1-3个主题，每个一句话，标记scope
- relation_patterns: 如果查询含因果/影响/对比关系，提取为(subject, predicate, object)三元组
- sub_questions: 复杂查询拆成原子问题，每个只问一个独立争议点/事实；查询中并列提出的多个独立问题或事实（用问号、分号、换行等分隔）必须逐一拆成独立子问题，不得合并成一个；简单查询可只含原问题
- sub_question_keywords: 为每个子问题生成2-3个检索关键词，覆盖口语词的同义/近义/规范表述（近义词、不同用词习惯都要覆盖，如「检查/检验/检测」这类近义词），帮助检索器跨越用词差异；与sub_questions一一对应，每个子问题一个关键词列表
- query_type: factual=事实类, causal=因果类, comparative=对比类, summary=总结类, multi_hop=多跳推理
- complexity: low=简单事实查询, medium=需要综合多条信息, high=多跳推理/跨文档分析

不要遗漏查询中的任何重要信息。实体名用「规范化概念」表述，遵循：
- 口语/俚语归一化为规范概念：俗称、指代、简称替换为领域内的规范全称
- 金额、数量、时间等量化词通常不作为实体，除非它们是问题的核心限定
- 用领域规范术语表达核心概念，而非查询的字面词；概念要具体到可检索粒度（含关键限定词），而非笼统的上位词
- 只抽取核心争议焦点与关键条件（2-4 个），不要派生/扩展概念；简单事实查询（complexity=low）实体越少越聚焦越好。"""

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

gap_queries 必须逐一对应 status=insufficient/missing 的子问题：用该子问题的核心概念构造精准检索查询（而非重复原始查询），每个未覆盖子问题至少对应一个 gap_query。

如果所有子问题都充分覆盖，overall_sufficient=true，gap_queries为空数组。"""


# ═══════════════════════════════════════════════════════════════════════════════
# Answerability 评估 Prompt（判断候选能否「直接回答」，而非「主题相关」）
# ═══════════════════════════════════════════════════════════════════════════════

# 结构性低信息量节点类型：只用于图遍历导航，不作检索证据（避免「第六章」这类
# 低信息量节点跨文档污染最终结果）
_STRUCTURAL_NODE_TYPES = {"chapter", "book", "part", "law", "section"}

_ANSWERABILITY_PROMPT = """你是一个检索证据验证器。给定用户查询拆解出的子问题，判断每个候选片段能否「直接回答」子问题，且其「适用前提」是否被查询事实满足。

判定标准（缺一不可，任一不满足 answerability 即为 0）：
1. 适用前提匹配：候选片段的适用主体、对象、条件、类型必须与子问题的事实一致。仅关键词重叠而适用前提不匹配（主体不同、对象不同、类型不同、前提条件未满足）的候选，视为不适用。
2. 排除性条件：若候选文本含「除…外」「不适用于」「不包含」「除外」等排除表述，且子问题的事实恰好落在被排除的范围内，则直接判定为不适用。
3. 限定条件一致性：子问题中的限定性修饰（适用对象类别、物类、范围、主体类别等限定词）必须与候选的适用对象一致。候选针对的对象类别与子问题的限定条件互斥时（子问题限定为某一类对象，候选针对另一类），即使关键词高度重叠，answerability 也为 0。
4. 直接回答：候选片段必须能直接回答子问题，而非仅主题相关。

否定信号（出现任一即判 0，即使关键词高度重叠）：
- 候选回答的是与子问题无关的另一件事（如子问题问「责任/处罚」，候选答「奖励/权限/程序性事项」）
- 候选的适用主体或对象与子问题的主体或对象明显不同（主体是 A，候选却针对 B）
- 候选适用的范畴与子问题事实的范畴不匹配（同一术语在不同范畴下含义不同，如「损失」在「人身」与「财产」两个范畴下是不同的规则）
- 候选的适用对象类别与子问题的限定条件互斥（子问题限定对象为某一类，候选针对另一类对象，限定词明确相反时即使关键词高度重叠也判 0）

子问题（编号从 1 开始）：
{sub_questions}

候选片段：
{candidates}

对每个候选，输出：
- "answerability"：0-1 分数，综合「适用前提匹配 × 直接回答程度」。适用前提不匹配则为 0（即使关键词高度重叠）；完全匹配且直接回答则为 1.0
- "sub_questions"：该候选能直接回答的子问题编号列表（无法回答任何子问题则为空数组）

严格输出 JSON（不要其他内容，scores 必须覆盖所有候选编号）：
{{"scores": {{"1": {{"answerability": 0.9, "sub_questions": [1, 3]}}, "2": {{"answerability": 0.0, "sub_questions": []}}}}}}"""

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
        top_k_per_path: int = 8,
        final_top_k: int = 12,
    ):
        self.fusion_alpha, self.fusion_beta, self.fusion_gamma, self.fusion_delta = fusion_weights
        self.max_gap_rounds = max_gap_rounds
        self.top_k_per_path = top_k_per_path
        self.final_top_k = final_top_k
        self._llm = None
        self._embedder = None
        self._bm25 = None
        self._last_decomp_was_fallback = False  # 查询分解是否走了规则回退（回退结果不缓存）
        # 持久线程池：避免每次检索创建/销毁线程的开销
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=6, thread_name_prefix="enhanced_retr"
        )
        # BM25 索引就绪标记（后台预构建）
        self._bm25_ready = threading.Event()
        # 启动后台 BM25 预构建
        self._executor.submit(self._eager_build_bm25)

    def _eager_build_bm25(self):
        """后台预构建 BM25 索引（超时保护，永不阻塞线程池）。"""
        try:
            # 用 15 秒超时包装，防止 Milvus 连接挂起永久占住 worker
            build_future = self._executor.submit(self._ensure_bm25_index)
            build_future.result(timeout=15.0)
        except (concurrent.futures.TimeoutError, Exception) as exc:
            logger.warning("[enhanced] eager BM25 build failed (timeout or error): %s", exc)
        finally:
            self._bm25_ready.set()

    # ── 属性（延迟加载）───────────────────────────────────────────────────────

    @property
    def llm(self):
        if self._llm is not None:
            return self._llm
        from app.llm.client import get_llm_client

        # The retriever may be reused across requests; do not retain another
        # user's per-chat model choice on this singleton-like service.
        return get_llm_client()

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
        knowledge_base_ids: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        """主入口：执行完整的增强检索流程（优化版：投机检索 + 缓存分解）。

        参数
        ----
        query : 用户原始查询
        history : 对话历史 [{"role":"user","content":"..."}, ...]
        knowledge_base_ids : 已授权的知识库 UUID；为空时拒绝检索

        返回
        ----
        RetrievalResult 包含 knowledge_blocks / raw_docs / sources 等
        """
        from app.rag.retriever import normalize_knowledge_base_ids

        allowed_ids = normalize_knowledge_base_ids(knowledge_base_ids)
        if not allowed_ids:
            logger.info("[enhanced] empty knowledge-base scope; retrieval denied")
            return RetrievalResult()

        t0 = time.perf_counter()

        # ── 第 0 步：BM25（可选，不阻塞主链路）───────────────────────────
        bm25_available = self._bm25_ready.wait(timeout=0.5) or self.bm25.doc_count > 0

        # ── 第 1 步：查询结构分解（带缓存）──────────────────────────────
        decomposition = self._cached_decompose_query(query, history)

        # ── 第 2 步：四路并行检索 ──────────────────────────────────────
        all_candidates = self._parallel_retrieve(
            query, decomposition, allowed_ids
        )

        # ── 第 3 步：图谱感知融合重排序 ─────────────────────────────────────
        ranked = self._fusion_rerank(all_candidates, decomposition)

        # ── 第 3.5 步：交叉编码器精排（可选，RERANKER_TYPE != disabled）────
        ranked = self._apply_reranker(query, ranked)

        # ── 第 3.6 步：answerability 评估（「能回答」压过「主题像」）────
        # 覆盖范围 = final_top_k + 子问题配额候选：确保所有可能进入最终 Top-K 的
        # 候选（含弱子问题的保底候选）都经过验证，避免未验证的高分伪相关残留
        n_sub = len(decomposition.sub_questions or [])
        min_per_sub = max(1, self.final_top_k // max(1, n_sub)) if n_sub else 1
        ans_top_k = min(len(ranked), self.final_top_k + max(1, n_sub) * min_per_sub)
        ranked = self._apply_answerability(query, decomposition, ranked, top_k=ans_top_k)

        # ── 第 3.7 步：逐子问题覆盖检查（缺口子问题针对性补充）────
        ranked, covered_all = self._coverage_gate(query, decomposition, ranked, allowed_ids)

        # ── 第 3.8 步：每子问题硬配额（弱子问题证据不被强子问题挤掉）────
        # 每个子问题至少 min_per_sub 条进最终 Top-K，其余名额按融合分补满。
        # 替代全局 [:final_top_k] 截断——全局截断会打破保底
        # （min_per_sub × 子问题数 > final_top_k 时，弱子问题证据被截在窗外）
        ranked = self._apply_per_sub_question_quota(
            ranked, decomposition,
            min_per_sub=min_per_sub, final_top_k=self.final_top_k,
        )

        # ── 第 4 步：知识块聚类 ────────────────────────────────────────────
        blocks = self._cluster_into_blocks(ranked, decomposition, allowed_ids)

        # ── 第 5 步：迭代缺口检测与补充 ────────────────────────────────────
        result = RetrievalResult(
            query_decomposition=decomposition,
            knowledge_blocks=blocks,
            raw_docs=ranked[: self.final_top_k],
            sources=self._extract_sources(ranked[: self.final_top_k]),
        )

        # 覆盖门已确定所有子问题有 answerable 证据时，跳过 LLM 缺口补充（省 1-2 次 LLM 调用）
        if decomposition.is_complex() and not covered_all:
            result = self._iterative_gap_fill(
                query, result, decomposition, allowed_ids
            )

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[enhanced] query=%r | paths=4 | candidates=%d | blocks=%d | gaps=%d | %.0fms",
            query[:60], len(all_candidates), len(blocks),
            result.gap_rounds, result.elapsed_ms,
        )
        return result

    def _cached_decompose_query(
        self, query: str, history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryDecomposition:
        """带缓存的查询分解。命中缓存跳过 LLM 调用。"""
        # 有历史上下文时不缓存（上下文会影响分解结果）
        if history and len(history) > 0:
            return self._decompose_query(query, history)

        cached = _decomp_cache.get(query)
        if cached is not None:
            logger.info("[enhanced] decomposition cache hit for query=%r", query[:60])
            return cached

        result = self._decompose_query(query, history)
        # 规则回退结果不缓存：LLM 恢复后不应继续吃规则分解（规则只拆 1-2 个子问题，
        # 与 LLM 分解差异巨大，缓存会让「同一问题」在 LLM 恢复后仍长时间结果不一致）
        if not getattr(self, "_last_decomp_was_fallback", False):
            _decomp_cache.put(query, result)
        return result

    @staticmethod
    def _merge_path_docs(
        all_docs: List[CandidateDoc],
        seen: Set[str],
        path_name: str,
        new_docs: List[CandidateDoc],
    ):
        """合并一条路径的结果，去重并更新跨路命中计数。"""
        for d in new_docs:
            key = EnhancedRetriever._dedup_key(d)
            if key in seen:
                for existing in all_docs:
                    if EnhancedRetriever._dedup_key(existing) == key:
                        existing.cross_path_hits += 1
                        existing.retrieval_path += "+" + path_name
                        break
            else:
                seen.add(key)
                if d.cross_path_hits == 0:
                    d.cross_path_hits = 1
                all_docs.append(d)

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
            self._last_decomp_was_fallback = False
            sub_questions = data.get("sub_questions") or [query]
            raw_keywords = data.get("sub_question_keywords") or []
            # 归一化：确保关键词列表与子问题一一对应，每个子问题最多 4 个关键词
            normalized_kw: List[List[str]] = []
            for i in range(len(sub_questions)):
                if i < len(raw_keywords) and isinstance(raw_keywords[i], list):
                    normalized_kw.append([str(k) for k in raw_keywords[i] if str(k).strip()][:3])
                else:
                    normalized_kw.append([])
            return QueryDecomposition(
                explicit_entities=data.get("explicit_entities") or [],
                themes=data.get("themes") or [],
                relation_patterns=data.get("relation_patterns") or [],
                sub_questions=sub_questions,
                sub_question_keywords=normalized_kw,
                query_type=data.get("query_type", "factual"),
                complexity=data.get("complexity", "medium"),
            )
        except Exception as exc:
            logger.warning("[enhanced] LLM query decomposition failed: %s, fallback to rule-based", exc)
            self._last_decomp_was_fallback = True
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
        knowledge_base_ids: Sequence[str],
    ) -> List[CandidateDoc]:
        """逐子问题独立检索 + 来源标注，合并去重。

        每个子问题独立做语义 + BM25 检索（标注来源子问题），而非把所有子问题
        混在一个结果集里——避免某个子问题的证据被其他子问题淹没，且让每个
        候选/知识块能回溯到它回答的是哪个子问题。
        """
        all_docs: List[CandidateDoc] = []
        seen_content: Set[str] = set()
        allowed_ids = set(knowledge_base_ids)
        sub_questions = decomposition.sub_questions or []

        # 构建检索任务：(来源子问题, 路径名, future)；空子问题 = 原始 query 级
        tasks: List[Tuple[str, str, "concurrent.futures.Future"]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            tasks.append(("", "entity", executor.submit(
                self._retrieve_entity_path, decomposition, knowledge_base_ids
            )))
            tasks.append(("", "relation", executor.submit(
                self._retrieve_relation_path, decomposition, knowledge_base_ids
            )))
            # 原始 query 的语义 + BM25（兜底，覆盖未拆分子问题的情况）
            tasks.append(("", "semantic", executor.submit(
                self._retrieve_semantic_path, query, decomposition, knowledge_base_ids
            )))
            tasks.append(("", "bm25", executor.submit(
                self._retrieve_bm25_path, query, knowledge_base_ids
            )))
            # 每个子问题独立检索（用检索关键词扩展，帮助口语表述映射到规范表述）
            keywords = decomposition.sub_question_keywords or []
            for i, sq in enumerate(sub_questions):
                kw = keywords[i] if i < len(keywords) else []
                expanded = (sq + " " + " ".join(kw)).strip() if kw else sq
                tasks.append((sq, f"semantic_sub{i}", executor.submit(
                    self._retrieve_semantic_path, expanded, decomposition, knowledge_base_ids
                )))
                tasks.append((sq, f"bm25_sub{i}", executor.submit(
                    self._retrieve_bm25_path, expanded, knowledge_base_ids
                )))

        # with 块退出后所有任务已完成，统一收集结果
        resolved: List[Tuple[str, str, List[CandidateDoc]]] = []
        for sub_q, path_name, future in tasks:
            try:
                resolved.append((sub_q, path_name, future.result(timeout=30)))
            except Exception as exc:
                logger.warning("[enhanced] Path %s failed: %s", path_name, exc)
                resolved.append((sub_q, path_name, []))

        for sub_q, path_name, docs in resolved:
            for d in docs:
                if d.metadata.get("knowledge_base_id") not in allowed_ids:
                    continue
                key = self._dedup_key(d)
                if key in seen_content:
                    # 多路 / 多子问题命中：追加来源
                    for existing in all_docs:
                        if self._dedup_key(existing) == key:
                            existing.cross_path_hits += 1
                            existing.retrieval_path += "+" + path_name
                            if sub_q and sub_q not in existing.sub_questions:
                                existing.sub_questions.append(sub_q)
                            break
                else:
                    seen_content.add(key)
                    if d.cross_path_hits == 0:
                        d.cross_path_hits = 1
                    if sub_q:
                        d.sub_questions.append(sub_q)
                    all_docs.append(d)

        logger.info(
            "[enhanced] per-sub-question retrieval: %d sub-questions + query → merged=%d docs",
            len(sub_questions), len(all_docs),
        )
        return all_docs

    @staticmethod
    def _dedup_key(doc: CandidateDoc) -> str:
        """生成去重键：内容前200字符的哈希。"""
        return hashlib.md5(doc.content[:200].encode()).hexdigest()

    # ── Path A: 精准实体路径 ──────────────────────────────────────────────────

    def _retrieve_entity_path(
        self,
        decomposition: QueryDecomposition,
        knowledge_base_ids: Sequence[str],
    ) -> List[CandidateDoc]:
        """Path A: 显式实体 → 图谱内存缓存匹配 → 一跳邻居 → 向量精排。"""
        entity_names = [e["name"] for e in decomposition.explicit_entities]
        if not entity_names:
            return []

        try:
            from app.rag.graph_cache import graph_cache
            matched = graph_cache.match_entities(
                entity_names,
                top_n=cfg.GRAPH_QUERY_TOP_ENTITIES,
                knowledge_base_ids=knowledge_base_ids,
            )
        except Exception as exc:
            logger.warning("[enhanced] Path A graph cache lookup failed: %s", exc)
            return []

        docs: List[CandidateDoc] = []
        for ent in matched:
            desc = ent.get("description", "")
            name = ent.get("name", "")
            ent_sf = ent.get("source_file") or None  # 命名空间：该实体所属文件
            ent_sources = {name: ent_sf}
            if desc:
                docs.append(CandidateDoc(
                    content=f"[实体] {name} ({ent.get('type','')}): {desc}",
                    metadata={
                        "source": "knowledge_graph",
                        "entity": name,
                        "entity_source_file": ent_sf,
                        "entity_source_files": dict(ent_sources),
                        "score": 1.0,
                        "knowledge_base_id": ent.get("kb_id", ""),
                    },
                    score=1.0,
                    retrieval_path="entity",
                    graph_entities=[name],
                ))
            # 邻居关系（按命名空间精确查询：同名实体只展开自己文件内的边，不跨文件混）
            for rel in graph_cache.get_neighbor_relations(
                name,
                knowledge_base_ids=knowledge_base_ids,
                source_file=ent_sf,
            )[:6]:
                rel_sf = rel.get("source_file") or ent_sf
                ent_sources[rel.get("source", "")] = rel_sf
                ent_sources[rel.get("target", "")] = rel_sf
                rel_text = (
                    f"[关系] {rel['source']} --[{rel['relation']}]--> "
                    f"{rel['target']}: {rel.get('description','')}"
                )
                docs.append(CandidateDoc(
                    content=rel_text,
                    metadata={
                        "source": "knowledge_graph",
                        "entity": name,
                        "entity_source_file": ent_sf,
                        "entity_source_files": dict(ent_sources),
                        "relation": rel["relation"],
                        "knowledge_base_id": rel.get("kb_id", ""),
                    },
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
        knowledge_base_ids: Sequence[str],
    ) -> List[CandidateDoc]:
        """Path B: 语义向量检索 + 反向追溯图谱实体以扩展上下文。

        1. 对原始查询和所有主题做向量检索
        2. 命中的 chunk 反向查找它关联了哪些图谱实体
        3. 如果有关联实体，把该实体的描述也作为候选加入
        """
        from app.rag.retriever import get_retriever

        retriever = get_retriever()
        docs: List[CandidateDoc] = []

        # 单查询语义检索（由 _parallel_retrieve 对每个子问题独立调用，实现逐子问题检索）
        # score_threshold 过滤低相似度候选：每个子问题最多取 top_k_per_path 条，
        # 且仅保留相似度 >= cfg.RAG_SCORE_THRESHOLD 的（宁缺毋滥，不凑数）
        try:
            raw_docs = retriever.retrieve(
                query,
                top_k=self.top_k_per_path,
                knowledge_base_ids=knowledge_base_ids,
                score_threshold=cfg.RAG_SCORE_THRESHOLD,
            )
        except Exception:
            return []

        seen_keys = set()
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
                    matched = graph_cache.match_entities(
                        list(set(keywords))[:5],
                        top_n=3,
                        knowledge_base_ids=knowledge_base_ids,
                    )
                    for ent in matched:
                        # 跳过结构性低信息量节点（chapter/book/part/law）：只用于图遍历导航，不作检索证据，
                        # 避免「第六章」这类低信息量节点跨文档污染最终结果
                        if ent.get("type") in _STRUCTURAL_NODE_TYPES:
                            continue
                        graph_extensions.append({
                            "entity": ent["name"],
                            "content": f"[实体] {ent['name']} ({ent['type']}): {ent.get('description','')}",
                            "score": 0.75,
                            "knowledge_base_id": ent.get("kb_id", ""),
                        })
                for ext in graph_extensions:
                    ext_doc = CandidateDoc(
                        content=ext["content"],
                        metadata={
                            "source": "knowledge_graph_reverse",
                            "entity": ext["entity"],
                            "knowledge_base_id": ext.get("knowledge_base_id", ""),
                        },
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

    def _retrieve_relation_path(
        self,
        decomposition: QueryDecomposition,
        knowledge_base_ids: Sequence[str],
    ) -> List[CandidateDoc]:
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

            chains = graph_cache.get_relations_by_predicate(
                search_entities,
                all_predicates,
                knowledge_base_ids=knowledge_base_ids,
            )

            for chain in chains:
                chain_text_parts = []
                for step in chain.get("steps", []):
                    chain_text_parts.append(
                        f"{step['source']} --[{step['relation']}]--> "
                        f"{step['target']}: {step.get('description','')}"
                    )
                docs.append(CandidateDoc(
                    content="\n".join(chain_text_parts),
                    metadata={
                        "source": "knowledge_graph_chain",
                        "chain_length": len(chain.get("steps", [])),
                        "knowledge_base_id": chain.get("kb_id", ""),
                    },
                    score=0.9,
                    retrieval_path="relation",
                    graph_entities=[s.get("source", "") for s in chain.get("steps", [])]
                    + [s.get("target", "") for s in chain.get("steps", [])],
                ))

        except Exception as exc:
            logger.warning("[enhanced] Path C relation search failed: %s", exc)

        return docs

    # ── Path D: 全文精确路径 (BM25) ──────────────────────────────────────────

    def _retrieve_bm25_path(
        self,
        query: str,
        knowledge_base_ids: Sequence[str],
    ) -> List[CandidateDoc]:
        """Path D: BM25 稀疏检索 — 精确匹配数字、代码、专有名词。

        BM25 是词法召回：字面命中不等于语义相关（如泛化词「登记」「处分」）。
        因此对每条 BM25 命中做向量相似度二次确认，把真实相似度写入
        score / metadata.score——融合重排的 vector_sim 维度用真实相似度，
        不再被 BM25 分数冒名顶替（否则字面噪音在融合时拿到虚高的向量分）。
        RAG_SCORE_THRESHOLD > 0 时，相似度低于阈值的 BM25 命中直接剔除。
        """
        try:
            results = self.bm25.search(
                query,
                top_k=self.top_k_per_path,
                knowledge_base_ids=knowledge_base_ids,
            )
        except Exception as exc:
            logger.debug("[enhanced] Path D BM25 failed: %s", exc)
            return []
        if not results:
            return []

        # 向量二次确认（embedding 失败时回退 BM25 分，不阻塞）
        sims = None
        try:
            import numpy as np
            embedder = get_embedder()
            q_vec = np.array(embedder.embed_query(query), dtype=float)
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
            d_vecs = embedder.embed_texts([r["content"] for r in results])
            sims = []
            for dv in d_vecs:
                d_vec = np.array(dv, dtype=float)
                sims.append(float(np.dot(q_vec, d_vec / (np.linalg.norm(d_vec) + 1e-9))))
        except Exception:
            sims = None

        threshold = cfg.RAG_SCORE_THRESHOLD
        docs: List[CandidateDoc] = []
        for i, r in enumerate(results):
            if sims is not None:
                sim = sims[i]
                if threshold > 0 and sim < threshold:
                    continue  # 低相似度 BM25 命中剔除（宁缺毋滥）
                score = sim
                meta = dict(r.get("metadata") or {})
                meta["score"] = round(sim, 4)
            else:
                score = r["score"] * 0.5  # 兜底：BM25 分数归一化
                meta = r.get("metadata", {})
            docs.append(CandidateDoc(
                content=r["content"],
                metadata=meta,
                score=score,
                retrieval_path="bm25",
            ))
        return docs

    def _retrieve_bm25_multi_path(
        self,
        queries: Sequence[str],
        knowledge_base_ids: Sequence[str],
    ) -> List[CandidateDoc]:
        """BM25 对多个查询（原始 query + 子问题）独立检索并合并去重。

        子问题独立检索能确保每个子问题的关键词证据（如「逾期」）都被召回，
        而不是只搜原始 query 导致某些子问题的证据被淹没。
        """
        docs: List[CandidateDoc] = []
        seen: Set[str] = set()
        for q in queries:
            for d in self._retrieve_bm25_path(q, knowledge_base_ids):
                key = self._dedup_key(d)
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
        return docs

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

        返回值 [0, 1]：1.0 = 精确命中；子串命中按重叠度降权；无命中 = 0.5（中性）。
        区分精确/子串命中：宽泛概念命中具体实体不应拿满分，否则图结构会放大
        「主题相关但答非所问」的候选。
        """
        if not query_entities or not doc.graph_entities:
            return 0.5

        best = 0.5
        for qe in query_entities:
            for ge in doc.graph_entities:
                if not qe or not ge:
                    continue
                if qe == ge:
                    return 1.0  # 精确命中
                if qe in ge or ge in qe:
                    # 子串命中：按重叠度降权（宽泛词命中长实体名，不给满分）
                    overlap = min(len(qe), len(ge)) / max(len(qe), len(ge))
                    best = max(best, 0.6 + 0.3 * overlap)

        return best

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

    def _apply_answerability(
        self,
        query: str,
        decomposition: QueryDecomposition,
        docs: List[CandidateDoc],
        top_k: int = 10,
    ) -> List[CandidateDoc]:
        """LLM 判断候选能否「直接回答」查询（answerability），融合到排序分数。

        answerability 与语义相似度各占一半权重：「能回答」应压过「主题像」，
        避免主题相关但答非所问的候选排前面。LLM 失败时返回原排序（不阻塞）。
        """
        if not docs:
            return docs

        n = min(len(docs), top_k)
        candidates = docs[:n]
        # 清理候选文本：移除可能导致 prompt 解析失败的特殊字符
        cand_text_parts = []
        for i, d in enumerate(candidates):
            # 截取内容并清理控制字符
            content = d.content[:120].replace('\r', ' ').replace('\n', ' ').strip()
            cand_text_parts.append(f"[{i + 1}] {content}")
        cand_text = "\n".join(cand_text_parts)

        sub_questions_str = "\n".join(
            f"  {i + 1}. {sq}" for i, sq in enumerate(decomposition.sub_questions)
        ) or "  （无）"

        try:
            # 增加 max_tokens 到 2500，避免长 prompt 导致输出被截断为空
            data = self.llm.chat_json_sync([{
                "role": "user",
                "content": _ANSWERABILITY_PROMPT.format(
                    query=query,
                    sub_questions=sub_questions_str,
                    candidates=cand_text,
                ),
            }], max_tokens=2500)
            scores = data.get("scores", {}) if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(
                "[enhanced] answerability LLM failed: %s, keeping fusion order", exc
            )
            return docs

        # 融合：answerability 0.5 + fusion 0.5；并用 answerability 的子问题标注替换检索命中标注
        sub_questions_list = decomposition.sub_questions or []
        for i, d in enumerate(candidates):
            entry = scores.get(str(i + 1), 0.5)
            if isinstance(entry, dict):
                ans = entry.get("answerability", 0.5)
                ans_subs = entry.get("sub_questions", [])
            else:
                # 兼容旧格式 {"1": 0.9}
                ans = entry
                ans_subs = []
            try:
                ans = float(ans)
            except (TypeError, ValueError):
                ans = 0.5
            ans = max(0.1, min(1.0, ans))  # 下限 0.1：避免 LLM 随机误判把正确候选完全清零
            d.metadata["answerability"] = round(ans, 4)
            # 乘法否决：answerability 低分候选被压到接近 0（「多次错误 ≠ 更正确」，
            # 一个维度为 0 则整体归零），而非加法加成放大错误命中
            d.score = d.score * ans
            # answerability 的子问题标注（能回答哪些子问题，而非被哪些子问题检索命中）
            mapped: List[str] = []
            for idx in ans_subs:
                try:
                    si = int(idx) - 1
                    if 0 <= si < len(sub_questions_list):
                        mapped.append(sub_questions_list[si])
                except (TypeError, ValueError):
                    continue
            # answerability 明确返回了子问题标注字段时，无论是否为空都替换（清空假阳性标注）
            if isinstance(entry, dict) and "sub_questions" in entry:
                d.sub_questions = mapped

        ranked_head = sorted(candidates, key=lambda d: d.score, reverse=True)
        logger.info(
            "[enhanced] answerability: %d candidates scored, top=%.4f",
            n, ranked_head[0].score if ranked_head else 0,
        )
        return ranked_head + docs[n:]

    def _coverage_gate(
        self,
        query: str,
        decomposition: QueryDecomposition,
        ranked: List[CandidateDoc],
        knowledge_base_ids: Sequence[str],
    ) -> Tuple[List[CandidateDoc], bool]:
        """逐子问题覆盖检查（基于 answerability 的子问题标注）。

        某个子问题在 top-K 候选里没有「能回答」它的证据时，用该子问题文本做
        针对性语义 + BM25 补充检索，确保每个子问题至少有一个 answerable 证据。
        返回 (排序后的候选, 是否所有子问题均已覆盖)，供上层决定是否还需 LLM 缺口补充。
        """
        sub_questions = decomposition.sub_questions or []
        if not sub_questions:
            return ranked, True

        covered = {sq for d in ranked for sq in d.sub_questions}
        uncovered = [sq for sq in sub_questions if sq not in covered]
        if not uncovered:
            return ranked, True

        logger.info(
            "[enhanced] coverage gate: %d/%d sub-questions uncovered → supplement",
            len(uncovered), len(sub_questions),
        )

        seen = {self._dedup_key(d) for d in ranked}
        extra: List[CandidateDoc] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for sq in uncovered[:3]:
                futures.append((sq, executor.submit(
                    self._retrieve_semantic_path, sq, decomposition, knowledge_base_ids
                )))
                futures.append((sq, executor.submit(
                    self._retrieve_bm25_path, sq, knowledge_base_ids
                )))
            for sq, f in futures:
                try:
                    docs = f.result(timeout=15)
                except Exception:
                    docs = []
                # 每个路径只保留 top-3，避免候选爆炸拖慢 answerability LLM 打分
                for d in docs[:3]:
                    key = self._dedup_key(d)
                    if key in seen:
                        continue
                    seen.add(key)
                    if sq not in d.sub_questions:
                        d.sub_questions.append(sq)
                    d.retrieval_path += "+coverage"
                    extra.append(d)

        if extra:
            extra = self._fusion_rerank(extra, decomposition)
            # 补充候选同样经过 answerability 乘法否决，拒绝补进来的伪相关
            # （限 top-6 打分，控制 LLM prompt 长度与耗时）
            extra = self._apply_answerability(query, decomposition, extra, top_k=6)
            ranked = ranked + extra

        # 补充后重新检查覆盖：所有子问题都有 answerable 证据则无需再 LLM 缺口补充
        covered_after = {sq for d in ranked for sq in d.sub_questions}
        still_uncovered = [sq for sq in sub_questions if sq not in covered_after]
        return ranked, not still_uncovered

    @staticmethod
    def _ensure_per_sub_question_minimum(
        docs: List[CandidateDoc],
        decomposition: QueryDecomposition,
        min_per_sub: int = 3,
    ) -> List[CandidateDoc]:
        """每个子问题至少保留 top-min_per_sub 候选，避免被其他子问题挤掉。

        复杂查询拆成多个子问题后，若所有子问题共享同一候选池竞争 final_top_k，
        弱子问题的证据会被强子问题淹没。这里把每个子问题的 top-min_per_sub
        候选提升到前面，保证每个子问题都有证据进入最终 Top-K。
        """
        sub_questions = decomposition.sub_questions or []
        if not sub_questions or not docs:
            return docs

        by_sub: Dict[str, List[CandidateDoc]] = {sq: [] for sq in sub_questions}
        for d in docs:
            for sq in d.sub_questions:
                if sq in by_sub:
                    by_sub[sq].append(d)

        guaranteed: List[CandidateDoc] = []
        seen: Set[int] = set()
        for sq in sub_questions:
            top = sorted(by_sub[sq], key=lambda d: d.score, reverse=True)[:min_per_sub]
            for d in top:
                if id(d) not in seen:
                    seen.add(id(d))
                    guaranteed.append(d)

        if not guaranteed:
            return docs

        rest = [d for d in docs if id(d) not in seen]
        return guaranteed + rest

    def _apply_per_sub_question_quota(
        self,
        docs: List[CandidateDoc],
        decomposition: QueryDecomposition,
        min_per_sub: int = 2,
        final_top_k: int = 8,
    ) -> List[CandidateDoc]:
        """每子问题硬配额：弱子问题证据不被强子问题挤掉（替代全局截断）。

        全局 `[:final_top_k]` 截断会让所有子问题共享一个排序列表竞争名额——
        弱子问题（候选少、分数低）的证据被强子问题淹没，即使做了保底提升，
        截断也会再次把保底候选切掉（min_per_sub × 子问题数 > final_top_k 时必然发生）。

        配额逻辑：
        1. 每个子问题选 top min_per_sub（按分数）
        2. 无子问题标注的候选（原始查询级）选 top min_per_sub
        3. 剩余名额按全局分数补满到 final_top_k
        返回长度 ≤ final_top_k，且每个子问题至少 min_per_sub 条。
        """
        if not docs:
            return docs
        sub_questions = decomposition.sub_questions or []
        if not sub_questions:
            return docs[:final_top_k]

        by_sub: Dict[str, List[CandidateDoc]] = {sq: [] for sq in sub_questions}
        query_level: List[CandidateDoc] = []
        for d in docs:
            if d.sub_questions:
                for sq in d.sub_questions:
                    if sq in by_sub:
                        by_sub[sq].append(d)
            else:
                query_level.append(d)

        picked: List[CandidateDoc] = []
        seen: Set[int] = set()

        def _pick(d: CandidateDoc) -> bool:
            if id(d) in seen:
                return False
            seen.add(id(d))
            picked.append(d)
            return True

        # 1) 每子问题保底 min_per_sub 条
        for sq in sub_questions:
            top = sorted(by_sub[sq], key=lambda d: d.score, reverse=True)[:min_per_sub]
            for d in top:
                _pick(d)
        # 2) 原始查询级保底
        for d in sorted(query_level, key=lambda x: x.score, reverse=True)[:min_per_sub]:
            _pick(d)
        # 3) 剩余名额按全局分数补满
        if len(picked) < final_top_k:
            for d in sorted(docs, key=lambda x: x.score, reverse=True):
                if len(picked) >= final_top_k:
                    break
                _pick(d)
        return picked[:final_top_k]

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
        knowledge_base_ids: Optional[Sequence[str]] = None,
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

            block_relations = self._collect_block_relations(
                block_entities[:10], knowledge_base_ids,
                entity_sources=self._collect_entity_sources(block_docs),
            )
            # 聚合该知识块覆盖的子问题（去重保序）
            block_sub_questions: List[str] = []
            for d in block_docs:
                for sq in d.sub_questions:
                    if sq not in block_sub_questions:
                        block_sub_questions.append(sq)
            block = KnowledgeBlock(
                block_id=f"block_{root}",
                entities=block_entities[:10],
                relations=block_relations,
                docs=block_docs,
                block_score=round(avg_score, 4),
                sub_questions=block_sub_questions,
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
                    sub_questions=docs[i].sub_questions,
                ))

        logger.info("[enhanced] clustered %d docs → %d blocks", len(docs), len(blocks))
        return blocks

    @staticmethod
    def _collect_entity_sources(docs: List[CandidateDoc]) -> Dict[str, Optional[str]]:
        """聚合块内文档携带的实体→来源文件映射（命名空间隔离用）。

        实体路径的文档在 metadata.entity_source_files 里记录了每个图谱实体的
        所属文件；其他路径（语义/BM25）没有该信息，返回空 dict 走聚合兜底。
        """
        merged: Dict[str, Optional[str]] = {}
        for d in docs:
            sf_map = (d.metadata or {}).get("entity_source_files") or {}
            for ent, sf in sf_map.items():
                if ent and ent not in merged:
                    merged[ent] = sf or None
        return merged

    def _collect_block_relations(
        self,
        entities: List[str],
        knowledge_base_ids: Optional[Sequence[str]] = None,
        entity_sources: Optional[Dict[str, Optional[str]]] = None,
    ) -> List[Dict[str, str]]:
        """从图谱缓存收集块内实体之间的关系（两端都在块内实体集合内）。

        KnowledgeBlock.relations 此前恒为空——聚类时丢失了图谱关系信息，
        导致前端「关系」展示与 format_blocks_for_prompt 的关系注入都失效。

        entity_sources 提供实体→来源文件映射时，按命名空间精确查询邻居
        （同名实体不跨文件混边）；缺省时按 name 聚合查询（老数据兜底）。
        """
        if not entities or not knowledge_base_ids:
            return []
        from app.rag.graph_cache import graph_cache

        entity_sources = entity_sources or {}
        entity_set = set(entities)
        relations: List[Dict[str, str]] = []
        seen: set = set()

        for ent in entities:
            try:
                neighbors = graph_cache.get_neighbor_relations(
                    ent, max_relations=20, knowledge_base_ids=knowledge_base_ids,
                    source_file=entity_sources.get(ent),
                )
            except Exception as exc:
                logger.warning(
                    "[enhanced] collect relations for %r failed: %s", ent, exc
                )
                continue
            for rel in neighbors:
                src = rel.get("source", "")
                tgt = rel.get("target", "")
                # 只保留两端都在块内实体的关系（块内边）
                if not src or not tgt or src not in entity_set or tgt not in entity_set:
                    continue
                key = (src, rel.get("relation", ""), tgt)
                if key in seen:
                    continue
                seen.add(key)
                relations.append({
                    "source": src,
                    "relation": rel.get("relation", ""),
                    "target": tgt,
                    "description": rel.get("description", ""),
                })
                if len(relations) >= 12:
                    return relations
        return relations

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
        knowledge_base_ids: Sequence[str],
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

            # 对每个缺口查询做补充检索（并行，仅 Path B 语义路径 + Path D BM25）
            gap_futures = {}
            for gq in gap_queries[:3]:
                gap_futures[gq] = self._executor.submit(
                    self._gap_supplement_search, gq, knowledge_base_ids
                )

            seen_keys = {self._dedup_key(d) for d in result.raw_docs}
            for gq, future in gap_futures.items():
                try:
                    raw = future.result(timeout=15)
                except Exception as exc:
                    logger.warning("[enhanced] gap query %r failed: %s", gq[:40], exc)
                    continue

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
        # 构建已检索内容的摘要（带子问题来源标注，便于 LLM 精确判断哪个子问题未覆盖）
        summary_parts = []
        for i, block in enumerate(result.knowledge_blocks[:5]):
            sq_tag = (
                f"（回答子问题: {'；'.join(block.sub_questions)}）"
                if block.sub_questions else "（原始查询）"
            )
            summary_parts.append(f"[知识块{i+1}]{sq_tag} {block.summary}")
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

    def _gap_supplement_search(
        self,
        query: str,
        knowledge_base_ids: Sequence[str],
    ) -> List[dict]:
        """缺口补充的单次检索（语义向量），供并行调用。"""
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        return retriever.retrieve(
            query,
            top_k=3,
            knowledge_base_ids=knowledge_base_ids,
        )

    def _ensure_bm25_index(self):
        """确保 BM25 索引已从向量库同步（超时保护，失败不抛异常）。"""
        if self.bm25.doc_count > 0:
            self._bm25_ready.set()
            return

        try:
            self.sync_bm25_from_vector_store()
        except Exception as exc:
            logger.warning("[enhanced] BM25 index sync failed (Path D unavailable): %s", exc)
        finally:
            self._bm25_ready.set()

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
                            output_fields=[
                                "content", "source", "knowledge_base_id"
                            ],
                            offset=offset,
                            limit=page_size,
                        )
                        for row in res:
                            all_chunks.append({
                                "id": row.get("id", ""),
                                "content": row.get("content", ""),
                                "metadata": {
                                    "source": row.get("source", "unknown"),
                                    "knowledge_base_id": row.get(
                                        "knowledge_base_id", ""
                                    ),
                                },
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
        """从候选文档提取去重的来源列表。

        过滤掉内部标记来源（knowledge_graph / knowledge_graph_reverse / knowledge_graph_chain），
        只保留真实的文件来源。
        """
        _INTERNAL_SOURCES = {
            "knowledge_graph", "knowledge_graph_reverse",
            "knowledge_graph_chain", "knowledge_graph",
        }
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for doc in docs:
            src = (doc.metadata.get("source") or "").strip()
            # 跳过内部来源标记
            if not src or src in seen or src in _INTERNAL_SOURCES:
                continue
            seen.add(src)
            sources.append({
                "title": src,
                "url": doc.metadata.get("url", ""),
                "type": doc.metadata.get("type", "kb"),
                "score": round(doc.score, 4),
                "retrieval_path": doc.retrieval_path,
                "knowledge_base_id": doc.metadata.get("knowledge_base_id", ""),
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

        # 覆盖的子问题（来源标注：该知识块检索自哪个子问题）
        if block.sub_questions:
            lines.append(f"### 回答的子问题: {'；'.join(block.sub_questions)}")

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
        # 从配置读取每路径/最终 Top-K（ENHANCED_TOP_K_PER_PATH / ENHANCED_FINAL_TOP_K），
        # 否则会落到 __init__ 的硬编码默认值（8/12），config 里配了也不生效
        _enhanced = EnhancedRetriever(
            top_k_per_path=cfg.ENHANCED_TOP_K_PER_PATH,
            final_top_k=cfg.ENHANCED_FINAL_TOP_K,
        )
    return _enhanced
