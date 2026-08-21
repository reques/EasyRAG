"""Tests for retrieval-side fixes: entity description w/ body, description matching, sub-question semantic search, domain-agnostic prompt."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

from app.rag import enhanced_retriever as er
from app.rag.graph_cache import graph_cache
from backend.services import graph_service as gs
from backend.storage.postgres.models_knowledge import KnowledgeEntity


def _extract(chunks) -> tuple[MagicMock, uuid.UUID]:
    kb = uuid.uuid4()
    session = MagicMock()
    asyncio.run(gs.extract_graph_from_chunks(session, kb, chunks, "doc.txt"))
    return session, kb


def _legal_chunks() -> list:
    return [
        ("[第十二章 借款合同]\n第六百八十条　【禁止高利放贷】 禁止高利放贷，借款的利率不得违反国家有关规定。借款合同对支付利息没有约定的，视为没有利息。", {"chunk_index": 0}),
        ("[第十二章 借款合同]\n第六百七十四条　【借款人支付利息的期限】 借款人应当按照约定的期限支付利息。", {"chunk_index": 1}),
        ("[第十二章 借款合同]\n第六百七十六条　【借款合同内容】 借款合同的内容一般包括借款种类、数额、利率、期限等。", {"chunk_index": 2}),
    ]


def test_article_description_includes_body():
    session, _ = _extract(_legal_chunks())
    e680 = next(
        c.args[0]
        for c in session.add.call_args_list
        if isinstance(c.args[0], KnowledgeEntity) and "第六百八十条" in c.args[0].name
    )
    assert "视为没有利息" in e680.description


def test_match_entities_matches_description():
    _, kb = _extract(_legal_chunks())
    matched = graph_cache.match_entities(
        ["视为没有利息"], top_n=10, knowledge_base_ids=[str(kb)]
    )
    assert any("第六百八十条" in m["name"] for m in matched)


def test_semantic_path_is_single_query():
    # semantic path 现在是单查询检索，逐子问题遍历由 _parallel_retrieve 负责
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    retr.top_k_per_path = 6
    decomp = er.QueryDecomposition(
        explicit_entities=[],
        themes=[{"theme": "t1", "scope": "broad"}],
        sub_questions=["subA", "subB"],
        query_type="factual",
        complexity="medium",
    )
    queries = []
    fake = MagicMock()
    fake.retrieve = lambda q, top_k=6, knowledge_base_ids=None, score_threshold=0.0: (queries.append(q) or [])
    with patch("app.rag.retriever.get_retriever", return_value=fake):
        retr._retrieve_semantic_path("orig", decomp, [str(uuid.uuid4())])
    assert queries == ["orig"]


def test_parallel_retrieve_searches_each_sub_question_independently():
    # 每个子问题独立检索（semantic + bm25）并标注来源子问题
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    semantic_calls, bm25_calls = [], []

    def fake_semantic(q, decomposition, kb_ids):
        semantic_calls.append(q)
        return [er.CandidateDoc(content=f"sem_{q}", metadata={"knowledge_base_id": "kb"}, score=0.8)]

    def fake_bm25(q, kb_ids):
        bm25_calls.append(q)
        return [er.CandidateDoc(content=f"bm25_{q}", metadata={"knowledge_base_id": "kb"}, score=0.7)]

    retr._retrieve_semantic_path = fake_semantic
    retr._retrieve_bm25_path = fake_bm25
    retr._retrieve_entity_path = lambda d, kb: []
    retr._retrieve_relation_path = lambda d, kb: []

    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=["subA", "subB"],
        query_type="factual", complexity="high",
    )
    docs = retr._parallel_retrieve("orig", decomp, ["kb"])

    # query + 2 子问题，各检索一次 semantic 和 bm25
    assert len(semantic_calls) == 3 and len(bm25_calls) == 3
    for sq in ["subA", "subB"]:
        assert sq in semantic_calls and sq in bm25_calls
    by_content = {d.content: d for d in docs}
    assert by_content["sem_subA"].sub_questions == ["subA"]
    assert by_content["sem_orig"].sub_questions == []


def test_query_decomposition_prompt_is_domain_agnostic():
    for w in ["利息", "自然人", "借款", "法律", "legal_concept", "条文"]:
        assert w not in er._QUERY_DECOMPOSITION_PROMPT


def test_graph_proximity_penalizes_broad_terms():
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)

    # 宽泛词命中长实体名 → 降权（非满分），避免图结构放大"主题相关但答非所问"的候选
    doc = er.CandidateDoc(content="", graph_entities=["第六百六十八条【借款合同形式和内容】"])
    assert retr._compute_graph_proximity(doc, ["借款合同"]) < 0.8

    # 精确命中 → 1.0
    doc2 = er.CandidateDoc(content="", graph_entities=["借款合同"])
    assert retr._compute_graph_proximity(doc2, ["借款合同"]) == 1.0

    # 无图命中 → 中性 0.5
    doc3 = er.CandidateDoc(content="")
    assert retr._compute_graph_proximity(doc3, ["借款合同"]) == 0.5


def test_bm25_multi_path_searches_each_query():
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    called = []

    def fake_bm25(q, kb_ids):
        called.append(q)
        return []

    retr._retrieve_bm25_path = fake_bm25
    retr._retrieve_bm25_multi_path(["q0", "q1", "q2"], ["kb"])
    assert called == ["q0", "q1", "q2"]


def test_answerability_reranks_by_direct_answer():
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    d1 = er.CandidateDoc(content="主题相关但答非所问", score=0.9)
    d2 = er.CandidateDoc(content="直接回答子问题1的答案", score=0.7)
    retr._llm = MagicMock()
    retr._llm.chat_json_sync = MagicMock(return_value={
        "scores": {
            "1": {"answerability": 0.1, "sub_questions": []},
            "2": {"answerability": 1.0, "sub_questions": [1]},
        }
    })
    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=["子问题1", "子问题2"],
        query_type="factual", complexity="low",
    )
    ranked = retr._apply_answerability("query", decomp, [d1, d2])
    # 乘法否决：高 answerability 候选（0.7×1.0）压过主题像候选（0.9×0.1）
    assert ranked[0].content == "直接回答子问题1的答案"
    assert abs(ranked[0].score - 0.7) < 1e-6
    assert abs(ranked[1].score - 0.09) < 1e-6
    # answerability 的子问题标注替换检索命中标注：答非所问的候选无标注
    assert ranked[0].sub_questions == ["子问题1"]
    assert ranked[1].sub_questions == []


def test_answerability_llm_failure_keeps_order():
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    retr._llm = MagicMock()
    retr._llm.chat_json_sync = MagicMock(side_effect=Exception("down"))
    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=["q"],
        query_type="factual", complexity="low",
    )
    docs = [er.CandidateDoc(content="a", score=0.8), er.CandidateDoc(content="b", score=0.6)]
    assert retr._apply_answerability("q", decomp, docs) == docs


def test_coverage_gate_supplements_uncovered_sub_questions():
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    sem_calls = []
    retr._retrieve_semantic_path = lambda q, d, kb: (
        sem_calls.append(q) or [er.CandidateDoc(content=f"sem_{q}", metadata={"knowledge_base_id": "kb"}, score=0.6)]
    )
    retr._retrieve_bm25_path = lambda q, kb: []
    retr._fusion_rerank = lambda docs, d: docs

    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=["子问题1", "子问题2", "子问题3"],
        query_type="factual", complexity="high",
    )
    # 只有子问题1 被覆盖，子问题2/3 应被针对性补充检索
    ranked = [er.CandidateDoc(content="x", sub_questions=["子问题1"])]
    # answerability LLM 降级（返回原排序），不影响补充候选保留
    retr._llm = MagicMock()
    retr._llm.chat_json_sync = MagicMock(side_effect=Exception("no llm"))
    out, covered_all = retr._coverage_gate("query", decomp, ranked, ["kb"])
    assert "子问题2" in sem_calls and "子问题3" in sem_calls
    # 补充的候选标注了来源子问题
    assert any(d.sub_questions == ["子问题2"] for d in out)
    assert any(d.sub_questions == ["子问题3"] for d in out)
    assert covered_all is True  # 补充后所有子问题已覆盖


def test_graph_reverse_filters_structural_nodes():
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    retr.top_k_per_path = 6
    fake_retriever = MagicMock()
    fake_retriever.retrieve = lambda q, top_k=6, knowledge_base_ids=None, score_threshold=0.0: [
        {"content": "内容", "metadata": {"score": 0.8, "knowledge_base_id": "kb"}}
    ]
    fake_cache = MagicMock()
    fake_cache.match_entities = MagicMock(return_value=[
        {"name": "第六章 法律责任", "type": "chapter", "description": "", "kb_id": "kb"},
        {"name": "第28条", "type": "article", "description": "培训义务", "kb_id": "kb"},
    ])
    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=["q"],
        query_type="factual", complexity="low",
    )
    with patch("app.rag.retriever.get_retriever", return_value=fake_retriever), \
         patch("app.rag.graph_cache.graph_cache", fake_cache):
        docs = retr._retrieve_semantic_path("q", decomp, ["kb"])
    names = [d.graph_entities[0] for d in docs if "graph_reverse" in d.retrieval_path]
    assert "第28条" in names  # article 保留
    assert not any("第六章" in n for n in names)  # chapter 过滤


def test_per_sub_question_quota_guarantees_weak_sub_question():
    # 弱子问题候选分数低、数量少，全局截断会把它挤出 final_top_k；
    # 每子问题硬配额必须保证它至少 min_per_sub 条进最终 Top-K
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[],
        sub_questions=["强子问题", "弱子问题"],
        query_type="factual", complexity="high",
    )
    docs = []
    # 强子问题 6 个高分候选
    for i in range(6):
        docs.append(er.CandidateDoc(
            content=f"strong_{i}", score=0.95 - i * 0.01,
            sub_questions=["强子问题"],
        ))
    # 弱子问题 2 个低分候选（0.40 / 0.35 —— 全局排名 7、8，在 final_top_k=6 之外）
    docs.append(er.CandidateDoc(content="weak_0", score=0.40, sub_questions=["弱子问题"]))
    docs.append(er.CandidateDoc(content="weak_1", score=0.35, sub_questions=["弱子问题"]))
    # 原始查询级候选（无标注）
    docs.append(er.CandidateDoc(content="query_level", score=0.90, sub_questions=[]))

    out = retr._apply_per_sub_question_quota(
        docs, decomp, min_per_sub=2, final_top_k=6,
    )
    assert len(out) <= 6
    weak = [d for d in out if "弱子问题" in d.sub_questions]
    strong = [d for d in out if "强子问题" in d.sub_questions]
    assert len(weak) >= 2   # 弱子问题保底进 2 条
    assert len(strong) >= 2
    # 原始查询级也有保底
    assert any(d.content == "query_level" for d in out)


def test_bm25_path_uses_real_vector_similarity():
    # BM25 是词法召回：融合时 vector_sim 维度必须用真实向量相似度，
    # 不能拿 BM25 分数冒名顶替（否则字面噪音拿到虚高的向量分）
    retr = er.EnhancedRetriever.__new__(er.EnhancedRetriever)
    retr.top_k_per_path = 6
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = [
        {"id": "1", "content": "字面命中但语义无关的内容", "metadata": {}, "score": 12.0},
        {"id": "2", "content": "真正相关的语义内容", "metadata": {}, "score": 8.0},
    ]
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [1.0, 0.0]
    # 已归一化：第一条与 query 余弦相似度 0.1（噪音），第二条 0.9（真相关）
    fake_embedder.embed_texts.return_value = [[0.1, 0.995], [0.9, 0.436]]

    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=[],
        query_type="factual", complexity="low",
    )
    with patch("app.rag.enhanced_retriever.get_embedder", return_value=fake_embedder), \
         patch.object(er, "cfg") as mock_cfg:
        mock_cfg.RAG_SCORE_THRESHOLD = 0.0  # 阈值 0：不剔除，但分数必须真实
        retr._bm25 = fake_bm25  # bm25 是只读 property，直接注入实例属性
        docs = retr._retrieve_bm25_path("查询", ["kb"])
    assert len(docs) == 2
    # BM25 高分（12.0）≠ 真实相似度（0.1）：score 必须用真实相似度
    assert abs(docs[0].score - 0.1) < 1e-4
    assert abs(docs[1].score - 0.9) < 1e-4
    assert abs(docs[0].metadata["score"] - 0.1) < 1e-4
    # 阈值 > 0 时低相似度 BM25 命中被剔除
    with patch("app.rag.enhanced_retriever.get_embedder", return_value=fake_embedder), \
         patch.object(er, "cfg") as mock_cfg:
        mock_cfg.RAG_SCORE_THRESHOLD = 0.5
        retr._bm25 = fake_bm25
        docs2 = retr._retrieve_bm25_path("查询", ["kb"])
    assert len(docs2) == 1
    assert abs(docs2[0].score - 0.9) < 1e-4


def test_per_sub_question_minimum_keeps_weak_sub_questions():
    # 弱子问题的证据（score 低）应被提升到前面，不被强子问题挤掉
    subs = ["培训", "设备", "赔偿", "过错"]
    docs = [er.CandidateDoc(content=f"培训_{i}", score=0.9 - i * 0.05, sub_questions=["培训"]) for i in range(5)]
    docs.append(er.CandidateDoc(content="过错_核心证据", score=0.3, sub_questions=["过错"]))
    docs.append(er.CandidateDoc(content="设备_核心证据", score=0.35, sub_questions=["设备"]))
    decomp = er.QueryDecomposition(
        explicit_entities=[], themes=[], sub_questions=subs,
        query_type="factual", complexity="high",
    )
    out = er.EnhancedRetriever._ensure_per_sub_question_minimum(docs, decomp, min_per_sub=3)
    guaranteed = [d.content for d in out[:5]]
    assert "过错_核心证据" in guaranteed
    assert "设备_核心证据" in guaranteed
