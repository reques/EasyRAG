"""run_evaluation 的 reference 构造语义测试。

核心断言：reference 必须是 question-specific 的相关 chunk 集，
而不是整份文件兜底；负样本单独走误报检测，不污染常规指标。
"""

from __future__ import annotations

from backend.services import evaluation_service


def _doc(chunk_id, source, score=0.8, file_id=None):
    metadata = {"source": source, "score": score}
    if file_id:
        metadata["file_id"] = file_id
    if chunk_id:
        metadata["chunk_id"] = chunk_id
    return {"content": f"content-{chunk_id}", "metadata": metadata}


class FakeRetriever:
    def __init__(self, docs, file_chunks=None):
        self.docs = docs
        self.file_chunks = file_chunks or []

    def retrieve(self, _question, **_kwargs):
        return [dict(doc) for doc in self.docs]

    def list_chunks_by_source(self, _kb_id, _source):
        return list(self.file_chunks)


FILE_ID = "33333333-3333-3333-3333-333333333333"
KB_ID = "11111111-1111-1111-1111-111111111111"


def _run(cases, retriever, monkeypatch, top_k=5):
    monkeypatch.setattr("app.rag.retriever.get_retriever", lambda: retriever)
    monkeypatch.setattr(evaluation_service.cfg, "RAGAS_ENABLED", False)
    return evaluation_service.run_evaluation(
        cases,
        top_k=top_k,
        knowledge_base_id=KB_ID,
    )


def test_chunk_ids_reference_is_question_specific(monkeypatch):
    retriever = FakeRetriever(docs=[
        _doc("chunk-hit", "paper.pdf", score=0.9, file_id=FILE_ID),
        _doc("chunk-miss", "paper.pdf", score=0.5, file_id=FILE_ID),
    ])
    metrics = _run([
        evaluation_service.EvaluationCase(
            question="q",
            expected_file_id=FILE_ID,
            expected_chunk_ids=("chunk-hit",),
            expected_source="paper.pdf",
        ),
    ], retriever, monkeypatch)

    detail = metrics["details"][0]
    assert detail["reference_mode"] == "chunk_ids"
    assert detail["expected_chunk_count"] == 1
    assert detail["chunk_hit_rank"] == 1
    # 相关集只有 1 条，Recall@K 就是 1.0（而不是被整份文件稀释）
    assert detail["chunk_metrics"]["recall_at_k"] == 1.0
    assert detail["chunk_metrics"]["precision_at_k"] == 0.2


def test_negative_case_counts_false_positive(monkeypatch):
    # 负样本：该问题不应命中 paper.pdf，但检索结果里混进了它
    retriever = FakeRetriever(docs=[
        _doc("chunk-a", "other.pdf", score=0.9, file_id="44444444-4444-4444-4444-444444444444"),
        _doc("chunk-b", "paper.pdf", score=0.6, file_id=FILE_ID),
    ])
    metrics = _run([
        evaluation_service.EvaluationCase(
            question="不该命中",
            expected_file_id=FILE_ID,
            expect_miss=True,
            expected_source="paper.pdf",
        ),
    ], retriever, monkeypatch)

    detail = metrics["details"][0]
    assert detail["reference_mode"] == "negative"
    assert detail["false_positive"] == 1
    assert metrics["analysis"]["false_positive_count"] == 1
    # 负样本不计入常规 missed 统计
    assert metrics["analysis"]["missed_count"] == 0


def test_negative_case_clean_when_not_recalled(monkeypatch):
    retriever = FakeRetriever(docs=[
        _doc("chunk-a", "other.pdf", score=0.9, file_id="44444444-4444-4444-4444-444444444444"),
    ])
    metrics = _run([
        evaluation_service.EvaluationCase(
            question="不该命中",
            expected_file_id=FILE_ID,
            expect_miss=True,
            expected_source="paper.pdf",
        ),
    ], retriever, monkeypatch)

    detail = metrics["details"][0]
    assert detail["false_positive"] == 0
    assert metrics["analysis"]["false_positive_count"] == 0


def test_file_fallback_marks_reference_mode_and_expands(monkeypatch):
    # 让检索结果的内容与文件展开内容一致，chunk id 才会对齐
    retriever = FakeRetriever(
        docs=[{
            "content": "content-1",
            "metadata": {"source": "paper.pdf", "score": 0.7, "file_id": FILE_ID},
        }],
        file_chunks=["content-1", "content-2"],
    )
    metrics = _run([
        evaluation_service.EvaluationCase(
            question="q",
            expected_file_id=FILE_ID,
            expected_source="paper.pdf",
        ),
    ], retriever, monkeypatch)

    detail = metrics["details"][0]
    assert detail["reference_mode"] == "file"
    assert detail["expected_chunk_count"] == 2
    # 兜底口径下 recall 被整文件稀释（2 条相关，只召回 1 条）
    assert detail["chunk_metrics"]["recall_at_k"] == 0.5


def test_ragas_samples_carry_question_specific_reference(monkeypatch):
    captured = {}

    class RecordingEvaluator:
        def evaluate(self, samples):
            captured["samples"] = samples
            return {"status": "completed", "metrics": {}}

    retriever = FakeRetriever(docs=[
        _doc("chunk-hit", "paper.pdf", score=0.9, file_id=FILE_ID),
    ])
    monkeypatch.setattr("app.rag.retriever.get_retriever", lambda: retriever)
    monkeypatch.setattr(evaluation_service.cfg, "RAGAS_ENABLED", True)
    monkeypatch.setattr(
        evaluation_service,
        "get_ragas_evaluator",
        lambda _settings, _metrics=None: RecordingEvaluator(),
    )

    evaluation_service.run_evaluation([
        evaluation_service.EvaluationCase(
            question="q",
            expected_file_id=FILE_ID,
            expected_chunk_ids=("chunk-hit",),
            expected_source="paper.pdf",
        ),
    ], top_k=5, knowledge_base_id=KB_ID)

    sample = captured["samples"][0]
    assert sample.reference_context_ids == ["chunk-hit"]
    assert sample.retrieved_context_ids == ["chunk-hit"]


def test_run_metadata_snapshot_is_recorded(monkeypatch):
    retriever = FakeRetriever(docs=[_doc("chunk-a", "paper.pdf", file_id=FILE_ID)])
    metrics = _run([
        evaluation_service.EvaluationCase(
            question="q",
            expected_file_id=FILE_ID,
            expected_chunk_ids=("chunk-a",),
            expected_source="paper.pdf",
        ),
    ], retriever, monkeypatch)

    meta = metrics["run_metadata"]
    assert meta["k"] == 5
    assert "chunk_strategy" in meta
    assert "embedding_type" in meta
    assert "enhanced_retrieval" in meta