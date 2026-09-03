from pathlib import Path


def knowledge_view_source() -> str:
    return (
        Path(__file__).parents[1] / "frontend/src/views/KnowledgeView.vue"
    ).read_text(encoding="utf-8")


def test_retrieval_lab_calls_the_scoped_backend_endpoint():
    source = knowledge_view_source()
    retrieval = source.split("async function runRetrievalPreview", 1)[1]
    retrieval = retrieval.split("function toggleCriterion", 1)[0]

    assert "api.post(`/knowledge/bases/${kbId}/retrieval/test`" in retrieval
    assert "query," in retrieval
    assert "top_k: retrievalTopK.value" in retrieval
    assert "score_threshold: retrievalThreshold.value" in retrieval


def test_stale_retrieval_results_are_ignored_after_kb_switch():
    source = knowledge_view_source()
    retrieval = source.split("async function runRetrievalPreview", 1)[1]
    retrieval = retrieval.split("function toggleCriterion", 1)[0]

    assert "requestRevision === retrievalRequestRevision" in retrieval
    assert "String(activeKb.value?.id || '') === kbId" in retrieval

    selection = source.split("async function selectKb", 1)[1]
    selection = selection.split("async function leaveKb", 1)[0]
    assert "resetRetrievalState()" in selection


def test_retrieval_results_render_rank_score_source_and_latency():
    source = knowledge_view_source()

    assert "retrievalRun.elapsed_ms" in source
    assert "#{{ hit.rank }}" in source
    assert "formatSimilarity(hit.score)" in source
    assert "hit.source || '未知来源'" in source
