from pathlib import Path


def test_upload_polling_cannot_replace_another_knowledge_base_file_list():
    source = (
        Path(__file__).parents[1] / "frontend/src/views/KnowledgeView.vue"
    ).read_text(encoding="utf-8")
    polling = source.split("async function pollIndexing", 1)[1]
    polling = polling.split("async function openPreview", 1)[0]

    assert "String(activeKb.value?.id || '') === String(kbId)" in polling
    assert "fileList.value = files" in polling


def test_stale_file_requests_are_ignored_after_knowledge_base_switch():
    source = (
        Path(__file__).parents[1] / "frontend/src/views/KnowledgeView.vue"
    ).read_text(encoding="utf-8")
    loading = source.split("async function loadFiles", 1)[1]
    loading = loading.split("async function createKb", 1)[0]

    assert "requestRevision === fileRequestRevision" in loading
    assert "selectionRevision === knowledgeSelectionRevision" in loading
    assert "String(activeKb.value?.id || '') === String(kbId)" in loading


def test_switching_knowledge_bases_invalidates_previous_selection():
    source = (
        Path(__file__).parents[1] / "frontend/src/views/KnowledgeView.vue"
    ).read_text(encoding="utf-8")
    selection = source.split("async function selectKb", 1)[1]
    selection = selection.split("async function leaveKb", 1)[0]

    assert "const selectionRevision = ++knowledgeSelectionRevision" in selection
    assert "selectionRevision !== knowledgeSelectionRevision" in selection
