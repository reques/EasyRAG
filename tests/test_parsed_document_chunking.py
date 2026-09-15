from __future__ import annotations

import pytest

from app.rag.chunker import cfg, chunk_parsed_document, parse_and_chunk, split_structured
from app.rag.parsers import (
    ParsedBlock,
    ParsedBlockType,
    ParsedContentFormat,
    ParsedDocument,
    ParserProvenance,
)


@pytest.mark.parametrize("explicit", [False, True])
def test_structured_chunking_preserves_pages_sections_and_atomic_tables(monkeypatch, explicit):
    monkeypatch.setattr(cfg, "CHUNK_STRATEGY", "fixed" if explicit else "structured")
    document = ParsedDocument(
        source_name="report.pdf",
        markdown="# Report\n\nIntroduction",
        page_count=2,
        source_sha256="a" * 64,
        provenance=ParserProvenance(
            parser_name="mineru",
            parser_version="3.4.4",
            task_id="task-1",
            backend="pipeline",
        ),
        blocks=(
            ParsedBlock(
                block_id="0",
                type=ParsedBlockType.HEADER,
                text="Repeated journal header",
                page_index=0,
            ),
            ParsedBlock(
                block_id="1",
                type=ParsedBlockType.TITLE,
                text="Report",
                heading_level=1,
                section_path=("Report",),
                page_index=0,
            ),
            ParsedBlock(
                block_id="2",
                type=ParsedBlockType.TEXT,
                text="First paragraph.",
                section_path=("Report",),
                page_index=0,
            ),
            ParsedBlock(
                block_id="3",
                type=ParsedBlockType.TEXT,
                text="Second paragraph.",
                section_path=("Report",),
                page_index=1,
            ),
            ParsedBlock(
                block_id="4",
                type=ParsedBlockType.TABLE,
                text="<table>" + "x" * 200 + "</table>",
                content_format=ParsedContentFormat.HTML,
                captions=("Table 1",),
                section_path=("Report",),
                page_index=1,
            ),
        ),
    )

    chunks = chunk_parsed_document(
        document, chunk_size=100, strategy="structured" if explicit else None
    )

    assert len(chunks) == 2
    text, metadata = chunks[0]
    assert "Repeated journal header" not in text
    assert text.startswith("[Report]")
    assert "First paragraph." in text and "Second paragraph." in text
    assert metadata["page_start"] == 1
    assert metadata["page_end"] == 2
    assert metadata["parser_task_id"] == "task-1"
    assert metadata["block_types"] == "text"

    table_text, table_metadata = chunks[1]
    assert len(table_text) > 100
    assert "Table 1" in table_text
    assert table_metadata["block_types"] == "table"
    assert table_metadata["page_start"] == 2


def test_local_document_uses_legacy_strategy_with_parser_metadata():
    document = ParsedDocument(
        source_name="notes.txt",
        markdown="abcdefghij",
        provenance=ParserProvenance(parser_name="local", parser_version="1"),
    )

    chunks = chunk_parsed_document(
        document,
        chunk_size=4,
        chunk_overlap=0,
        strategy="fixed",
    )

    assert [text for text, _ in chunks] == ["abcd", "efgh", "ij"]
    assert all(metadata["parser_name"] == "local" for _, metadata in chunks)
    assert all(metadata["strategy"] == "fixed" for _, metadata in chunks)


def test_chunk_parsed_document_rejects_wrong_input_type():
    with pytest.raises(TypeError, match="ParsedDocument"):
        chunk_parsed_document("not-a-document")


@pytest.mark.parametrize("parser_name", ["local", "mineru"])
@pytest.mark.parametrize("strategy", ["fixed", "recursive", "markdown", "parent_child", "legal"])
@pytest.mark.parametrize("legal_text", [False, True])
def test_parsed_documents_follow_configured_strategy(monkeypatch, parser_name, strategy, legal_text):
    monkeypatch.setattr(cfg, "CHUNK_STRATEGY", strategy)
    monkeypatch.setattr(cfg, "CHUNK_OVERLAP", 10)
    body = (
        "\n".join(f"第{i}条 条款内容。" for i in range(1, 25))
        if legal_text else "Ordinary paragraph.\n\n" * 24
    )
    text = "# Report\n\n" + body
    document = ParsedDocument(
        source_name="report.md",
        markdown=text,
        provenance=ParserProvenance(parser_name=parser_name),
        blocks=(ParsedBlock(block_id="0", type=ParsedBlockType.TEXT, text=text),),
    )

    actual = chunk_parsed_document(document, chunk_size=100, chunk_overlap=10)
    expected = parse_and_chunk(text.encode(), "report.md", chunk_size=100, chunk_overlap=10)

    assert actual
    assert [value for value, _ in actual] == [value for value, _ in expected]
    assert all(meta["strategy"] == strategy for _, meta in actual)
    assert all(meta["parser_name"] == parser_name for _, meta in actual)


@pytest.mark.parametrize("parser_name", ["local", "mineru"])
def test_explicit_strategy_overrides_config(monkeypatch, parser_name):
    monkeypatch.setattr(cfg, "CHUNK_STRATEGY", "legal")
    document = ParsedDocument(
        source_name="notes.txt",
        markdown="abcdefghij",
        provenance=ParserProvenance(parser_name=parser_name),
    )
    chunks = chunk_parsed_document(document, strategy="fixed", chunk_size=4, chunk_overlap=0)
    assert [text for text, _ in chunks] == ["abcd", "efgh", "ij"]
    assert all(meta["strategy"] == "fixed" for _, meta in chunks)


@pytest.mark.parametrize("parser_name", ["local", "mineru"])
def test_structured_without_blocks_uses_text_structure(monkeypatch, parser_name):
    monkeypatch.setattr(cfg, "CHUNK_STRATEGY", "structured")
    text = "# Report\n" + "abcdefghij" * 30
    document = ParsedDocument(
        source_name="report.md",
        markdown=text,
        provenance=ParserProvenance(parser_name=parser_name),
    )
    chunks = chunk_parsed_document(document, chunk_size=100, chunk_overlap=10)
    assert [value for value, _ in chunks] == split_structured(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(meta["strategy"] == "structured" for _, meta in chunks)
    assert all(meta["section_path"] == "Report" for _, meta in chunks)
