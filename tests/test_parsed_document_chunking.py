from __future__ import annotations

import pytest

from app.rag.chunker import chunk_parsed_document
from app.rag.parsers import (
    ParsedBlock,
    ParsedBlockType,
    ParsedContentFormat,
    ParsedDocument,
    ParserProvenance,
)


def test_structured_chunking_preserves_pages_sections_and_atomic_tables():
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

    chunks = chunk_parsed_document(document, chunk_size=100)

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
