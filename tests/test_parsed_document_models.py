from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.rag.parsers import (
    ParsedBlock,
    ParsedBlockType,
    ParsedBoundingBox,
    ParsedContentFormat,
    ParsedDocument,
    ParsedImage,
    ParserProvenance,
)


def test_parsed_document_preserves_structured_content_and_provenance():
    image = ParsedImage(
        path="images/chart.jpg",
        data=b"jpeg-data",
        mime_type="image/jpeg",
        page_index=1,
    )
    document = ParsedDocument(
        source_name="report.pdf",
        source_mime_type="application/pdf",
        source_sha256="a" * 64,
        markdown="# Report\n\nBody",
        provenance=ParserProvenance(
            parser_name="mineru",
            parser_version="3.4.4",
            task_id="task-1",
            backend="pipeline",
            parse_method="auto",
            languages=("ch",),
        ),
        page_count=2,
        blocks=(
            ParsedBlock(
                block_id="block-0",
                type=ParsedBlockType.TITLE,
                text="Report",
                page_index=0,
                heading_level=1,
                bbox=ParsedBoundingBox(10, 20, 100, 40),
                section_path=("Report",),
                source_type="text",
            ),
            ParsedBlock(
                block_id="block-1",
                type=ParsedBlockType.CHART,
                image_path="images/chart.jpg",
                captions=("Figure 1",),
                page_index=1,
            ),
        ),
        images=(image,),
    )

    assert document.text.startswith("# Report")
    assert document.page_numbers == (1, 2)
    assert document.image_map["images/chart.jpg"].sha256 == image.sha256
    assert image.size_bytes == 9
    assert document.blocks[0].bbox.as_tuple() == (10, 20, 100, 40)


def test_document_text_falls_back_to_ordered_blocks():
    document = ParsedDocument(
        source_name="notes.txt",
        markdown="",
        provenance=ParserProvenance(parser_name="local"),
        blocks=(
            ParsedBlock(block_id="0", type=ParsedBlockType.TEXT, text="First"),
            ParsedBlock(
                block_id="1",
                type=ParsedBlockType.TABLE,
                text="<table></table>",
                content_format=ParsedContentFormat.HTML,
            ),
        ),
    )

    assert document.text == "First\n\n<table></table>"


@pytest.mark.parametrize(
    "path",
    ("../secret.jpg", "/absolute.jpg", "images\\windows.jpg", ""),
)
def test_image_artifact_path_cannot_escape_document(path):
    with pytest.raises(ValueError, match="Artifact path"):
        ParsedImage(path=path, data=b"data", mime_type="image/jpeg")


def test_page_index_must_fit_declared_page_count():
    with pytest.raises(ValueError, match="exceeds"):
        ParsedDocument(
            source_name="report.pdf",
            markdown="body",
            provenance=ParserProvenance(parser_name="mineru"),
            page_count=1,
            blocks=(
                ParsedBlock(
                    block_id="0",
                    type=ParsedBlockType.TEXT,
                    text="body",
                    page_index=1,
                ),
            ),
        )


def test_empty_document_and_invalid_checksum_are_rejected():
    provenance = ParserProvenance(parser_name="local")
    with pytest.raises(ValueError, match="must contain"):
        ParsedDocument(source_name="empty.txt", markdown="", provenance=provenance)
    with pytest.raises(ValueError, match="source_sha256"):
        ParsedDocument(
            source_name="notes.txt",
            markdown="body",
            provenance=provenance,
            source_sha256="not-a-checksum",
        )


def test_models_are_immutable():
    document = ParsedDocument(
        source_name="notes.txt",
        markdown="body",
        provenance=ParserProvenance(parser_name="local"),
    )

    with pytest.raises(FrozenInstanceError):
        document.markdown = "changed"
