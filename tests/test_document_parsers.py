from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from app.rag.parsers import (
    LocalParser,
    MinerUHealth,
    MinerUParseOptions,
    MinerUParser,
    MinerUResponseError,
    MinerUSubmission,
    MinerUTask,
    MinerUTaskStatus,
    ParsedBlockType,
    ParsedContentFormat,
    ParserOutputError,
    TransientDocumentParserError,
    UnsupportedDocumentError,
)


class FakeMinerUClient:
    def __init__(self, result_zip: bytes, health_error: Exception | None = None):
        self.result_zip = result_zip
        self.health_error = health_error
        self.options = MinerUParseOptions(backend="pipeline", languages=("ch",))
        self.submitted: tuple[bytes, str, str | None] | None = None
        self.closed = False

    def default_options(self) -> MinerUParseOptions:
        return self.options

    async def health(self) -> MinerUHealth:
        if self.health_error:
            raise self.health_error
        return MinerUHealth(status="healthy", version="3.4.4", protocol_version=2)

    async def submit_document(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type: str | None = None,
        options: MinerUParseOptions | None = None,
    ) -> MinerUSubmission:
        assert options == self.options
        self.submitted = raw, filename, content_type
        return MinerUSubmission(
            task_id="task-123",
            status=MinerUTaskStatus.PENDING,
            backend="pipeline",
            file_names=(filename,),
        )

    async def wait_for_completion(self, task_id: str) -> MinerUTask:
        assert task_id == "task-123"
        return MinerUTask(task_id=task_id, status=MinerUTaskStatus.COMPLETED)

    async def download_result(self, task_id: str, destination: Path) -> Path:
        assert task_id == "task-123"
        destination.write_bytes(self.result_zip)
        return destination

    async def aclose(self) -> None:
        self.closed = True


def _mineru_zip(*, unsafe_entry: str | None = None) -> bytes:
    content_list = [
        {
            "type": "text",
            "text": "Report",
            "text_level": 1,
            "page_idx": 0,
            "bbox": [10, 20, 100, 40],
        },
        {
            "type": "text",
            "text": "Introduction",
            "text_level": 2,
            "page_idx": 0,
        },
        {"type": "text", "text": "Body", "page_idx": 0},
        {
            "type": "table",
            "table_body": "<table><tr><td>1</td></tr></table>",
            "table_caption": ["Table 1"],
            "page_idx": 1,
        },
        {
            "type": "equation",
            "text": "x^2",
            "text_format": "latex",
            "page_idx": 1,
        },
        {
            "type": "chart",
            "img_path": "images/chart.jpg",
            "chart_caption": ["Figure 1"],
            "page_idx": 1,
            "bbox": [20, 50, 200, 250],
        },
    ]
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("report/auto/report.md", "# Report\n\nBody")
        archive.writestr(
            "report/auto/report_content_list.json",
            json.dumps(content_list),
        )
        archive.writestr(
            "report/auto/report_content_list_v2.json",
            json.dumps({"ignored": True}),
        )
        archive.writestr("report/auto/images/chart.jpg", b"jpeg-image")
        if unsafe_entry:
            archive.writestr(unsafe_entry, b"unsafe")
    return output.getvalue()


@pytest.mark.asyncio
async def test_local_parser_implements_unified_contract():
    parser = LocalParser()
    document = await parser.parse(
        "第一段\n第二段".encode(),
        "notes.txt",
        content_type="text/plain",
    )

    assert parser.supports("notes.TXT")
    assert document.source_name == "notes.txt"
    assert document.text == "第一段\n第二段"
    assert document.provenance.parser_name == "local"
    assert document.blocks[0].type is ParsedBlockType.TEXT
    assert document.source_sha256 is not None


@pytest.mark.asyncio
async def test_local_parser_rejects_unsupported_file_type():
    with pytest.raises(UnsupportedDocumentError, match=".exe"):
        await LocalParser().parse(b"content", "program.exe")


@pytest.mark.asyncio
async def test_mineru_parser_normalizes_structured_archive():
    client = FakeMinerUClient(_mineru_zip())
    parser = MinerUParser(client=client)

    document = await parser.parse(
        b"%PDF-test",
        "report.pdf",
        content_type="application/pdf",
    )

    assert client.submitted == (b"%PDF-test", "report.pdf", "application/pdf")
    assert client.closed is False
    assert document.provenance.parser_name == "mineru"
    assert document.provenance.parser_version == "3.4.4"
    assert document.provenance.task_id == "task-123"
    assert document.page_count == 2
    assert document.page_numbers == (1, 2)
    assert len(document.blocks) == 6
    assert len(document.images) == 1
    assert document.images[0].path == "images/chart.jpg"
    assert document.images[0].page_index == 1

    title, subtitle, body, table, equation, chart = document.blocks
    assert title.type is ParsedBlockType.TITLE
    assert title.section_path == ("Report",)
    assert subtitle.section_path == ("Report", "Introduction")
    assert body.section_path == ("Report", "Introduction")
    assert table.type is ParsedBlockType.TABLE
    assert table.content_format is ParsedContentFormat.HTML
    assert equation.content_format is ParsedContentFormat.LATEX
    assert chart.image_path == "images/chart.jpg"
    assert chart.captions == ("Figure 1",)


@pytest.mark.asyncio
async def test_mineru_parser_rejects_unsafe_zip_entries():
    client = FakeMinerUClient(_mineru_zip(unsafe_entry="../escape.txt"))
    parser = MinerUParser(client=client)

    with pytest.raises(ParserOutputError, match="Unsafe MinerU ZIP entry"):
        await parser.parse(b"%PDF-test", "report.pdf")


@pytest.mark.asyncio
async def test_mineru_service_5xx_is_classified_as_transient():
    client = FakeMinerUClient(
        _mineru_zip(),
        health_error=MinerUResponseError(503, "starting"),
    )

    with pytest.raises(TransientDocumentParserError, match="503"):
        await MinerUParser(client=client).parse(b"%PDF-test", "report.pdf")
