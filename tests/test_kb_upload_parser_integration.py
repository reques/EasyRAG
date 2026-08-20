from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from app.api.kb_routes import upload_file
from app.rag.parsers import (
    ParsedBlock,
    ParsedBlockType,
    ParsedDocument,
    ParserProvenance,
    TransientDocumentParserError,
)
from backend.server.routers.knowledge_router import FileResponse
from backend.storage.postgres.models_knowledge import KnowledgeFile


class FakeParserRouter:
    supported_extensions = frozenset({".pdf", ".txt"})

    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.call = None

    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_extensions

    async def parse(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type=None,
        preferred_parser="auto",
    ):
        self.call = raw, filename, content_type, preferred_parser
        if self.error:
            raise self.error
        return ParsedDocument(
            source_name=filename,
            markdown="# Report\n\nParsed body",
            page_count=1,
            provenance=ParserProvenance(
                parser_name="mineru",
                parser_version="3.4.4",
                task_id="task-1",
            ),
            blocks=(
                ParsedBlock(
                    block_id="0",
                    type=ParsedBlockType.TEXT,
                    text="Parsed body",
                    section_path=("Report",),
                    page_index=0,
                ),
            ),
        )


class CapturingRetriever:
    def __init__(self):
        self.texts = None
        self.metadatas = None

    def add_documents(self, texts, metadatas):
        self.texts = texts
        self.metadatas = metadatas
        return len(texts)


def _upload() -> UploadFile:
    return UploadFile(
        BytesIO(b"%PDF-test"),
        filename="report.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )


@pytest.mark.asyncio
async def test_legacy_kb_upload_uses_parser_router_and_parsed_chunker(monkeypatch):
    parser_router = FakeParserRouter()
    retriever = CapturingRetriever()
    monkeypatch.setattr(
        "app.rag.parsers.get_parser_router",
        lambda: parser_router,
    )
    monkeypatch.setattr(
        "app.rag.retriever.get_retriever",
        lambda: retriever,
    )

    response = await upload_file(
        file=_upload(), chunk_size=0, chunk_overlap=0, parser="mineru"
    )

    assert response.indexed == 1
    assert "with mineru" in response.message
    assert parser_router.call == (
        b"%PDF-test", "report.pdf", "application/pdf", "mineru"
    )
    assert retriever.texts == ["[Report]\nParsed body"]
    metadata = retriever.metadatas[0]
    assert metadata["parser_name"] == "mineru"
    assert metadata["parser_task_id"] == "task-1"
    assert metadata["page_start"] == 1
    assert metadata["page_end"] == 1


@pytest.mark.asyncio
async def test_upload_reports_transient_parser_failure_as_503(monkeypatch):
    parser_router = FakeParserRouter(
        error=TransientDocumentParserError("MinerU unavailable")
    )
    monkeypatch.setattr(
        "app.rag.parsers.get_parser_router",
        lambda: parser_router,
    )

    with pytest.raises(HTTPException) as raised:
        await upload_file(
            file=_upload(), chunk_size=0, chunk_overlap=0, parser="mineru"
        )

    assert raised.value.status_code == 503
    assert "MinerU unavailable" in raised.value.detail


def test_primary_background_ingestion_no_longer_calls_legacy_parse_and_chunk():
    source = (
        Path(__file__).parents[1]
        / "backend/server/routers/knowledge_router.py"
    ).read_text(encoding="utf-8")
    ingestion = source.split("async def _run_ingestion", 1)[1]
    ingestion = ingestion.split("@router.get", 1)[0]

    assert "preferred_parser=preferred_parser" in ingestion
    assert "parsed_document = await parser_router.parse" in ingestion
    assert "chunk_parsed_document(parsed_document" in ingestion
    assert "parse_and_chunk" not in ingestion
    assert "f.text_content = parsed_document.text" in ingestion
    assert "f.parser_name = provenance.parser_name" in ingestion
    assert "f.parser_task_id = provenance.task_id" in ingestion
    assert "batch_size =" in ingestion
    assert "await asyncio.to_thread" in ingestion
    assert "stage=\"indexing\"" in ingestion
    assert "progress_callback=report_graph_progress" in ingestion


def test_knowledge_file_persists_parser_traceability_fields():
    columns = KnowledgeFile.__table__.columns

    assert columns["parser_name"].type.length == 32
    assert columns["parser_version"].type.length == 64
    assert columns["parser_task_id"].type.length == 128
    assert "parser_backend" in columns
    assert "parse_method" in columns
    assert "parser_warnings" in columns
    assert "processing_stage" in columns
    assert "progress_message" in columns
    assert "progress_current" in columns
    assert "progress_total" in columns


def test_file_response_exposes_parser_traceability_fields():
    fields = FileResponse.model_fields

    assert "parser_name" in fields
    assert "parser_version" in fields
    assert "parser_task_id" in fields
    assert "parser_backend" in fields
    assert "parse_method" in fields
    assert "parser_warnings" in fields
    assert "processing_stage" in fields
    assert "progress_message" in fields
    assert "progress_current" in fields
    assert "progress_total" in fields
