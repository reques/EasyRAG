"""Adapter for EasyRAG's existing in-process text extractors."""
from __future__ import annotations

import asyncio
import hashlib
import io
import mimetypes
from pathlib import PurePath

from app.rag.chunker import extract_text
from app.rag.parsers.base import DocumentParser, DocumentParserError, EmptyDocumentError
from app.rag.parsers.models import (
    ParsedBlock,
    ParsedBlockType,
    ParsedContentFormat,
    ParsedDocument,
    ParserProvenance,
)


class LocalParser(DocumentParser):
    """Normalize output from the legacy pypdf/docx/OCR extractors."""

    parser_name = "local"
    parser_version = "1"
    supported_extensions = frozenset(
        {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )

    async def parse(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type: str | None = None,
    ) -> ParsedDocument:
        self.validate_input(raw, filename)
        try:
            text = await asyncio.to_thread(extract_text, raw, filename)
        except DocumentParserError:
            raise
        except Exception as exc:
            raise DocumentParserError(
                f"Local parsing failed for '{filename}': {exc}"
            ) from exc
        if not text.strip():
            raise EmptyDocumentError(
                f"Local parser produced no text for '{filename}'"
            )

        extension = PurePath(filename).suffix.lower()
        page_count = await asyncio.to_thread(_local_page_count, raw, extension)
        content_format = (
            ParsedContentFormat.MARKDOWN
            if extension == ".md"
            else ParsedContentFormat.PLAIN_TEXT
        )
        warnings: tuple[str, ...] = ()
        if extension == ".pdf":
            warnings = (
                "Local PDF parsing does not preserve per-block page coordinates.",
            )

        return ParsedDocument(
            source_name=filename,
            source_mime_type=(
                content_type
                or mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            ),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            markdown=text,
            blocks=(
                ParsedBlock(
                    block_id="local-0",
                    type=ParsedBlockType.TEXT,
                    text=text,
                    content_format=content_format,
                    page_index=0 if page_count == 1 else None,
                    source_type="local_text",
                ),
            ),
            page_count=page_count,
            provenance=ParserProvenance(
                parser_name=self.parser_name,
                parser_version=self.parser_version,
            ),
            warnings=warnings,
        )


def _local_page_count(raw: bytes, extension: str) -> int | None:
    if extension in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return 1
    if extension != ".pdf":
        return None
    try:
        import pypdf
    except ImportError:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ImportError:
            return None
    return len(pypdf.PdfReader(io.BytesIO(raw)).pages)
