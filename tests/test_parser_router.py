from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.core.config import Settings
from app.rag.parsers import (
    DocumentParser,
    DocumentParserError,
    ParsedBlock,
    ParsedBlockType,
    ParsedDocument,
    ParserProvenance,
    ParserRouter,
    TransientDocumentParserError,
    UnsupportedDocumentError,
)


@dataclass
class StubParser(DocumentParser):
    parser_name: str
    supported_extensions: frozenset[str]
    error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    async def parse(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type: str | None = None,
    ) -> ParsedDocument:
        self.calls.append(filename)
        if self.error:
            raise self.error
        return ParsedDocument(
            source_name=filename,
            markdown=f"parsed by {self.parser_name}",
            provenance=ParserProvenance(parser_name=self.parser_name),
            blocks=(
                ParsedBlock(
                    block_id="0",
                    type=ParsedBlockType.TEXT,
                    text=f"parsed by {self.parser_name}",
                ),
            ),
        )


def _settings(*, enabled: bool, fallback: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        MINERU_ENABLED=enabled,
        MINERU_FALLBACK_TO_LOCAL=fallback,
    )


def _router(
    *,
    enabled: bool,
    fallback: bool = True,
    mineru_error: Exception | None = None,
) -> tuple[ParserRouter, StubParser, StubParser]:
    local = StubParser("local", frozenset({".txt", ".md", ".pdf"}))
    mineru = StubParser(
        "mineru",
        frozenset({".pdf", ".docx", ".pptx"}),
        error=mineru_error,
    )
    router = ParserRouter(
        settings=_settings(enabled=enabled, fallback=fallback),
        local_parser=local,
        mineru_parser=mineru,
    )
    return router, local, mineru


@pytest.mark.asyncio
async def test_enabled_router_prefers_mineru_but_keeps_text_local():
    router, local, mineru = _router(enabled=True)

    pdf = await router.parse(b"pdf", "report.pdf")
    text = await router.parse(b"text", "notes.txt")

    assert pdf.provenance.parser_name == "mineru"
    assert text.provenance.parser_name == "local"
    assert mineru.calls == ["report.pdf"]
    assert local.calls == ["notes.txt"]
    assert router.supports("slides.pptx") is True


@pytest.mark.asyncio
async def test_disabled_router_uses_local_and_rejects_mineru_only_format():
    router, local, mineru = _router(enabled=False)

    pdf = await router.parse(b"pdf", "report.pdf")

    assert pdf.provenance.parser_name == "local"
    assert local.calls == ["report.pdf"]
    assert mineru.calls == []
    assert router.supports("slides.pptx") is False
    with pytest.raises(UnsupportedDocumentError, match="MINERU_ENABLED is false"):
        await router.parse(b"pptx", "slides.pptx")


@pytest.mark.asyncio
async def test_transient_mineru_failure_falls_back_with_audit_metadata():
    router, local, mineru = _router(
        enabled=True,
        mineru_error=TransientDocumentParserError("service offline"),
    )

    document = await router.parse(b"pdf", "report.pdf")

    assert mineru.calls == ["report.pdf"]
    assert local.calls == ["report.pdf"]
    assert document.provenance.parser_name == "local"
    assert document.metadata["parser_fallback_from"] == "mineru"
    assert document.metadata["parser_fallback_reason"] == "TransientDocumentParserError"
    assert "service offline" in document.warnings[0]


@pytest.mark.asyncio
async def test_fallback_can_be_disabled():
    router, local, _ = _router(
        enabled=True,
        fallback=False,
        mineru_error=TransientDocumentParserError("service offline"),
    )

    with pytest.raises(TransientDocumentParserError, match="service offline"):
        await router.parse(b"pdf", "report.pdf")
    assert local.calls == []


@pytest.mark.asyncio
async def test_non_transient_mineru_failure_is_not_hidden_by_fallback():
    router, local, _ = _router(
        enabled=True,
        mineru_error=DocumentParserError("invalid document"),
    )

    with pytest.raises(DocumentParserError, match="invalid document"):
        await router.parse(b"pdf", "report.pdf")
    assert local.calls == []


@pytest.mark.asyncio
async def test_user_can_force_either_supported_parser():
    router, local, mineru = _router(enabled=True)

    local_document = await router.parse(
        b"pdf", "report.pdf", preferred_parser="local"
    )
    mineru_document = await router.parse(
        b"pdf", "report.pdf", preferred_parser="mineru"
    )

    assert local_document.provenance.parser_name == "local"
    assert mineru_document.provenance.parser_name == "mineru"
    assert local.calls == ["report.pdf"]
    assert mineru.calls == ["report.pdf"]


@pytest.mark.asyncio
async def test_forced_mineru_does_not_silently_fallback():
    router, local, _ = _router(
        enabled=True,
        mineru_error=TransientDocumentParserError("service offline"),
    )

    with pytest.raises(TransientDocumentParserError, match="service offline"):
        await router.parse(
            b"pdf", "report.pdf", preferred_parser="mineru"
        )
    assert local.calls == []


def test_forced_parser_validates_choice_and_file_support():
    router, _, _ = _router(enabled=True)

    with pytest.raises(ValueError, match="auto, mineru, local"):
        router.select_parser("report.pdf", preferred_parser="unknown")
    with pytest.raises(UnsupportedDocumentError, match="does not support"):
        router.select_parser("notes.txt", preferred_parser="mineru")
