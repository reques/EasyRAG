"""Configuration-driven selection between document parser implementations."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.parsers.base import (
    DocumentParser,
    TransientDocumentParserError,
    UnsupportedDocumentError,
)
from app.rag.parsers.local_parser import LocalParser
from app.rag.parsers.mineru_parser import MinerUParser
from app.rag.parsers.models import ParsedDocument

logger = get_logger(__name__)


class ParserRouter(DocumentParser):
    """Choose a parser without exposing parser-specific behavior to callers."""

    parser_name = "router"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        local_parser: DocumentParser | None = None,
        mineru_parser: DocumentParser | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.local_parser = local_parser or LocalParser()
        self.mineru_parser = mineru_parser or MinerUParser()
        enabled_extensions = set(self.local_parser.supported_extensions)
        if self.settings.MINERU_ENABLED:
            enabled_extensions.update(self.mineru_parser.supported_extensions)
        self.supported_extensions = frozenset(enabled_extensions)

    def select_parser(
        self,
        filename: str,
        *,
        preferred_parser: str = "auto",
    ) -> DocumentParser:
        """Return the requested parser, or choose one when ``auto`` is used."""
        preference = (preferred_parser or "auto").strip().lower()
        if preference not in {"auto", "mineru", "local"}:
            raise ValueError(
                "Parser must be one of: auto, mineru, local"
            )

        mineru_supports = self.mineru_parser.supports(filename)
        local_supports = self.local_parser.supports(filename)

        if preference == "mineru":
            if not self.settings.MINERU_ENABLED:
                raise UnsupportedDocumentError(
                    "MinerU was selected, but MINERU_ENABLED is false"
                )
            if not mineru_supports:
                raise UnsupportedDocumentError(
                    f"MinerU does not support '{filename}'"
                )
            return self.mineru_parser

        if preference == "local":
            if not local_supports:
                raise UnsupportedDocumentError(
                    f"Local parser does not support '{filename}'"
                )
            return self.local_parser

        if self.settings.MINERU_ENABLED and mineru_supports:
            return self.mineru_parser
        if local_supports:
            return self.local_parser
        if mineru_supports:
            raise UnsupportedDocumentError(
                f"'{filename}' requires MinerU, but MINERU_ENABLED is false"
            )
        raise UnsupportedDocumentError(
            f"No configured parser supports '{filename}'"
        )

    async def parse(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type: str | None = None,
        preferred_parser: str = "auto",
    ) -> ParsedDocument:
        preference = (preferred_parser or "auto").strip().lower()
        parser = self.select_parser(filename, preferred_parser=preference)
        try:
            return await parser.parse(raw, filename, content_type=content_type)
        except TransientDocumentParserError as exc:
            if not self._can_fallback(parser, filename, preference):
                raise
            logger.warning(
                "[parser-router] MinerU unavailable for '%s'; falling back to local: %s",
                filename,
                exc,
            )
            document = await self.local_parser.parse(
                raw,
                filename,
                content_type=content_type,
            )
            warning = f"MinerU was unavailable; local parser fallback was used: {exc}"
            metadata = {
                **dict(document.metadata),
                "parser_fallback_from": self.mineru_parser.parser_name,
                "parser_fallback_reason": type(exc).__name__,
            }
            return replace(
                document,
                warnings=(warning, *document.warnings),
                metadata=metadata,
            )

    def _can_fallback(
        self,
        selected: DocumentParser,
        filename: str,
        preference: str,
    ) -> bool:
        return bool(
            preference == "auto"
            and selected is self.mineru_parser
            and self.settings.MINERU_FALLBACK_TO_LOCAL
            and self.local_parser.supports(filename)
        )


@lru_cache(maxsize=1)
def get_parser_router() -> ParserRouter:
    """Return the process-wide parser router."""
    return ParserRouter()
