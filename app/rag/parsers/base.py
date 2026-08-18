"""Common interface and errors for document parsers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import PurePath

from app.rag.parsers.models import ParsedDocument


class DocumentParserError(RuntimeError):
    """Base error raised at the parser boundary."""


class UnsupportedDocumentError(DocumentParserError):
    """The selected parser cannot handle the input file type."""


class EmptyDocumentError(DocumentParserError):
    """The input or its parsed textual result is empty."""


class ParserOutputError(DocumentParserError):
    """A parser returned incomplete, malformed, or unsafe output."""


class TransientDocumentParserError(DocumentParserError):
    """Parsing may succeed if retried after an external service recovers."""


class DocumentParser(ABC):
    """Parser-neutral asynchronous document parsing contract."""

    parser_name: str
    supported_extensions: frozenset[str]

    def supports(self, filename: str) -> bool:
        return _file_extension(filename) in self.supported_extensions

    @abstractmethod
    async def parse(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type: str | None = None,
    ) -> ParsedDocument:
        """Normalize one source document into a ``ParsedDocument``."""

    def validate_input(self, raw: bytes, filename: str) -> None:
        if not isinstance(raw, bytes):
            raise TypeError("Document parser input must be bytes")
        if not raw:
            raise EmptyDocumentError("Cannot parse an empty document")
        if (
            not filename.strip()
            or any(separator in filename for separator in ("/", "\\"))
            or PurePath(filename).name != filename
        ):
            raise ValueError("Document filename must be a non-empty basename")
        if not self.supports(filename):
            extension = _file_extension(filename) or "<none>"
            raise UnsupportedDocumentError(
                f"Parser '{self.parser_name}' does not support extension '{extension}'"
            )


def _file_extension(filename: str) -> str:
    suffix = PurePath(filename).suffix
    return suffix.lower()
