"""Parser-neutral document models used by every ingestion backend.

The types in this module are intentionally independent of MinerU. A local
parser, MinerU, or a future cloud parser must all normalize their output into
``ParsedDocument`` before chunking and indexing begin.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping


class ParsedBlockType(str, Enum):
    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    EQUATION = "equation"
    IMAGE = "image"
    CHART = "chart"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


class ParsedContentFormat(str, Enum):
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"


@dataclass(frozen=True)
class ParsedBoundingBox:
    """A block rectangle in the parser's source coordinate system."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Bounding-box coordinates must be finite")
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding-box end coordinates must not precede start")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass(frozen=True)
class ParsedBlock:
    """One normalized, ordered structural block from a document."""

    block_id: str
    type: ParsedBlockType
    text: str = ""
    content_format: ParsedContentFormat = ParsedContentFormat.PLAIN_TEXT
    page_index: int | None = None
    bbox: ParsedBoundingBox | None = None
    heading_level: int | None = None
    section_path: tuple[str, ...] = ()
    image_path: str | None = None
    captions: tuple[str, ...] = ()
    footnotes: tuple[str, ...] = ()
    source_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("Parsed block_id must not be empty")
        if self.page_index is not None and self.page_index < 0:
            raise ValueError("Parsed block page_index must be non-negative")
        if self.heading_level is not None and not 1 <= self.heading_level <= 6:
            raise ValueError("Parsed block heading_level must be between 1 and 6")
        if self.image_path is not None:
            _validate_relative_artifact_path(self.image_path)

    @property
    def has_content(self) -> bool:
        return bool(
            self.text.strip()
            or self.image_path
            or any(item.strip() for item in self.captions)
        )


@dataclass(frozen=True)
class ParsedImage:
    """An extracted image artifact referenced by one or more parsed blocks."""

    path: str
    data: bytes = field(repr=False)
    mime_type: str = "application/octet-stream"
    page_index: int | None = None
    bbox: ParsedBoundingBox | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _validate_relative_artifact_path(self.path)
        if not isinstance(self.data, bytes):
            raise TypeError("Parsed image data must be bytes")
        if not self.data:
            raise ValueError("Parsed image data must not be empty")
        if not self.mime_type.strip():
            raise ValueError("Parsed image mime_type must not be empty")
        if self.page_index is not None and self.page_index < 0:
            raise ValueError("Parsed image page_index must be non-negative")

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class ParserProvenance:
    """Identifies how a parsed document was produced."""

    parser_name: str
    parser_version: str | None = None
    task_id: str | None = None
    backend: str | None = None
    parse_method: str | None = None
    languages: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.parser_name.strip():
            raise ValueError("Parser name must not be empty")


@dataclass(frozen=True)
class ParsedDocument:
    """Canonical output of document parsing, before chunking and indexing.

    Page indexes are zero-based throughout the backend. User-facing citations
    can convert them to one-based page numbers at the presentation boundary.
    """

    source_name: str
    markdown: str
    provenance: ParserProvenance
    blocks: tuple[ParsedBlock, ...] = ()
    images: tuple[ParsedImage, ...] = ()
    page_count: int | None = None
    source_mime_type: str | None = None
    source_sha256: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.source_name.strip() or any(
            separator in self.source_name for separator in ("/", "\\")
        ):
            raise ValueError("Parsed document source_name must be a non-empty basename")
        if self.page_count is not None and self.page_count < 0:
            raise ValueError("Parsed document page_count must be non-negative")
        if self.source_sha256 is not None and not _is_sha256(self.source_sha256):
            raise ValueError("Parsed document source_sha256 must be 64 hexadecimal characters")
        if not self.markdown.strip() and not self.blocks and not self.images:
            raise ValueError("Parsed document must contain markdown, blocks, or images")

        if self.page_count is not None:
            page_indexes = [
                item.page_index
                for item in (*self.blocks, *self.images)
                if item.page_index is not None
            ]
            if any(page_index >= self.page_count for page_index in page_indexes):
                raise ValueError("Parsed item page_index exceeds document page_count")

    @property
    def text(self) -> str:
        """Best textual representation for parser-agnostic consumers."""
        if self.markdown.strip():
            return self.markdown
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())

    @property
    def image_map(self) -> Mapping[str, ParsedImage]:
        return {image.path: image for image in self.images}

    @property
    def page_numbers(self) -> tuple[int, ...]:
        """Sorted one-based pages represented by structural blocks."""
        return tuple(
            sorted(
                {
                    block.page_index + 1
                    for block in self.blocks
                    if block.page_index is not None
                }
            )
        )


def _validate_relative_artifact_path(path: str) -> None:
    if not path.strip() or "\\" in path:
        raise ValueError("Artifact path must be a non-empty POSIX relative path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise ValueError("Artifact path must not escape the parsed document")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)
