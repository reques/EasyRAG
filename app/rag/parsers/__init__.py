"""Document parser integrations."""

from app.rag.parsers.base import (
    DocumentParser,
    DocumentParserError,
    EmptyDocumentError,
    ParserOutputError,
    TransientDocumentParserError,
    UnsupportedDocumentError,
)
from app.rag.parsers.local_parser import LocalParser
from app.rag.parsers.models import (
    ParsedBlock,
    ParsedBlockType,
    ParsedBoundingBox,
    ParsedContentFormat,
    ParsedDocument,
    ParsedImage,
    ParserProvenance,
)
from app.rag.parsers.mineru_parser import MinerUParser
from app.rag.parsers.router import ParserRouter, get_parser_router
from app.rag.parsers.mineru_client import (
    MinerUClient,
    MinerUConnectionError,
    MinerUError,
    MinerUHealth,
    MinerUParseOptions,
    MinerUProtocolError,
    MinerUResponseError,
    MinerUSubmission,
    MinerUTask,
    MinerUTaskFailedError,
    MinerUTaskStatus,
    MinerUTaskTimeoutError,
)

__all__ = [
    "DocumentParser",
    "DocumentParserError",
    "EmptyDocumentError",
    "LocalParser",
    "MinerUClient",
    "MinerUConnectionError",
    "MinerUError",
    "MinerUHealth",
    "MinerUParseOptions",
    "MinerUProtocolError",
    "MinerUResponseError",
    "MinerUSubmission",
    "MinerUTask",
    "MinerUTaskFailedError",
    "MinerUTaskStatus",
    "MinerUTaskTimeoutError",
    "MinerUParser",
    "ParserOutputError",
    "ParserRouter",
    "ParsedBlock",
    "ParsedBlockType",
    "ParsedBoundingBox",
    "ParsedContentFormat",
    "ParsedDocument",
    "ParsedImage",
    "ParserProvenance",
    "TransientDocumentParserError",
    "UnsupportedDocumentError",
    "get_parser_router",
]
