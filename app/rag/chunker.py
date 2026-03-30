"""Document chunker – parses uploaded files and splits into text chunks.

Supported file types:
    .txt / .md  – plain text
    .pdf        – via pypdf (``pip install pypdf``)
    .docx       – via python-docx (``pip install python-docx``)

Chunking strategy: fixed-size sliding window with overlap.
"""
from __future__ import annotations

import io
from typing import List, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

# Chunk = (text_content, metadata_dict)
Chunk = Tuple[str, dict]


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_txt(raw: bytes, filename: str) -> str:
    """Decode plain text / markdown files."""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(raw: bytes, filename: str) -> str:
    """Extract text from PDF using pypdf."""
    try:
        import pypdf
    except ImportError:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pypdf is not installed. Run: pip install pypdf"
            ) from exc
    reader = pypdf.PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _extract_docx(raw: bytes, filename: str) -> str:
    """Extract text from .docx using python-docx."""
    try:
        from docx import Document  # python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is not installed. Run: pip install python-docx"
        ) from exc
    doc = Document(io.BytesIO(raw))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


_EXTRACTORS = {
    ".txt":  _extract_txt,
    ".md":   _extract_txt,
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
}


def extract_text(raw: bytes, filename: str) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(_EXTRACTORS)}"
        )
    logger.info("[chunker] extracting '%s' (ext=%s, size=%d bytes)", filename, ext, len(raw))
    return extractor(raw, filename)


# ── Sliding-window chunker ────────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """Split *text* into overlapping fixed-size chunks (character-level)."""
    size = chunk_size or cfg.CHUNK_SIZE
    overlap = chunk_overlap or cfg.CHUNK_OVERLAP
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += size - overlap
    return chunks


# ── High-level entry point ────────────────────────────────────────────────────

def parse_and_chunk(
    raw: bytes,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """Extract text from *raw* bytes and return a list of (chunk_text, metadata) tuples.

    Args:
        raw:          Raw file bytes.
        filename:     Original filename (used for extension detection and metadata).
        chunk_size:   Override ``Settings.CHUNK_SIZE``.
        chunk_overlap: Override ``Settings.CHUNK_OVERLAP``.

    Returns:
        List of (text, metadata) where metadata contains at least
        ``{"source": filename, "chunk_index": int}``.
    """
    full_text = extract_text(raw, filename)
    raw_chunks = split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result: List[Chunk] = [
        (chunk, {"source": filename, "chunk_index": i})
        for i, chunk in enumerate(raw_chunks)
    ]
    logger.info(
        "[chunker] '%s' -> %d chars -> %d chunks",
        filename, len(full_text), len(result),
    )
    return result
