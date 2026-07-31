"""Document chunker – parses uploaded files and splits into text chunks.

Supported file types:
    .txt / .md  – plain text
    .pdf        – via pypdf (``pip install pypdf``)
    .docx       – via python-docx (``pip install python-docx``)

Chunking strategies (阶段 2A, selected by ``Settings.CHUNK_STRATEGY``):
    fixed        – fixed-size sliding window with overlap (original behaviour)
    recursive    – recursive separator split (paragraph → sentence → word)
    markdown     – Markdown structure-aware (heading hierarchy, code blocks kept whole)
    parent_child – small child chunks for retrieval + large parent chunk as context
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
    """Extract text from PDF using pypdf；扫描版（整份文档几乎无文字层）OCR 兜底。"""
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
    text = "\n".join(pages)
    # 阶段 2B：扫描版 PDF — 平均每页不足 20 个字符视为无文字层，走 OCR
    if len(reader.pages) > 0 and len(text.strip()) / len(reader.pages) < 20:
        logger.info("[chunker] '%s' looks like a scanned PDF, falling back to OCR", filename)
        from app.rag.ocr import ocr_pdf_bytes
        ocr_text = ocr_pdf_bytes(raw)
        if ocr_text.strip():
            return ocr_text
        logger.warning("[chunker] OCR produced no text for '%s', keeping pypdf result", filename)
    return text


def _extract_image(raw: bytes, filename: str) -> str:
    """阶段 2B：图片文件直接 OCR。"""
    from app.rag.ocr import ocr_image_bytes
    return ocr_image_bytes(raw)


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
    ".png":  _extract_image,
    ".jpg":  _extract_image,
    ".jpeg": _extract_image,
    ".bmp":  _extract_image,
    ".webp": _extract_image,
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


# ── fixed: sliding-window chunker（原行为） ───────────────────────────────────

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


# ── 阶段 2A: recursive 递归分隔符切分 ─────────────────────────────────────────

# 分隔符按语义强度排序：段落 → 句子 → 分句 → 词 → 字符
_RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " ", ""]


def _recursive_split(text: str, size: int, separators: List[str]) -> List[str]:
    """递归切分：优先在最强分隔符处断开，保证块尽量落在语义边界上。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sep = separators[0]
    rest = separators[1:]
    if sep == "":  # 兜底：硬切
        return [text[i:i + size] for i in range(0, len(text), size)]

    pieces = text.split(sep)
    # 若当前分隔符无法让任何块变小，降一级
    if len(pieces) == 1:
        return _recursive_split(text, size, rest)

    chunks: List[str] = []
    buf = ""
    for piece in pieces:
        candidate = (buf + sep + piece) if buf else piece
        if len(candidate) <= size:
            buf = candidate
            continue
        # buf 已满：先落盘，再处理当前 piece
        if buf:
            chunks.append(buf)
        if len(piece) <= size:
            buf = piece
        else:  # piece 自身超长了，降一级继续拆
            chunks.extend(_recursive_split(piece, size, rest))
            buf = ""
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def split_recursive(text: str, chunk_size: int | None = None) -> List[str]:
    """递归分隔符切分，不加 overlap（边界本身更干净）。"""
    size = chunk_size or cfg.CHUNK_SIZE
    return _recursive_split(text.strip(), size, _RECURSIVE_SEPARATORS)


# ── 阶段 2A: markdown 结构感知切分 ───────────────────────────────────────────

def _flush_section(chunks: List[str], title_path: List[str], body: List[str], size: int) -> None:
    """把一个标题区块落盘为若干 chunk；超长区块退回 recursive 拆分。"""
    text = "\n".join(body).strip()
    if not text:
        return
    header = " > ".join(title_path)
    prefixed = f"[{header}]\n{text}" if header else text
    if len(prefixed) <= size:
        chunks.append(prefixed)
    else:
        for sub in split_recursive(text, size):
            chunks.append(f"[{header}]\n{sub}" if header else sub)


def split_markdown(text: str, chunk_size: int | None = None) -> List[str]:
    """Markdown 结构感知：按标题层级聚合内容，代码块（```）不拆。

    每个 chunk 携带所属标题路径前缀（如 "[章节A > 小节B]"），检索时可还原上下文。
    """
    size = chunk_size or cfg.CHUNK_SIZE
    lines = text.split("\n")
    chunks: List[str] = []
    title_stack: List[Tuple[int, str]] = []  # (level, title)
    body: List[str] = []
    in_code = False

    def flush() -> None:
        nonlocal body
        _flush_section(chunks, [t for _, t in title_stack], body, size)
        body = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            body.append(line)
            continue
        if not in_code and stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if 1 <= level <= 6 and stripped[level:].strip():
                flush()
                title = stripped[level:].strip()
                title_stack = [(lv, t) for lv, t in title_stack if lv < level]
                title_stack.append((level, title))
                continue
        body.append(line)
    flush()
    return chunks


# ── 阶段 2A: parent_child 父子分块 ───────────────────────────────────────────

def split_parent_child(
    text: str,
    chunk_size: int | None = None,
    parent_size: int | None = None,
) -> List[Tuple[str, str]]:
    """父子分块：返回 (child_text, parent_text) 列表。

    child 小块（CHUNK_SIZE）用于向量索引保证召回精度；
    parent 大块（PARENT_CHUNK_SIZE）作为回答时的上下文，避免上下文残缺。
    """
    size = chunk_size or cfg.CHUNK_SIZE
    psize = parent_size or cfg.PARENT_CHUNK_SIZE
    parents = split_recursive(text, psize)
    pairs: List[Tuple[str, str]] = []
    for parent in parents:
        children = split_recursive(parent, size) if len(parent) > size else [parent]
        for child in children:
            pairs.append((child, parent))
    return pairs


# ── High-level entry point ────────────────────────────────────────────────────

def parse_and_chunk(
    raw: bytes,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    strategy: str | None = None,
) -> List[Chunk]:
    """Extract text from *raw* bytes and return a list of (chunk_text, metadata) tuples.

    Args:
        raw:          Raw file bytes.
        filename:     Original filename (used for extension detection and metadata).
        chunk_size:   Override ``Settings.CHUNK_SIZE``.
        chunk_overlap: Override ``Settings.CHUNK_OVERLAP`` (fixed 策略专用).
        strategy:     Override ``Settings.CHUNK_STRATEGY``.

    Returns:
        List of (text, metadata). metadata 至少含 ``{"source", "chunk_index", "strategy"}``；
        markdown 策略额外含 ``section_path``；parent_child 策略额外含 ``parent_text``，
        检索命中后用 parent_text 替换返回内容以获得完整上下文。
    """
    full_text = extract_text(raw, filename)
    strategy = strategy or cfg.CHUNK_STRATEGY

    base_meta = {"source": filename, "strategy": strategy}
    result: List[Chunk] = []

    if strategy == "fixed":
        raw_chunks = split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        result = [(c, {**base_meta, "chunk_index": i}) for i, c in enumerate(raw_chunks)]

    elif strategy == "recursive":
        raw_chunks = split_recursive(full_text, chunk_size=chunk_size)
        result = [(c, {**base_meta, "chunk_index": i}) for i, c in enumerate(raw_chunks)]

    elif strategy == "markdown":
        raw_chunks = split_markdown(full_text, chunk_size=chunk_size)
        result = []
        for i, c in enumerate(raw_chunks):
            section = ""
            if c.startswith("[") and "]\n" in c:
                section = c[1:c.index("]\n")]
            result.append((c, {**base_meta, "chunk_index": i, "section_path": section}))

    elif strategy == "parent_child":
        pairs = split_parent_child(full_text, chunk_size=chunk_size)
        result = [
            (child, {**base_meta, "chunk_index": i, "parent_text": parent})
            for i, (child, parent) in enumerate(pairs)
        ]

    else:
        raise ValueError(f"Unknown chunk strategy '{strategy}'")

    logger.info(
        "[chunker] '%s' -> %d chars -> %d chunks (strategy=%s)",
        filename, len(full_text), len(result), strategy,
    )
    return result
