"""OCR 链路（阶段 2B）— RapidOCR 图片文字识别。

用途：
  - 图片文件（.png/.jpg/.jpeg/.bmp/.webp）直接 OCR 提取文本；
  - 扫描版 PDF（pypdf 提取不到文字的页）渲染成图片后 OCR 兜底。

RapidOCR 是本地推理（ONNX），无需外部服务；首次调用会加载模型，用进程级单例避免重复加载。
"""
from __future__ import annotations

import io
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)

_engine = None


def _get_engine():
    """RapidOCR 进程级单例（模型加载昂贵）。"""
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
        logger.info("[ocr] RapidOCR engine initialised")
    return _engine


def ocr_image_bytes(raw: bytes) -> str:
    """对图片字节做 OCR，返回拼接的纯文本（按行）。"""
    engine = _get_engine()
    result = engine(raw)
    # RapidOCROutput: .txts 为识别出的文本行列表
    txts: List[str] = list(getattr(result, "txts", None) or [])
    text = "\n".join(t for t in txts if t and t.strip())
    logger.info("[ocr] image -> %d lines, %d chars", len(txts), len(text))
    return text


def ocr_pdf_bytes(raw: bytes, dpi: int = 200) -> str:
    """对 PDF 逐页渲染成图片后 OCR（扫描版 PDF 兜底）。

    依赖 pypdfium2（pdfium 的 Python 绑定）；未安装时给出明确报错。
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError(
            "pypdfium2 is not installed. Run: pip install pypdfium2"
        ) from exc

    pdf = pdfium.PdfDocument(raw)
    page_texts: List[str] = []
    for page in pdf:
        bitmap = page.render(scale=dpi / 72)
        img = bitmap.to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text = ocr_image_bytes(buf.getvalue())
        if text.strip():
            page_texts.append(text)
    logger.info("[ocr] pdf -> %d/%d pages with text", len(page_texts), len(pdf))
    return "\n\n".join(page_texts)
