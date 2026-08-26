"""图片 OCR 引擎 — 当所选对话模型不支持多模态输入时的回退方案。

优先使用 MinerU 服务（独立部署的文档解析 API，中文效果好，见 .env 的 MINERU_* 配置）；
MinerU 不可用时回退到本地 RapidOCR（纯 ONNX，无 PaddlePaddle 依赖，Windows 友好）。
引擎懒加载，仅在首次调用时初始化，避免拖慢后端启动。
"""
from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = None  # 延迟导入，避免在模块导入阶段触发 app 日志配置


def _get_logger():
    global logger
    if logger is None:
        from app.core.logger import get_logger
        logger = get_logger(__name__)
    return logger


class OCRUnavailableError(RuntimeError):
    """RapidOCR 未安装或初始化失败。"""


_ENGINE: Optional[object] = None
_ENGINE_FAILED: Optional[str] = None


def _lazy_engine():
    """返回单例 RapidOCR 引擎；首次调用时初始化，失败则缓存异常信息。"""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED is not None:
        raise OCRUnavailableError(_ENGINE_FAILED)
    try:
        from rapidocr import RapidOCR
        _ENGINE = RapidOCR()
        _get_logger().info("[ocr] RapidOCR 引擎初始化成功")
        return _ENGINE
    except Exception as exc:  # 安装缺失 / 模型下载失败 / DLL 缺失
        _ENGINE_FAILED = (
            "RapidOCR 未就绪，无法识别图片文字。请先安装："
            "pip install rapidocr （需要 onnxruntime，本项目已包含）。"
            f"原始错误：{exc}"
        )
        _get_logger().warning("[ocr] RapidOCR 初始化失败：%s", exc)
        raise OCRUnavailableError(_ENGINE_FAILED) from exc


def _decode_image(image_data_url: str):
    """从 data URL（data:image/png;base64,...）解码为 BGR numpy 数组。"""
    import cv2
    import numpy as np
    from PIL import Image
    match = re.match(r"data:image/[^;]+;base64,(.+)", image_data_url.strip())
    if not match:
        raise ValueError("图片格式不正确：期望 data:image/...;base64,... 形式")
    raw = base64.b64decode(match.group(1))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # 兜底：用 PIL 解码再转 BGR
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("图片解码失败")
    return img


def _decode_data_url_bytes(image_data_url: str) -> Tuple[bytes, str]:
    """从 data URL 解码出原始字节和 MIME 类型（供 MinerU 上传使用）。"""
    match = re.match(r"data:(image/[^;]+);base64,(.+)", image_data_url.strip())
    if not match:
        raise ValueError("图片格式不正确：期望 data:image/...;base64,... 形式")
    return base64.b64decode(match.group(2)), match.group(1).lower()


def _ocr_with_rapidocr(image_data_url: str) -> str:
    """用本地 RapidOCR 识别一张图片，返回纯文本（多行合并）。"""
    engine = _lazy_engine()
    img = _decode_image(image_data_url)
    result = engine(img)
    txts: List[str] = []
    txts_field = getattr(result, "txts", None)
    if txts_field is not None:
        # 新版对象形式
        txts = [str(t) for t in txts_field]
    elif isinstance(result, tuple) and len(result) >= 2:
        # 旧版元组形式 (boxes, txts, scores)
        txts = [str(t) for t in (result[1] or [])]
    return "\n".join(line.strip() for line in txts if str(line).strip())


async def _ocr_with_mineru(image_data_url: str) -> str:
    """用已部署的 MinerU 服务识别图片，返回 Markdown 文本。"""
    from app.core.config import get_settings
    from app.rag.parsers.mineru_client import MinerUClient, MinerUParseOptions

    cfg = get_settings()
    raw, mime = _decode_data_url_bytes(image_data_url)
    ext = mimetypes.guess_extension(mime) or ".png"
    filename = f"chat_image{ext}"
    options = MinerUParseOptions(
        backend=cfg.MINERU_BACKEND,
        languages=(cfg.MINERU_LANG,),
        parse_method="auto",
        formula_enable=False,
        table_enable=False,
        image_analysis=False,
        return_markdown=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        return_original_file=False,
    )
    client = MinerUClient()
    try:
        submission = await client.submit_document(
            raw, filename, content_type=mime, options=options
        )
        await client.wait_for_completion(submission.task_id, timeout=90)
        with tempfile.TemporaryDirectory(prefix="easyrag-ocr-") as temp_dir:
            result_path = Path(temp_dir) / "result.zip"
            await client.download_result(submission.task_id, result_path)
            with zipfile.ZipFile(result_path) as archive:
                markdown_names = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".md")
                ]
                if not markdown_names:
                    return ""
                return (
                    archive.read(markdown_names[0])
                    .decode("utf-8", errors="replace")
                    .strip()
                )
    finally:
        await client.aclose()


async def ocr_image_to_text(image_data_url: str) -> str:
    """对一张图片做 OCR，返回识别出的纯文本（多行合并）。

    优先 MinerU（若 .env 开启 MINERU_ENABLED），失败时回退 RapidOCR；
    两者都不可用时抛出 OCRUnavailableError，由调用方提示用户。
    """
    from app.core.config import get_settings

    if get_settings().MINERU_ENABLED:
        try:
            return await _ocr_with_mineru(image_data_url)
        except Exception as exc:
            _get_logger().warning("[ocr] MinerU OCR 失败，回退 RapidOCR：%s", exc)
    return await asyncio.to_thread(_ocr_with_rapidocr, image_data_url)


def is_available() -> bool:
    """OCR 链路当前是否可用（MinerU 已配置或 RapidOCR 可初始化）。"""
    from app.core.config import get_settings

    if get_settings().MINERU_ENABLED:
        return True
    try:
        _lazy_engine()
        return True
    except OCRUnavailableError:
        return False
