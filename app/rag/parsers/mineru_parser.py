"""MinerU implementation of the unified document parser interface."""
from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from app.rag.parsers.base import (
    DocumentParser,
    DocumentParserError,
    ParserOutputError,
    TransientDocumentParserError,
)
from app.rag.parsers.mineru_client import (
    MinerUClient,
    MinerUConnectionError,
    MinerUError,
    MinerUParseOptions,
    MinerUResponseError,
    MinerUTaskTimeoutError,
)
from app.rag.parsers.models import (
    ParsedBlock,
    ParsedBlockType,
    ParsedBoundingBox,
    ParsedContentFormat,
    ParsedDocument,
    ParsedImage,
    ParserProvenance,
)


class MinerUParser(DocumentParser):
    """Parse documents remotely and normalize MinerU protocol-v2 artifacts."""

    parser_name = "mineru"
    supported_extensions = frozenset(
        {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
            ".gif",
            ".tif",
            ".tiff",
            ".jp2",
            ".docx",
            ".pptx",
            ".xlsx",
        }
    )

    def __init__(
        self,
        *,
        client: MinerUClient | None = None,
        options: MinerUParseOptions | None = None,
        max_archive_entries: int = 5000,
        max_uncompressed_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if max_archive_entries <= 0 or max_uncompressed_bytes <= 0:
            raise ValueError("MinerU archive limits must be positive")
        self._client = client
        self._options = options
        self._max_archive_entries = max_archive_entries
        self._max_uncompressed_bytes = max_uncompressed_bytes

    async def parse(
        self,
        raw: bytes,
        filename: str,
        *,
        content_type: str | None = None,
    ) -> ParsedDocument:
        self.validate_input(raw, filename)
        client = self._client or MinerUClient()
        owns_client = self._client is None

        try:
            health = await client.health()
            if health.status != "healthy":
                raise TransientDocumentParserError(
                    f"MinerU service is not healthy: {health.status}"
                )
            options = self._options or client.default_options()
            submission = await client.submit_document(
                raw,
                filename,
                content_type=content_type,
                options=options,
            )
            await client.wait_for_completion(submission.task_id)

            with tempfile.TemporaryDirectory(prefix="easyrag-mineru-") as temp_dir:
                result_path = Path(temp_dir) / "result.zip"
                await client.download_result(submission.task_id, result_path)
                return await asyncio.to_thread(
                    _read_mineru_archive,
                    result_path,
                    source_name=filename,
                    source_raw=raw,
                    source_mime_type=(
                        content_type
                        or mimetypes.guess_type(filename)[0]
                        or "application/octet-stream"
                    ),
                    parser_version=health.version,
                    task_id=submission.task_id,
                    options=options,
                    file_names=submission.file_names,
                    max_entries=self._max_archive_entries,
                    max_uncompressed_bytes=self._max_uncompressed_bytes,
                )
        except TransientDocumentParserError:
            raise
        except (MinerUConnectionError, MinerUTaskTimeoutError) as exc:
            raise TransientDocumentParserError(str(exc)) from exc
        except MinerUResponseError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                raise TransientDocumentParserError(str(exc)) from exc
            raise DocumentParserError(f"MinerU parsing failed: {exc}") from exc
        except MinerUError as exc:
            raise DocumentParserError(f"MinerU parsing failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()


def _read_mineru_archive(
    archive_path: Path,
    *,
    source_name: str,
    source_raw: bytes,
    source_mime_type: str,
    parser_version: str | None,
    task_id: str,
    options: MinerUParseOptions,
    file_names: tuple[str, ...],
    max_entries: int,
    max_uncompressed_bytes: int,
) -> ParsedDocument:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = _validated_entries(
                archive,
                max_entries=max_entries,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            markdown_path = _single_artifact(
                entries, lambda path: path.suffix.lower() == ".md", "Markdown"
            )
            content_list_path = _single_artifact(
                entries,
                lambda path: path.name.endswith("_content_list.json"),
                "content_list JSON",
            )
            artifact_root = markdown_path.parent
            if content_list_path.parent != artifact_root:
                raise ParserOutputError(
                    "MinerU Markdown and content_list artifacts have different roots"
                )

            markdown = _read_utf8(archive, entries[markdown_path], "Markdown")
            content_list = _read_content_list(archive, entries[content_list_path])
            blocks, warnings = _normalize_blocks(content_list)
            images = _read_images(
                archive,
                entries,
                artifact_root=artifact_root,
                blocks=blocks,
            )
    except zipfile.BadZipFile as exc:
        raise ParserOutputError("MinerU result is not a valid ZIP archive") from exc

    page_indexes = [
        block.page_index for block in blocks if block.page_index is not None
    ]
    page_count = max(page_indexes) + 1 if page_indexes else None
    image_paths = {image.path for image in images}
    missing_images = sorted(
        {
            block.image_path
            for block in blocks
            if block.image_path and block.image_path not in image_paths
        }
    )
    if missing_images:
        warnings.append(
            "MinerU referenced images missing from result: " + ", ".join(missing_images)
        )

    return ParsedDocument(
        source_name=source_name,
        source_mime_type=source_mime_type,
        source_sha256=hashlib.sha256(source_raw).hexdigest(),
        markdown=markdown,
        blocks=tuple(blocks),
        images=tuple(images),
        page_count=page_count,
        provenance=ParserProvenance(
            parser_name="mineru",
            parser_version=parser_version,
            task_id=task_id,
            backend=options.backend,
            parse_method=options.parse_method,
            languages=options.languages,
            options=asdict(options),
        ),
        warnings=tuple(warnings),
        metadata={
            "mineru_file_names": file_names,
            "artifact_count": len(entries),
        },
    )


def _validated_entries(
    archive: zipfile.ZipFile,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
) -> dict[PurePosixPath, zipfile.ZipInfo]:
    infos = [info for info in archive.infolist() if not info.is_dir()]
    if len(infos) > max_entries:
        raise ParserOutputError(
            f"MinerU ZIP contains too many entries ({len(infos)} > {max_entries})"
        )
    if sum(info.file_size for info in infos) > max_uncompressed_bytes:
        raise ParserOutputError("MinerU ZIP exceeds the uncompressed size limit")

    entries: dict[PurePosixPath, zipfile.ZipInfo] = {}
    for info in infos:
        if info.flag_bits & 0x1:
            raise ParserOutputError("MinerU ZIP must not contain encrypted entries")
        if "\\" in info.filename or "\x00" in info.filename:
            raise ParserOutputError(f"Unsafe MinerU ZIP entry: {info.filename!r}")
        path = PurePosixPath(info.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or any(":" in part for part in path.parts)
        ):
            raise ParserOutputError(f"Unsafe MinerU ZIP entry: {info.filename!r}")
        if path in entries:
            raise ParserOutputError(f"Duplicate MinerU ZIP entry: {info.filename!r}")
        entries[path] = info
    return entries


def _single_artifact(
    entries: Mapping[PurePosixPath, zipfile.ZipInfo],
    predicate: Callable[[PurePosixPath], bool],
    label: str,
) -> PurePosixPath:
    matches = [path for path in entries if predicate(path)]
    if len(matches) != 1:
        raise ParserOutputError(
            f"Expected exactly one MinerU {label} artifact, found {len(matches)}"
        )
    return matches[0]


def _read_utf8(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    label: str,
) -> str:
    try:
        return archive.read(info).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParserOutputError(f"MinerU {label} is not UTF-8") from exc


def _read_content_list(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(_read_utf8(archive, info, "content_list JSON"))
    except json.JSONDecodeError as exc:
        raise ParserOutputError("MinerU content_list is invalid JSON") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ParserOutputError("MinerU content_list must be a list of objects")
    return payload


def _normalize_blocks(
    content_list: list[Mapping[str, Any]],
) -> tuple[list[ParsedBlock], list[str]]:
    blocks: list[ParsedBlock] = []
    warnings: list[str] = []
    title_stack: list[tuple[int, str]] = []

    for index, raw_block in enumerate(content_list):
        source_type = raw_block.get("type")
        source_type = source_type if isinstance(source_type, str) else "unknown"
        heading_level = _heading_level(raw_block)
        block_type = _block_type(source_type, heading_level)
        text, content_format = _block_text(raw_block, block_type)

        if block_type is ParsedBlockType.TITLE and heading_level and text.strip():
            title_stack = [
                item for item in title_stack if item[0] < heading_level
            ]
            title_stack.append((heading_level, text.strip()))

        image_path = raw_block.get("img_path")
        image_path = image_path if isinstance(image_path, str) and image_path else None
        bbox = _bounding_box(raw_block.get("bbox"), index, warnings)
        page_index = _nonnegative_int(raw_block.get("page_idx"))
        captions = _strings(
            raw_block.get("chart_caption") or raw_block.get("table_caption")
        )
        footnotes = _strings(
            raw_block.get("chart_footnote") or raw_block.get("table_footnote")
        )
        if block_type is ParsedBlockType.UNKNOWN:
            warnings.append(f"Unknown MinerU block type: {source_type}")

        excluded = {
            "type",
            "text",
            "table_body",
            "list_items",
            "img_path",
            "bbox",
            "page_idx",
            "chart_caption",
            "table_caption",
            "chart_footnote",
            "table_footnote",
        }
        try:
            block = ParsedBlock(
                block_id=f"mineru-{index}",
                type=block_type,
                text=text,
                content_format=content_format,
                page_index=page_index,
                bbox=bbox,
                heading_level=heading_level,
                section_path=tuple(title for _, title in title_stack),
                image_path=image_path,
                captions=captions,
                footnotes=footnotes,
                source_type=source_type,
                metadata={
                    key: value for key, value in raw_block.items() if key not in excluded
                },
            )
        except (TypeError, ValueError) as exc:
            raise ParserOutputError(
                f"Invalid MinerU block {index}: {exc}"
            ) from exc
        blocks.append(block)
    return blocks, warnings


def _block_type(source_type: str, heading_level: int | None) -> ParsedBlockType:
    if source_type == "text":
        return ParsedBlockType.TITLE if heading_level else ParsedBlockType.TEXT
    mapping = {
        "list": ParsedBlockType.LIST,
        "table": ParsedBlockType.TABLE,
        "equation": ParsedBlockType.EQUATION,
        "interline_equation": ParsedBlockType.EQUATION,
        "inline_equation": ParsedBlockType.EQUATION,
        "image": ParsedBlockType.IMAGE,
        "figure": ParsedBlockType.IMAGE,
        "chart": ParsedBlockType.CHART,
        "code": ParsedBlockType.CODE,
        "header": ParsedBlockType.HEADER,
        "footer": ParsedBlockType.FOOTER,
        "page_number": ParsedBlockType.PAGE_NUMBER,
        "page_footnote": ParsedBlockType.FOOTNOTE,
        "footnote": ParsedBlockType.FOOTNOTE,
    }
    return mapping.get(source_type, ParsedBlockType.UNKNOWN)


def _block_text(
    raw_block: Mapping[str, Any],
    block_type: ParsedBlockType,
) -> tuple[str, ParsedContentFormat]:
    if block_type is ParsedBlockType.TABLE:
        value = raw_block.get("table_body")
        return (value if isinstance(value, str) else ""), ParsedContentFormat.HTML
    if block_type is ParsedBlockType.LIST:
        items = _strings(raw_block.get("list_items"))
        return "\n".join(f"- {item}" for item in items), ParsedContentFormat.MARKDOWN
    value = raw_block.get("text")
    if not isinstance(value, str) and block_type is ParsedBlockType.CHART:
        value = raw_block.get("content")
    text = value if isinstance(value, str) else ""
    if block_type is ParsedBlockType.EQUATION:
        return text, ParsedContentFormat.LATEX
    if block_type is ParsedBlockType.CODE:
        return text, ParsedContentFormat.MARKDOWN
    return text, ParsedContentFormat.PLAIN_TEXT


def _read_images(
    archive: zipfile.ZipFile,
    entries: Mapping[PurePosixPath, zipfile.ZipInfo],
    *,
    artifact_root: PurePosixPath,
    blocks: list[ParsedBlock],
) -> list[ParsedImage]:
    references = {
        block.image_path: (block.page_index, block.bbox)
        for block in blocks
        if block.image_path
    }
    images: list[ParsedImage] = []
    for path, info in entries.items():
        try:
            relative = path.relative_to(artifact_root)
        except ValueError:
            continue
        if len(relative.parts) < 2 or relative.parts[0] != "images":
            continue
        image_path = relative.as_posix()
        data = archive.read(info)
        if not data:
            continue
        page_index, bbox = references.get(image_path, (None, None))
        images.append(
            ParsedImage(
                path=image_path,
                data=data,
                mime_type=(
                    mimetypes.guess_type(image_path)[0]
                    or "application/octet-stream"
                ),
                page_index=page_index,
                bbox=bbox,
            )
        )
    return images


def _heading_level(raw_block: Mapping[str, Any]) -> int | None:
    value = raw_block.get("text_level")
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 6:
        return value
    return None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _bounding_box(
    value: Any,
    block_index: int,
    warnings: list[str],
) -> ParsedBoundingBox | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    ):
        return None
    try:
        return ParsedBoundingBox(*value)
    except ValueError:
        warnings.append(f"Invalid bounding box on MinerU block {block_index}")
        return None
