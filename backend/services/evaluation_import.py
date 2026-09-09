"""Parse JSON/CSV Golden Set uploads into a small, format-neutral payload."""

from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any


MAX_IMPORT_BYTES = 5 * 1024 * 1024


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "是",
    }


def _parse_chunk_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif value is None:
        raw = []
    else:
        text = str(value).strip()
        if not text:
            raw = []
        elif text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = []
            raw = decoded if isinstance(decoded, list) else []
        else:
            separator = "|" if "|" in text else ";" if ";" in text else ","
            raw = text.split(separator)
    result: list[str] = []
    for item in raw:
        item = str(item).strip()
        if item and item not in result:
            result.append(item)
    return result


def _normalise_case(raw: Any, row_number: int) -> tuple[dict[str, Any], list[dict]]:
    if not isinstance(raw, dict):
        return {}, [{"row": row_number, "field": "row", "message": "必须是对象"}]

    case = {
        "question": str(raw.get("question") or "").strip(),
        "reference_answer": str(raw.get("reference_answer") or "").strip(),
        "expected_file_id": str(raw.get("expected_file_id") or "").strip(),
        "expected_filename": str(raw.get("expected_filename") or "").strip(),
        "expected_chunk_ids": _parse_chunk_ids(raw.get("expected_chunk_ids")),
        "expect_miss": _parse_bool(raw.get("expect_miss")),
    }
    errors: list[dict] = []
    if not case["question"]:
        errors.append({"row": row_number, "field": "question", "message": "不能为空"})
    if not case["expected_file_id"] and not case["expected_filename"]:
        errors.append({
            "row": row_number,
            "field": "expected_file",
            "message": "expected_file_id 和 expected_filename 至少填写一个",
        })
    return case, errors


def parse_dataset_import(filename: str, content: bytes) -> dict[str, Any]:
    if len(content) > MAX_IMPORT_BYTES:
        raise ValueError("Import file exceeds the 5 MB limit")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Import file must use UTF-8 encoding") from exc

    suffix = Path(filename or "").suffix.lower()
    dataset_name = Path(filename or "golden-set").stem or "golden-set"
    description = ""
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {exc.lineno}: {exc.msg}") from exc
        if isinstance(payload, dict):
            dataset_name = str(payload.get("name") or dataset_name).strip()
            description = str(payload.get("description") or "").strip()
            rows = payload.get("cases")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("JSON must be a case array or an object containing cases")
    elif suffix == ".csv":
        reader = csv.DictReader(StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV header is missing")
        rows = list(reader)
    else:
        raise ValueError("Only .json and .csv imports are supported")

    cases: list[dict[str, Any]] = []
    errors: list[dict] = []
    for index, raw in enumerate(rows, start=1):
        case, row_errors = _normalise_case(raw, index)
        cases.append(case)
        errors.extend(row_errors)
    if not cases:
        errors.append({"row": 0, "field": "cases", "message": "至少需要一条用例"})
    return {
        "name": dataset_name,
        "description": description,
        "cases": cases,
        "errors": errors,
    }
