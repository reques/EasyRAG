"""Async client for the independently deployed MinerU API.

The client deliberately stops at the service boundary: it submits one document,
tracks the asynchronous task and downloads the result ZIP. Extracting and
interpreting MinerU artifacts belongs to the parser layer built on top of it.
"""
from __future__ import annotations

import asyncio
import mimetypes
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import httpx

from app.core.config import get_settings


class MinerUError(RuntimeError):
    """Base error for MinerU client operations."""


class MinerUConnectionError(MinerUError):
    """MinerU could not be reached or an HTTP request timed out."""


class MinerUProtocolError(MinerUError):
    """MinerU returned a payload that does not match its API contract."""


class MinerUResponseError(MinerUError):
    """MinerU returned an unexpected HTTP response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"MinerU returned HTTP {status_code}: {detail}")


class MinerUTaskFailedError(MinerUError):
    """A MinerU asynchronous parse task ended in failure."""

    def __init__(self, task_id: str, detail: str):
        self.task_id = task_id
        self.detail = detail
        super().__init__(f"MinerU task {task_id} failed: {detail}")


class MinerUTaskTimeoutError(MinerUError):
    """A MinerU task did not finish before the caller's deadline."""

    def __init__(self, task_id: str, timeout_seconds: float):
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"MinerU task {task_id} did not finish within {timeout_seconds:g} seconds"
        )


class MinerUTaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class MinerUHealth:
    status: str
    version: str | None = None
    protocol_version: int | str | None = None
    max_concurrent_requests: int | None = None
    processing_window_size: int | None = None


@dataclass(frozen=True)
class MinerUTask:
    task_id: str
    status: MinerUTaskStatus
    backend: str | None = None
    file_names: tuple[str, ...] = ()
    queued_ahead: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MinerUSubmission:
    task_id: str
    status: MinerUTaskStatus
    backend: str | None = None
    file_names: tuple[str, ...] = ()
    queued_ahead: int | None = None


@dataclass(frozen=True)
class MinerUParseOptions:
    """Options for EasyRAG's structured-output MinerU request."""

    backend: str = "pipeline"
    parse_method: str = "auto"
    languages: tuple[str, ...] = ("ch",)
    formula_enable: bool = True
    table_enable: bool = True
    image_analysis: bool = True
    return_markdown: bool = True
    return_middle_json: bool = False
    return_model_output: bool = False
    return_content_list: bool = True
    return_images: bool = True
    return_original_file: bool = False
    start_page_id: int = 0
    end_page_id: int | None = None

    def to_form_data(self) -> dict[str, str | list[str]]:
        if not self.backend.strip():
            raise ValueError("MinerU backend must not be empty")
        if not self.languages:
            raise ValueError("MinerU languages must contain at least one language")
        if self.start_page_id < 0:
            raise ValueError("MinerU start_page_id must be non-negative")
        if self.end_page_id is not None and self.end_page_id < self.start_page_id:
            raise ValueError("MinerU end_page_id must not precede start_page_id")

        def boolean(value: bool) -> str:
            return str(value).lower()
        return {
            "backend": self.backend,
            "parse_method": self.parse_method,
            "lang_list": list(self.languages),
            "formula_enable": boolean(self.formula_enable),
            "table_enable": boolean(self.table_enable),
            "image_analysis": boolean(self.image_analysis),
            "return_md": boolean(self.return_markdown),
            "return_middle_json": boolean(self.return_middle_json),
            "return_model_output": boolean(self.return_model_output),
            "return_content_list": boolean(self.return_content_list),
            "return_images": boolean(self.return_images),
            "response_format_zip": "true",
            "return_original_file": boolean(self.return_original_file),
            "client_side_output_generation": "false",
            "start_page_id": str(self.start_page_id),
            "end_page_id": str(99999 if self.end_page_id is None else self.end_page_id),
        }


class MinerUClient:
    """Small async boundary around MinerU API protocol v2."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        backend: str | None = None,
        language: str | None = None,
        connect_timeout: float | None = None,
        request_timeout: float | None = None,
        result_download_timeout: float | None = None,
        task_timeout: float | None = None,
        poll_interval: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.MINERU_API_URL).rstrip("/")
        self.backend = backend or settings.MINERU_BACKEND
        self.language = language or settings.MINERU_LANG
        self.task_timeout = (
            settings.MINERU_TASK_TIMEOUT if task_timeout is None else task_timeout
        )
        self.poll_interval = (
            settings.MINERU_POLL_INTERVAL if poll_interval is None else poll_interval
        )
        self.result_download_timeout = (
            settings.MINERU_RESULT_DOWNLOAD_TIMEOUT
            if result_download_timeout is None
            else result_download_timeout
        )
        if not self.base_url:
            raise ValueError("MinerU API URL must not be empty")
        if self.task_timeout <= 0:
            raise ValueError("MinerU task timeout must be positive")
        if self.poll_interval < 0:
            raise ValueError("MinerU poll interval must be non-negative")

        self._owns_http_client = http_client is None
        if http_client is None:
            connect = (
                settings.MINERU_CONNECT_TIMEOUT
                if connect_timeout is None
                else connect_timeout
            )
            read = (
                settings.MINERU_REQUEST_TIMEOUT
                if request_timeout is None
                else request_timeout
            )
            timeout = httpx.Timeout(read, connect=connect)
            http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                follow_redirects=True,
                trust_env=False,
                headers={"User-Agent": "EasyRAG-MinerUClient/1.0"},
            )
        self._http = http_client

    async def __aenter__(self) -> "MinerUClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    def default_options(self) -> MinerUParseOptions:
        return MinerUParseOptions(
            backend=self.backend,
            languages=(self.language,),
        )

    async def health(self) -> MinerUHealth:
        response = await self._request("GET", "/health")
        self._expect_status(response, 200)
        payload = self._json_object(response)
        status = payload.get("status")
        if not isinstance(status, str):
            raise MinerUProtocolError("MinerU health payload has no string status")
        return MinerUHealth(
            status=status,
            version=self._optional_string(payload.get("version")),
            protocol_version=self._optional_protocol_version(
                payload.get("protocol_version")
            ),
            max_concurrent_requests=self._optional_int(
                payload.get("max_concurrent_requests")
            ),
            processing_window_size=self._optional_int(
                payload.get("processing_window_size")
            ),
        )

    async def submit_document(
        self,
        content: bytes,
        filename: str,
        *,
        content_type: str | None = None,
        options: MinerUParseOptions | None = None,
    ) -> MinerUSubmission:
        if not content:
            raise ValueError("Cannot submit an empty document to MinerU")
        if not filename or Path(filename).name != filename:
            raise ValueError("MinerU filename must be a non-empty basename")
        mime_type = (
            content_type
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )
        response = await self._request(
            "POST",
            "/tasks",
            data=(options or self.default_options()).to_form_data(),
            files={"files": (filename, content, mime_type)},
        )
        self._expect_status(response, 202)
        task = self._parse_task(self._json_object(response))
        return MinerUSubmission(
            task_id=task.task_id,
            status=task.status,
            backend=task.backend,
            file_names=task.file_names,
            queued_ahead=task.queued_ahead,
        )

    async def submit_file(
        self,
        file_path: str | Path,
        *,
        filename: str | None = None,
        options: MinerUParseOptions | None = None,
    ) -> MinerUSubmission:
        path = Path(file_path)
        upload_name = filename or path.name
        content = await asyncio.to_thread(path.read_bytes)
        return await self.submit_document(content, upload_name, options=options)

    async def get_task(self, task_id: str) -> MinerUTask:
        self._validate_task_id(task_id)
        response = await self._request("GET", f"/tasks/{task_id}")
        self._expect_status(response, 200)
        return self._parse_task(self._json_object(response), expected_task_id=task_id)

    async def wait_for_completion(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> MinerUTask:
        effective_timeout = self.task_timeout if timeout is None else timeout
        if effective_timeout <= 0:
            raise ValueError("MinerU task timeout must be positive")
        deadline = time.monotonic() + effective_timeout

        while True:
            try:
                task = await self.get_task(task_id)
            except MinerUConnectionError:
                if time.monotonic() >= deadline:
                    raise MinerUTaskTimeoutError(task_id, effective_timeout) from None
            else:
                if task.status is MinerUTaskStatus.COMPLETED:
                    return task
                if task.status is MinerUTaskStatus.FAILED:
                    raise MinerUTaskFailedError(
                        task.task_id, task.error or "unknown task error"
                    )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MinerUTaskTimeoutError(task_id, effective_timeout)
            await asyncio.sleep(min(self.poll_interval, remaining))

    async def download_result(
        self,
        task_id: str,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Stream a completed task's ZIP to disk and return its final path."""
        self._validate_task_id(task_id)
        output_path = Path(destination)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"MinerU result already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(
            f".{output_path.name}.{uuid4().hex}.part"
        )

        try:
            try:
                stream = self._http.stream(
                    "GET",
                    self._url(f"/tasks/{task_id}/result"),
                    timeout=self.result_download_timeout,
                )
                async with stream as response:
                    if response.status_code == 202:
                        raise MinerUResponseError(202, "task result is not ready")
                    if response.status_code == 409:
                        detail = await self._async_response_detail(response)
                        raise MinerUTaskFailedError(task_id, detail)
                    if response.status_code != 200:
                        detail = await self._async_response_detail(response)
                        raise MinerUResponseError(response.status_code, detail)
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/zip" not in content_type:
                        raise MinerUProtocolError(
                            "MinerU task result is not an application/zip response"
                        )
                    with partial_path.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            handle.write(chunk)
            except httpx.TimeoutException as exc:
                raise MinerUConnectionError(
                    f"Timed out downloading MinerU result for task {task_id}"
                ) from exc
            except httpx.RequestError as exc:
                raise MinerUConnectionError(
                    f"Could not download MinerU result for task {task_id}: {exc}"
                ) from exc

            if output_path.exists() and not overwrite:
                raise FileExistsError(f"MinerU result already exists: {output_path}")
            os.replace(partial_path, output_path)
            return output_path
        finally:
            partial_path.unlink(missing_ok=True)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._http.request(method, self._url(path), **kwargs)
        except httpx.TimeoutException as exc:
            raise MinerUConnectionError(
                f"Timed out calling MinerU at {self.base_url}{path}"
            ) from exc
        except httpx.RequestError as exc:
            raise MinerUConnectionError(
                f"Could not reach MinerU at {self.base_url}{path}: {exc}"
            ) from exc

    def _url(self, path: str) -> str:
        # Avoid trusting status_url/result_url echoed by a remote service.
        return f"{self.base_url}{path}"

    @classmethod
    def _parse_task(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_task_id: str | None = None,
    ) -> MinerUTask:
        task_id = payload.get("task_id")
        raw_status = payload.get("status")
        if not isinstance(task_id, str) or not task_id:
            raise MinerUProtocolError("MinerU task payload has no valid task_id")
        if expected_task_id is not None and task_id != expected_task_id:
            raise MinerUProtocolError("MinerU returned a different task_id")
        try:
            status = MinerUTaskStatus(raw_status)
        except (TypeError, ValueError) as exc:
            raise MinerUProtocolError(
                f"MinerU task payload has unknown status: {raw_status!r}"
            ) from exc

        raw_file_names = payload.get("file_names")
        file_names: tuple[str, ...] = ()
        if isinstance(raw_file_names, list) and all(
            isinstance(item, str) for item in raw_file_names
        ):
            file_names = tuple(raw_file_names)

        return MinerUTask(
            task_id=task_id,
            status=status,
            backend=cls._optional_string(payload.get("backend")),
            file_names=file_names,
            queued_ahead=cls._optional_int(payload.get("queued_ahead")),
            created_at=cls._optional_string(payload.get("created_at")),
            started_at=cls._optional_string(payload.get("started_at")),
            completed_at=cls._optional_string(payload.get("completed_at")),
            error=cls._optional_string(payload.get("error")),
        )

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not task_id or any(character in task_id for character in "/?#"):
            raise ValueError("Invalid MinerU task_id")

    @staticmethod
    def _expect_status(response: httpx.Response, expected: int) -> None:
        if response.status_code == expected:
            return
        raise MinerUResponseError(response.status_code, MinerUClient._response_detail(response))

    @staticmethod
    def _json_object(response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MinerUProtocolError("MinerU returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise MinerUProtocolError("MinerU returned a non-object JSON payload")
        return payload

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:500] or "empty response"
        if isinstance(payload, dict):
            for key in ("detail", "message", "error"):
                value = payload.get(key)
                if value:
                    return str(value)[:500]
        return str(payload)[:500]

    @staticmethod
    async def _async_response_detail(response: httpx.Response) -> str:
        await response.aread()
        return MinerUClient._response_detail(response)

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _optional_protocol_version(value: Any) -> int | str | None:
        if isinstance(value, bool):
            return None
        return value if isinstance(value, (int, str)) else None
