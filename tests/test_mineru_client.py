from __future__ import annotations

import zipfile
from io import BytesIO

import httpx
import pytest

from app.rag.parsers.mineru_client import (
    MinerUClient,
    MinerUParseOptions,
    MinerUProtocolError,
    MinerUTaskFailedError,
    MinerUTaskStatus,
)


def _client(handler, **kwargs):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return MinerUClient(
        base_url="http://mineru.test",
        http_client=http,
        poll_interval=0,
        **kwargs,
    ), http


@pytest.mark.asyncio
async def test_health_and_submit_use_protocol_v2_structured_outputs():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "version": "3.4.4",
                    "protocol_version": 2,
                    "max_concurrent_requests": 1,
                },
            )
        return httpx.Response(
            202,
            json={
                "task_id": "task-123",
                "status": "pending",
                "backend": "pipeline",
                "file_names": ["sample.pdf"],
                "queued_ahead": 0,
            },
        )

    client, http = _client(handler)
    try:
        health = await client.health()
        submission = await client.submit_document(b"%PDF-test", "sample.pdf")
    finally:
        await http.aclose()

    assert health.status == "healthy"
    assert health.protocol_version == 2
    assert submission.task_id == "task-123"
    assert submission.status is MinerUTaskStatus.PENDING
    body = requests[1].content
    assert b'name="backend"' in body and b"pipeline" in body
    assert b'name="return_content_list"' in body and b"true" in body
    assert b'name="response_format_zip"' in body and b"true" in body
    assert b'filename="sample.pdf"' in body


@pytest.mark.asyncio
async def test_wait_for_completion_polls_until_completed():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = "pending" if calls == 1 else "processing" if calls == 2 else "completed"
        return httpx.Response(200, json={"task_id": "task-1", "status": status})

    client, http = _client(handler)
    try:
        task = await client.wait_for_completion("task-1", timeout=1)
    finally:
        await http.aclose()

    assert task.status is MinerUTaskStatus.COMPLETED
    assert calls == 3


@pytest.mark.asyncio
async def test_failed_task_raises_domain_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"task_id": "task-2", "status": "failed", "error": "out of memory"},
        )

    client, http = _client(handler)
    try:
        with pytest.raises(MinerUTaskFailedError, match="out of memory"):
            await client.wait_for_completion("task-2", timeout=1)
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_download_result_streams_valid_zip(tmp_path):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sample.md", "parsed")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tasks/task-3/result"
        return httpx.Response(
            200,
            content=buffer.getvalue(),
            headers={"content-type": "application/zip"},
        )

    client, http = _client(handler)
    destination = tmp_path / "result.zip"
    try:
        result = await client.download_result("task-3", destination)
    finally:
        await http.aclose()

    assert result == destination
    with zipfile.ZipFile(result) as archive:
        assert archive.read("sample.md") == b"parsed"


@pytest.mark.asyncio
async def test_unknown_task_status_is_rejected():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"task_id": "task-4", "status": "mystery"})

    client, http = _client(handler)
    try:
        with pytest.raises(MinerUProtocolError, match="unknown status"):
            await client.get_task("task-4")
    finally:
        await http.aclose()


def test_parse_options_validate_page_range():
    with pytest.raises(ValueError, match="end_page_id"):
        MinerUParseOptions(start_page_id=2, end_page_id=1).to_form_data()
