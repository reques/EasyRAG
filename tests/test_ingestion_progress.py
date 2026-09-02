from __future__ import annotations

import uuid

import pytest

from backend.services.graph_service import extract_graph_from_chunks


class FakeSession:
    def add(self, _item) -> None:
        pass

    async def execute(self, *_args, **_kwargs):
        return []  # 库级去重预查：无已有数据


@pytest.mark.asyncio
async def test_graph_extraction_reports_before_and_after_each_chunk(monkeypatch):
    class FakeLLM:
        async def chat_json(self, _messages):
            return {"entities": [], "relations": []}

    monkeypatch.setattr("app.llm.client.get_llm_client", lambda: FakeLLM())
    events: list[tuple[int, int, str]] = []

    async def capture(current: int, total: int, message: str) -> None:
        events.append((current, total, message))

    await extract_graph_from_chunks(
        FakeSession(),
        uuid.uuid4(),
        [("too short", {"chunk_index": 0}), ("also short", {"chunk_index": 1})],
        "notes.md",
        progress_callback=capture,
    )

    assert [(current, total) for current, total, _ in events] == [
        (0, 2),
        (1, 2),
        (1, 2),
        (2, 2),
    ]
    assert events[-1][2] == "正在抽取知识图谱 2/2"
