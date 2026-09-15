from types import SimpleNamespace

from app.agents.events import (
    EVENT_TYPES,
    add_token_usage,
    emit,
    public_event,
    use_event_sink,
    use_request_trace,
    use_span,
)
from backend.services.trace_service import summarize_trace


def test_canonical_event_contract_and_tool_parenting():
    with use_request_trace("conversation-1") as trace:
        start = emit("agent", "agent_start", "开始", "用户问题", input="用户问题")
        call = emit("tool", "tool_start", "调用 search", '{"q":"x"}', tool="search")
        result = emit(
            "tool", "tool_end", "search 完成", "找到结果",
            tool="search", elapsed_ms=18,
        )
        final = emit(
            "agent", "final_response", "完成", "答案",
            output="答案", token_usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        )

    assert {event["type"] for event in trace.events} <= EVENT_TYPES
    assert call["parent_id"] == start["id"]
    assert result["parent_id"] == call["id"]
    assert final["parent_id"] == start["id"]
    assert set(public_event(call)) == {
        "id", "parent_id", "type", "status", "timestamp",
        "input", "output", "metadata", "trace_id",
    }


def test_file_and_code_tools_have_specific_types():
    with use_request_trace() as trace:
        emit("tool", "tool_start", "读取", "a.py", tool="read_file")
        emit("tool", "tool_start", "执行", "print(1)", tool="/not-really-python")

    assert trace.events[0]["type"] == "file_operation"
    assert trace.events[1]["type"] == "code_execution"


def test_token_usage_normalization_and_trace_summary():
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = set()
    message = SimpleNamespace(
        id="m1",
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        response_metadata={},
    )
    add_token_usage(usage, message, seen)
    add_token_usage(usage, message, seen)
    assert usage == {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}

    with use_request_trace() as trace:
        emit("agent", "agent_start", "开始")
        emit(
            "agent", "final_response", "完成", "ok",
            token_usage=usage, elapsed_ms=42,
        )
    summary = summarize_trace(trace.events)
    assert summary["status"] == "completed"
    assert summary["duration_ms"] == 42
    assert summary["total_tokens"] == 13


def test_progress_deltas_update_one_event_and_stream_full_snapshots():
    streamed = []
    with use_request_trace("conversation-1") as trace, use_event_sink(streamed.append):
        root = emit("agent", "agent_start", "开始")
        first = emit("artifact", "reason", "行动说明", "先", artifact_kind="thought", id="p1", streaming=True)
        emit("tool", "tool_start", "调用 search", "{}", tool="search")
        emit("artifact", "reason", "行动说明", "查知识库", artifact_kind="thought", id="p1", streaming=True)
        last = emit("artifact", "reason", "行动说明", "", artifact_kind="thought", id="p1", streaming=False)
        emit("artifact", "reason", "行动说明", "再核对来源", artifact_kind="thought", id="p2", streaming=False)

    progress = [event for event in trace.events if event["type"] == "reasoning_summary"]
    assert len(progress) == 2
    assert first["id"] == last["id"]
    assert first["output"] == "先"  # Previously sent snapshots stay immutable.
    assert last["output"] == last["content"] == "先查知识库"
    assert last["status"] == "completed"
    assert last["parent_id"] == root["id"]
    assert first["timestamp"] == last["timestamp"]
    assert progress[0] == last
    updates = [public_event(e) for e in streamed if e["id"] == first["id"]]
    assert [e["output"] for e in updates] == ["先", "先查知识库", "先查知识库"]
    assert all(e["metadata"]["stream_mode"] == "snapshot" for e in updates)


def test_progress_streams_are_isolated_between_spans_and_runs():
    def progress(text):
        return emit("artifact", "reason", "行动说明", text, artifact_kind="thought", id="p1", streaming=True)

    with use_request_trace() as outer:
        main = progress("主任务")
        with use_span("worker"):
            worker = progress("子任务")
        with use_request_trace() as inner:
            nested = progress("新请求")
        resumed = progress("继续")
    assert len(outer.events) == 2
    assert len(inner.events) == 1
    assert len({main["id"], worker["id"], nested["id"]}) == 3
    assert resumed["output"] == "主任务继续"
    assert worker["output"] == "子任务"


def test_last_call_context_is_separate_from_cumulative_usage():
    total, last_call, seen = {}, {}, set()
    first = SimpleNamespace(id="first", usage_metadata={"input_tokens": 10000, "output_tokens": 100, "total_tokens": 10100})
    second = SimpleNamespace(id="second", usage_metadata={"input_tokens": 12000, "output_tokens": 200, "total_tokens": 12200})
    with use_request_trace() as trace:
        add_token_usage(total, first, seen, last_call)
        add_token_usage(total, second, seen, last_call)
        add_token_usage(total, second, seen, last_call)
    assert total == {"input_tokens": 22000, "output_tokens": 300, "total_tokens": 22300}
    assert last_call == second.usage_metadata
    assert len(trace.events) == 2
    assert trace.events[0]["metadata"]["context_usage"] == first.usage_metadata
    assert trace.events[1]["metadata"]["token_usage"] == total
    assert trace.events[1]["metadata"]["context_usage"]["input_tokens"] == 12000


def test_context_usage_survives_trace_serialization():
    from datetime import datetime, timezone
    from backend.services.trace_service import serialize_trace
    trace = SimpleNamespace(
        id="trace", conversation_id="conversation", source_message_id=1, mode="dynamic", model_id="model",
        status="completed", input_tokens=22000, output_tokens=300, total_tokens=22300, duration_ms=500,
        started_at=datetime.now(timezone.utc), completed_at=None, events=[],
        metadata_json='{"context_usage":{"input_tokens":12000,"output_tokens":200,"total_tokens":12200}}',
    )
    payload = serialize_trace(trace)
    assert payload["token_usage"]["input_tokens"] == 22000
    assert payload["context_usage"]["input_tokens"] == 12000
