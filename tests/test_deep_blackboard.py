"""阶段 3：结构化黑板（deep/blackboard.py）— post/读取/版本/两级数据/事件。"""
from __future__ import annotations

from app.agents.deep.blackboard import Blackboard
from app.agents.events import use_request_trace


def test_post_and_get():
    board = Blackboard()
    art = board.post("task_a", producer="research-agent",
                     summary="研究结论摘要", data={"full": "全量结果"})
    got = board.get("task_a")
    assert got is art
    assert got.producer == "research-agent"
    assert got.summary == "研究结论摘要"
    assert got.data == {"full": "全量结果"}  # 全量可从 data 取（两级）
    assert got.version == 1


def test_missing_key_returns_none():
    assert Blackboard().get("nope") is None


def test_overwrite_increments_version():
    board = Blackboard()
    board.post("k", producer="p", summary="v1")
    art2 = board.post("k", producer="p", summary="v2")
    assert art2.version == 2
    assert board.get("k").summary == "v2"


def test_summary_truncated_to_limit():
    board = Blackboard()
    art = board.post("k", producer="p", summary="x" * 2000)
    assert len(art.summary) == 500


def test_keys_lists_all_artifacts():
    board = Blackboard()
    board.post("b", producer="p", summary="s")
    board.post("a", producer="p", summary="s")
    assert sorted(board.keys()) == ["a", "b"]


def test_render_for_injection_uses_summaries():
    """依赖注入文本：[producer/key] summary 格式，缺失的 key 跳过。"""
    board = Blackboard()
    board.post("task_a", producer="research-agent", summary="A 的结论")
    board.post("task_b", producer="coding-agent", summary="B 的结论")
    text = board.render_for_injection(["task_a", "task_b", "missing"])
    assert "[research-agent/task_a] A 的结论" in text
    assert "[coding-agent/task_b] B 的结论" in text
    assert "missing" not in text


def test_render_for_injection_empty_when_no_deps():
    board = Blackboard()
    assert board.render_for_injection([]) == ""
    assert board.render_for_injection(["not_there"]) == ""


def test_post_emits_event_into_trace():
    """写通知：post 发出 blackboard/post 事件（供前端实时展示）。"""
    board = Blackboard()
    with use_request_trace(session_id="s") as rt:
        board.post("task_a", producer="research-agent", summary="结论",
                   tags=("research",))
    events = [e for e in rt.events if e["kind"] == "blackboard"]
    assert len(events) == 1
    ev = events[0]
    assert ev["stage"] == "post"
    assert ev["key"] == "task_a"
    assert ev["producer"] == "research-agent"
    assert ev["version"] == 1
    assert ev["tags"] == ["research"]


def test_post_noop_event_without_trace():
    """无 trace 上下文时 post 正常工作（事件 no-op）。"""
    board = Blackboard()
    art = board.post("k", producer="p", summary="s")
    assert art.key == "k"
