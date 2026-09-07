"""Agent checkpoint 持久化与运行配置测试。"""
from __future__ import annotations

from types import SimpleNamespace

from app.agents import checkpointing


def test_checkpoint_config_has_thread_namespace_and_limit():
    config = checkpointing.checkpoint_run_config("conversation-1", "dynamic", 20)
    assert config["configurable"] == {
        "thread_id": "dynamic:conversation-1",
    }
    assert config["recursion_limit"] == 20


def test_sqlite_checkpoint_survives_saver_recreation(tmp_path, monkeypatch):
    from langgraph.graph import END, START, StateGraph

    monkeypatch.setattr(
        checkpointing,
        "get_settings",
        lambda: SimpleNamespace(
            AGENT_CHECKPOINT_BACKEND="sqlite",
            AGENT_CHECKPOINT_PATH=str(tmp_path / "agent-state.sqlite3"),
        ),
    )
    checkpointing.reset_checkpointer_for_tests()
    config = checkpointing.checkpoint_run_config("thread-1", "test", 10)

    graph = StateGraph(dict)
    graph.add_node("write", lambda state: {**state, "saved": True})
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    app = graph.compile(checkpointer=checkpointing.get_agent_checkpointer())
    assert app.invoke({"value": 1}, config)["saved"] is True

    checkpointing.reset_checkpointer_for_tests()
    restored = checkpointing.get_agent_checkpointer().get_tuple(config)
    assert restored is not None
    assert restored.checkpoint["channel_values"]["__root__"]["saved"] is True
    assert checkpointing.begin_checkpoint_run("thread-1", "test") is True
    checkpointing.reset_checkpointer_for_tests()


def test_turn_message_id_is_stable_for_retry_and_changes_with_history():
    history = [{"role": "assistant", "content": "上一轮回答"}]
    first = checkpointing.checkpoint_turn_message_id("thread-1", "继续", history)

    assert first == checkpointing.checkpoint_turn_message_id(
        "thread-1", "继续", list(history)
    )
    assert first != checkpointing.checkpoint_turn_message_id(
        "thread-1",
        "继续",
        history + [{"role": "assistant", "content": "新回答"}],
    )


def test_pending_checkpoint_resumes_from_next_node(tmp_path, monkeypatch):
    from langgraph.graph import END, START, StateGraph

    monkeypatch.setattr(
        checkpointing,
        "get_settings",
        lambda: SimpleNamespace(
            AGENT_CHECKPOINT_BACKEND="sqlite",
            AGENT_CHECKPOINT_PATH=str(tmp_path / "resume.sqlite3"),
        ),
    )
    checkpointing.reset_checkpointer_for_tests()
    calls = []

    def first(state):
        calls.append("first")
        return {**state, "first_done": True}

    def second(state):
        calls.append("second")
        return {**state, "second_done": True}

    graph = StateGraph(dict)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    app = graph.compile(
        checkpointer=checkpointing.get_agent_checkpointer(),
        interrupt_after=["first"],
    )
    config = checkpointing.checkpoint_run_config("thread-1", "dynamic", 10)
    app.invoke({"value": 1}, config)

    graph_input, _count, resumed = checkpointing.prepare_checkpoint_input(
        app, config, [], resume=True
    )
    result = app.invoke(graph_input, config)

    assert resumed is True
    assert result["second_done"] is True
    assert calls == ["first", "second"]
    checkpointing.reset_checkpointer_for_tests()


def test_checkpoint_message_baseline_counts_replacements_without_duplication():
    existing = SimpleNamespace(id="context:user-facts")
    snapshot = SimpleNamespace(values={"messages": [existing]}, next=())
    agent = SimpleNamespace(get_state=lambda _config: snapshot)

    graph_input, processed, resumed = checkpointing.prepare_checkpoint_input(
        agent,
        {"configurable": {"thread_id": "dynamic:t"}},
        [
            {"role": "system", "id": "context:user-facts", "content": "新事实"},
            {"role": "user", "id": "turn:2", "content": "继续"},
        ],
    )

    assert graph_input is not None
    assert processed == 2
    assert resumed is False
