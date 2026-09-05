"""Public progress is incremental; tool preambles and protocol tags are not answers."""
import pytest

from app.agents.response_stream import ProgressEventCollector, ResponseStream, message_text


@pytest.mark.parametrize("size", [1, 2, 7, 1000])
def test_split_delimiters_stream_public_progress_and_answer(size):
    progress, answer = [], []
    parser = ResponseStream(lambda text, done: progress.append((text, done)), answer.append)
    source = "<progress>先核对合同中的付款期限。</progress>\n<answer>**期限**为 30 天。</answer>"
    for start in range(0, len(source), size):
        parser.feed(source[start:start + size])
    # Visible before the completed values message, with no raw delimiters.
    assert "".join(text for text, _ in progress) == "先核对合同中的付款期限。"
    assert "".join(answer) == "**期限**为 30 天。"
    assert parser.finish() == "**期限**为 30 天。"
    assert progress[-1] == ("", True)


@pytest.mark.parametrize("calls", [True, False])
def test_untagged_text_waits_for_tool_call_classification(calls):
    progress, answer = [], []
    parser = ResponseStream(lambda text, done: progress.append(text), answer.append)
    parser.feed("查询逾期付款的条款")
    assert not progress and not answer
    final = parser.finish(tool_calls=calls)
    assert "".join(progress if calls else answer) == "查询逾期付款的条款"
    assert final == ("" if calls else "查询逾期付款的条款")
    assert not (answer if calls else progress)


def test_missing_closing_tag_and_literal_angle_brackets():
    answer = []
    parser = ResponseStream(lambda *_: None, answer.append)
    parser.feed("<answer>判断 x < 3，保留 `<progress>` 示例")
    assert parser.finish() == "判断 x < 3，保留 `<progress>` 示例"
    assert "".join(answer) == parser.answer


def test_content_blocks_exclude_reasoning_and_tool_arguments():
    assert message_text([
        {"type": "reasoning", "text": "provider private reasoning"},
        {"type": "text", "text": "公开"},
        {"type": "tool_use", "input": {"query": "tool parameter"}},
        {"type": "output_text", "text": "说明"},
    ]) == "公开说明"


def test_collector_replays_interleaved_stream_as_one_summary_and_excludes_answers():
    collector = ProgressEventCollector()
    first = collector.artifact({"id": "p1", "kind": "thought", "content": "先核对", "streaming": True})
    action = collector.artifact({"kind": "tool", "content": "{}", "tool_call_id": "c1"})
    collector.artifact({"id": "p1", "kind": "thought", "content": "付款期限。", "streaming": True})
    collector.artifact({"id": "p1", "kind": "thought", "content": "", "streaming": False})
    observed = collector.artifact({"kind": "tool_result", "content": "30 天", "tool_call_id": "c1"})
    collector.step("generate", "正在组织回答")
    collector.artifact({"kind": "answer", "content": "30 天", "streaming": True})
    collector.artifact({"kind": "answer", "content": "", "streaming": False})
    assert first["sequence"] < action["sequence"] < observed["sequence"]
    assert len(collector.artifacts) == 3
    assert collector.artifacts[0]["content"] == "先核对付款期限。"
    assert collector.artifacts[0]["streaming"] is False
    assert collector.artifacts[0]["sequence"] == first["sequence"]
    assert collector.artifacts[1]["tool_call_id"] == collector.artifacts[2]["tool_call_id"]
    assert collector.answer_streamed
