"""DeepSeek 思考模式多轮回传（S8）测试。

2026-08-21 修复：langchain-openai 1.4.1 的消息转换会丢弃 DeepSeek
reasoning 模型的 ``reasoning_content``，导致 create_react_agent 多轮工具
调用报 400（``The reasoning_content in the thinking mode must be passed
back to the API``）。``DeepSeekChatOpenAI`` 响应侧保存、请求侧回传该字段。

全部本地构造，不调网络。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage


def _make_model():
    from app.agents.deep.llm import DeepSeekChatOpenAI

    return DeepSeekChatOpenAI(
        model="deepseek-test",
        api_key="k",
        base_url="http://localhost:1",
        max_tokens=100,
    )


def _make_response(reasoning_content="思考过程"):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-test",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "回答",
                **({"reasoning_content": reasoning_content}
                   if reasoning_content else {}),
            },
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_model_is_chatopenai_subclass():
    from langchain_openai import ChatOpenAI

    assert isinstance(_make_model(), ChatOpenAI)


def test_request_payload_passes_back_reasoning_content():
    m = _make_model()
    msgs = [
        HumanMessage(content="你好"),
        AIMessage(content="思考后回答", additional_kwargs={"reasoning_content": "内部思考"}),
    ]
    payload = m._get_request_payload(msgs)
    assistant = [d for d in payload["messages"] if d["role"] == "assistant"][0]
    assert assistant["reasoning_content"] == "内部思考"
    assert assistant["content"] == "思考后回答"


def test_request_payload_plain_messages_unchanged():
    m = _make_model()
    payload = m._get_request_payload([HumanMessage(content="你好")])
    assert payload["messages"][0]["role"] == "user"
    assert "reasoning_content" not in payload["messages"][0]


def test_request_payload_with_tool_calls_keeps_reasoning():
    m = _make_model()
    msgs = [
        HumanMessage(content="查一下"),
        AIMessage(
            content="",
            additional_kwargs={
                "reasoning_content": "先检索",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "kb_search", "arguments": '{"query": "x"}'},
                }],
            },
        ),
    ]
    payload = m._get_request_payload(msgs)
    assistant = [d for d in payload["messages"] if d["role"] == "assistant"][0]
    assert assistant["reasoning_content"] == "先检索"
    assert assistant["tool_calls"], "工具调用轮必须保留 tool_calls"


def test_create_chat_result_saves_reasoning_content():
    m = _make_model()
    result = m._create_chat_result(_make_response())
    gen = result.generations[0].message
    assert gen.additional_kwargs.get("reasoning_content") == "思考过程"


def test_create_chat_result_without_reasoning_unchanged():
    m = _make_model()
    result = m._create_chat_result(_make_response(reasoning_content=None))
    gen = result.generations[0].message
    assert "reasoning_content" not in gen.additional_kwargs


def test_extract_reasoning_content_from_openai_model():
    """openai.BaseModel 响应（SDK 解析的 DeepSeek 响应）也能提取。"""
    from openai.types.chat import ChatCompletion

    from app.agents.deep.llm import _extract_reasoning_content

    raw = _make_response()
    obj = ChatCompletion.model_validate(raw)
    assert _extract_reasoning_content(obj) == "思考过程"
    assert _extract_reasoning_content(_make_response(reasoning_content=None)) is None


def test_get_langchain_model_returns_adapter():
    from app.agents.deep.llm import DeepSeekChatOpenAI, get_langchain_model

    model = get_langchain_model.__wrapped__(
        model_name="deepseek-test-2", temperature=0.1
    )
    assert isinstance(model, DeepSeekChatOpenAI)
    assert model.model_name == "deepseek-test-2"
