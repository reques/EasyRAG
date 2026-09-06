"""Reasoning-only structured replies must not exhaust nested retry budgets."""
import asyncio
from types import SimpleNamespace

import pytest

from app.core.exceptions import LLMOutputParseError
from app.llm.client import LLMClient

RECRUITMENT_QUERY = (
    "无锡市热门重点企业！聚焦无人驾驶核心技术在智慧矿山、智慧海洋、智慧园区和轨道交通领域的解决方案，"
    "招募 SLAM、规控、图像、激光雷达感知等算法工程师。各位2027 届毕业的硕博生，这里有竞争力薪酬、"
    "导师 1V1 带教、国家重点项目历练。\n活力向上，智绘新未来，期待你的加入！（点击链接了解详情）"
)


def _response(content="", *, reasoning="", finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=reasoning),
            finish_reason=finish,
        )],
        usage=SimpleNamespace(completion_tokens=2048),
    )


def _client(monkeypatch, responder, model="deepseek-v4-flash"):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = object.__new__(LLMClient)
    client.model, client.temperature, client.max_tokens = model, 0.0, 2048
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responder(kwargs)

    async def acreate(**kwargs):
        return create(**kwargs)

    client._sync_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=acreate)))
    return client, calls


@pytest.mark.parametrize("use_async", [False, True])
def test_json_budget_is_for_output_not_hidden_thinking(monkeypatch, use_async):
    def respond(kwargs):
        if kwargs.get("extra_body", {}).get("thinking", {}).get("type") == "disabled":
            return _response('{"sub_questions":["付款期限？","违约责任？"]}')
        return _response(reasoning="model thinking" * 300, finish="length")

    client, calls = _client(monkeypatch, respond)
    messages = [{"role": "user", "content": "把付款期限和违约责任拆成 JSON 子问题"}]
    result = asyncio.run(client.chat_json(messages)) if use_async else client.chat_json_sync(messages)
    assert result["sub_questions"] == ["付款期限？", "违约责任？"]
    assert len(calls) == 1


@pytest.mark.parametrize("use_async", [False, True])
def test_empty_json_has_one_shared_retry_budget(monkeypatch, use_async):
    client, calls = _client(monkeypatch, lambda _: _response(reasoning="thinking", finish="length"))
    with pytest.raises(LLMOutputParseError):
        if use_async:
            asyncio.run(client.chat_json([{"role": "user", "content": "JSON"}]))
        else:
            client.chat_json_sync([{"role": "user", "content": "JSON"}])
    assert len(calls) == 2, "JSON parse and empty-body retries must not multiply"


def test_explicit_thinking_and_budget_are_preserved_without_mutating_input(monkeypatch):
    client, calls = _client(monkeypatch, lambda _: _response('{"ok":true}'))
    body = {"thinking": {"type": "enabled"}, "custom_option": True}
    client.chat_json_sync([{"role": "user", "content": "JSON"}], extra_body=body, max_tokens=16000)
    assert calls[0]["extra_body"] == body
    assert calls[0]["max_tokens"] == 16000
    assert body == {"thinking": {"type": "enabled"}, "custom_option": True}


def test_plain_chat_keeps_reasoning_and_its_budget(monkeypatch):
    client, calls = _client(monkeypatch, lambda _: _response("回答"))
    assert client.chat_sync([{"role": "user", "content": "解释问题"}]) == "回答"
    assert calls[0]["max_tokens"] == 2048
    assert "extra_body" not in calls[0]


def test_other_providers_do_not_receive_deepseek_parameters(monkeypatch):
    client, calls = _client(monkeypatch, lambda _: _response('{"ok":true}'), model="custom-model")
    client.chat_json_sync([{"role": "user", "content": "JSON"}])
    assert "thinking" not in calls[0].get("extra_body", {})
