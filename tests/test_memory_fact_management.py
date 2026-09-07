"""用户事实更新、删除及用户隔离测试。"""
from __future__ import annotations

import uuid

import pytest

from app.memory.manager import (
    DuplicateUserFactError,
    build_memory_intent_prompt,
    delete_user_fact,
    evaluate_and_apply_memory,
    parse_episode_memory,
    parse_memory_operations,
    rank_episode_records,
    update_user_fact,
)
from backend.storage.postgres.models_memory import UserFact


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def first(self):
        if isinstance(self.value, list):
            return self.value[0] if self.value else None
        return self.value


class _Session:
    def __init__(self, *results):
        self.results = list(results)
        self.flushed = 0
        self.deleted = []
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.results.pop(0))

    async def flush(self):
        self.flushed += 1

    async def delete(self, record):
        self.deleted.append(record)


@pytest.mark.asyncio
async def test_update_user_fact_trims_and_updates_owned_record():
    user_id = uuid.uuid4()
    record = UserFact(id=uuid.uuid4(), user_id=user_id, fact="旧偏好")
    session = _Session(record, None)

    updated = await update_user_fact(
        session, user_id, record.id, "  用户喜欢详细回答  "
    )

    assert updated is record
    assert record.fact == "用户喜欢详细回答"
    assert session.flushed == 1


@pytest.mark.asyncio
async def test_update_user_fact_rejects_duplicate():
    user_id = uuid.uuid4()
    record = UserFact(id=uuid.uuid4(), user_id=user_id, fact="旧偏好")
    duplicate = UserFact(id=uuid.uuid4(), user_id=user_id, fact="已有偏好")
    session = _Session(record, duplicate)

    with pytest.raises(DuplicateUserFactError):
        await update_user_fact(session, user_id, record.id, "已有偏好")
    assert record.fact == "旧偏好"
    assert session.flushed == 0


@pytest.mark.asyncio
async def test_update_and_delete_hide_missing_or_other_users_records():
    user_id = uuid.uuid4()
    fact_id = uuid.uuid4()

    update_session = _Session(None)
    assert await update_user_fact(update_session, user_id, fact_id, "新内容") is None
    update_sql = str(update_session.statements[0])
    assert "user_facts.id" in update_sql and "user_facts.user_id" in update_sql

    delete_session = _Session(None)
    assert await delete_user_fact(delete_session, user_id, fact_id) is False
    assert delete_session.deleted == []
    delete_sql = str(delete_session.statements[0])
    assert "user_facts.id" in delete_sql and "user_facts.user_id" in delete_sql


@pytest.mark.asyncio
async def test_delete_user_fact_deletes_owned_record():
    user_id = uuid.uuid4()
    record = UserFact(id=uuid.uuid4(), user_id=user_id, fact="待删除")
    session = _Session(record)

    assert await delete_user_fact(session, user_id, record.id) is True
    assert session.deleted == [record]
    assert session.flushed == 1


def test_memory_intent_parser_accepts_valid_operations_and_rejects_unknown_ids():
    user_id = uuid.uuid4()
    record = UserFact(id=uuid.uuid4(), user_id=user_id, fact="用户喜欢详细回答")
    raw = f'''```json
    {{"operations":[
      {{"action":"update","fact_id":"{record.id}","fact":"用户喜欢简洁回答"}},
      {{"action":"delete","fact_id":"{uuid.uuid4()}","fact":null}},
      {{"action":"create","fact_id":null,"fact":"项目使用 FastAPI"}}
    ]}}
    ```'''

    operations = parse_memory_operations(raw, [record])

    assert [operation["action"] for operation in operations] == ["update", "create"]
    prompt = build_memory_intent_prompt("以后回答简洁些", [record])
    assert str(record.id) in prompt
    assert "普通问题" in prompt
    assert "密码/API Key" in prompt


def test_episode_parser_and_relevance_ranking():
    import json

    user_id = uuid.uuid4()
    raw = json.dumps({
        "operations": [],
        "episode": {
            "title": "修复 FastAPI 部署",
            "goal": "恢复 EasyRAG API",
            "outcome": "success",
            "summary": "修正数据库连接后部署成功",
            "lessons": "启动前先检查连接字符串",
            "unfinished": None,
        },
    }, ensure_ascii=False)
    episode = parse_episode_memory(raw)
    assert episode is not None and episode["outcome"] == "success"

    from backend.storage.postgres.models_memory import EpisodeMemory

    records = [
        EpisodeMemory(
            id=uuid.uuid4(), user_id=user_id, title="旅行安排", goal="预订酒店",
            outcome="success", summary="完成预订", lessons=None,
        ),
        EpisodeMemory(
            id=uuid.uuid4(), user_id=user_id, title="修复 FastAPI 部署",
            goal="恢复 API", outcome="success", summary="修正数据库连接",
            lessons="先检查连接字符串",
        ),
    ]
    assert rank_episode_records("FastAPI 数据库连接失败", records, limit=1)[0] is records[1]


@pytest.mark.asyncio
async def test_fast_llm_can_capture_episode_with_execution_context(monkeypatch):
    import json

    response = {
        "operations": [],
        "episode": {
            "title": "修复部署",
            "goal": "恢复 API",
            "outcome": "partial",
            "summary": "定位到数据库连接错误",
            "lessons": "先验证连接配置",
            "unfinished": "重新执行迁移",
        },
    }

    class FakeLLM:
        async def chat(self, messages, **kwargs):
            assert "助手结果" in messages[0]["content"]
            assert "数据库连接错误" in messages[0]["content"]
            return json.dumps(response, ensure_ascii=False)

    captured = {}

    async def fake_list(*args, **kwargs):
        return []

    async def fake_add_episode(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("app.llm.client.get_llm_client", lambda **kwargs: FakeLLM())
    monkeypatch.setattr("app.memory.manager.list_user_fact_records", fake_list)
    monkeypatch.setattr("app.memory.manager.add_episode_memory", fake_add_episode)
    message_id = 42

    changed = await evaluate_and_apply_memory(
        object(),
        uuid.uuid4(),
        "修复部署问题",
        uuid.uuid4(),
        assistant_response="发现数据库连接错误",
        source_message_id=message_id,
        execution={"steps": ["检查日志"]},
    )

    assert changed == 1
    assert captured["source_message_id"] == message_id
    assert captured["execution"] == {"steps": ["检查日志"]}


@pytest.mark.asyncio
async def test_fast_llm_can_create_update_and_delete_memory(monkeypatch):
    user_id = uuid.uuid4()
    first = UserFact(id=uuid.uuid4(), user_id=user_id, fact="用户喜欢详细回答")
    second = UserFact(id=uuid.uuid4(), user_id=user_id, fact="用户住在上海")
    response = {
        "operations": [
            {"action": "update", "fact_id": str(first.id), "fact": "用户喜欢简洁回答"},
            {"action": "delete", "fact_id": str(second.id), "fact": None},
            {"action": "create", "fact_id": None, "fact": "项目使用 FastAPI"},
        ]
    }

    class FakeLLM:
        async def chat(self, *args, **kwargs):
            import json
            return json.dumps(response, ensure_ascii=False)

    calls = []

    async def fake_list(*args, **kwargs):
        return [first, second]

    async def fake_update(_session, _user_id, fact_id, fact):
        calls.append(("update", fact_id, fact))
        return first

    async def fake_delete(_session, _user_id, fact_id):
        calls.append(("delete", fact_id, None))
        return True

    async def fake_add(_session, _user_id, fact, conversation_id):
        calls.append(("create", conversation_id, fact))
        return UserFact(id=uuid.uuid4(), user_id=user_id, fact=fact)

    monkeypatch.setattr("app.llm.client.get_llm_client", lambda **kwargs: FakeLLM())
    monkeypatch.setattr("app.memory.manager.list_user_fact_records", fake_list)
    monkeypatch.setattr("app.memory.manager.update_user_fact", fake_update)
    monkeypatch.setattr("app.memory.manager.delete_user_fact", fake_delete)
    monkeypatch.setattr("app.memory.manager.add_user_fact", fake_add)
    conversation_id = uuid.uuid4()

    changed = await evaluate_and_apply_memory(
        object(), user_id, "我不住上海了，回答简洁些，项目改用 FastAPI", conversation_id
    )

    assert changed == 3
    assert [call[0] for call in calls] == ["update", "delete", "create"]


@pytest.mark.asyncio
async def test_fast_llm_sees_normal_question_and_can_choose_no_change(monkeypatch):
    observed = {}

    class FakeLLM:
        async def chat(self, messages, **kwargs):
            observed["prompt"] = messages[0]["content"]
            return '{"operations":[{"action":"none","fact_id":null,"fact":null}]}'

    async def fake_list(*args, **kwargs):
        return []

    monkeypatch.setattr("app.llm.client.get_llm_client", lambda **kwargs: FakeLLM())
    monkeypatch.setattr("app.memory.manager.list_user_fact_records", fake_list)

    changed = await evaluate_and_apply_memory(
        object(), uuid.uuid4(), "民法典第10条是什么"
    )

    assert changed == 0
    assert "民法典第10条是什么" in observed["prompt"]
