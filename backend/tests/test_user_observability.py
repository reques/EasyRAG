"""User workspace authorization, remote isolation and real database judge lifecycle."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import uuid

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from backend.server.routers import observability_router as router
from backend.server.utils.auth_middleware import get_current_user
from backend.services import turn_judge
from backend.services.observation_reader import display_content, project_observations
from backend.services import observation_reader
from backend.storage.postgres.manager import Base
from backend.storage.postgres.models_user import Department, User
from backend.storage.postgres.models_conversation import WorkspaceSection, Workspace, Conversation, Message
from backend.storage.postgres.models_chat_stream import ChatStream
from backend.storage.postgres.models_trace import ExecutionTrace, TraceEvent
from backend.storage.postgres.models_observability import MessageFeedback, ObservabilityScore, TurnJudgement
from backend.storage.postgres.models_knowledge import KnowledgeBase, EvaluationRun, EvaluationDataset, EvaluationBenchmark


@asynccontextmanager
async def harness(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    tables = (Department, User, WorkspaceSection, Workspace, Conversation, Message, ChatStream, ExecutionTrace,
              TraceEvent, MessageFeedback, ObservabilityScore, TurnJudgement, KnowledgeBase,
              EvaluationRun, EvaluationDataset, EvaluationBenchmark)
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=[t.__table__ for t in tables]))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    users = [User(id=uuid.uuid4(), username=f'observe-{i}', hashed_password='unused') for i in range(2)]
    trace_ids = ['a'*16, 'b'*16, 'c'*16]
    now = datetime.now(timezone.utc)
    async with factory() as session:
        session.add_all(users)
        await session.flush()
        for i, trace_id in enumerate(trace_ids):
            user = users[0 if i < 2 else 1]
            conversation = Conversation(id=uuid.uuid4(), user_id=user.id, title=f'conversation-{i}')
            session.add(conversation)
            await session.flush()
            question = Message(conversation_id=conversation.id, role='user', content=f'question-{i}',
                metadata_json=json.dumps({'agent_id':'agent:a', 'agent_name':'Research', 'agent_snapshot':{'system_prompt':'NEVER-SHOW'}}))
            session.add(question)
            await session.flush()
            session.add(ExecutionTrace(id=trace_id, conversation_id=conversation.id, user_id=user.id,
                source_message_id=question.id, model_id='deepseek-v4-flash', mode='dynamic', status='completed',
                observability_trace_id=str(i+1)*32, started_at=now-timedelta(days=40 if i == 1 else 0), completed_at=now,
                total_tokens=100, duration_ms=500))
            answer = Message(conversation_id=conversation.id, role='assistant', content='answer '+str(i),
                metadata_json=json.dumps({'trace_id':trace_id}))
            session.add(answer)
            await session.flush()
            if i == 0:
                session.add(TraceEvent(id='event-one', trace_id=trace_id, sequence=1, type='tool_result', status='completed',
                    timestamp=now, output_json=json.dumps({'content':'evidence sk-test-secret'}),
                    metadata_json=json.dumps({'tool':'kb_search', 'token_usage':{'total_tokens':100}})))
                session.add(MessageFeedback(id=uuid.uuid4(), user_id=user.id, message_id=answer.id, rating='like', reason='Useful'))
        kb = KnowledgeBase(id=uuid.uuid4(), owner_id=users[0].id, name='owner kb', collection_name='test')
        session.add(kb)
        run_id = uuid.uuid4()
        session.add(EvaluationRun(id=run_id, name='experiment', knowledge_base_id=kb.id, metrics_json=json.dumps({
            'details':[{'question':'eval question', 'observability':{'experiment_id':str(run_id),
                'trace_id':uuid.uuid5(run_id, '0').hex, 'observation_id':'d'*16}}]})))
        await session.commit()
    owner = [users[0]]
    app = FastAPI()
    app.include_router(router.router)
    app.dependency_overrides[get_current_user] = lambda: owner[0]
    monkeypatch.setattr(router, 'get_session', factory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        yield client, factory, users, owner, trace_ids, run_id
    await engine.dispose()


@pytest.mark.asyncio
async def test_owner_list_detail_filter_totals_and_feedback(monkeypatch):
    async with harness(monkeypatch) as (client, _, _, owner, ids, __):
        current = (await client.get('/observability/traces')).json()
        assert current['total'] == 1 and current['items'][0]['id'] == ids[0]
        all_rows = (await client.get('/observability/traces?window=all&page_size=1')).json()
        assert all_rows['total'] == 2 and len(all_rows['items']) == 1
        summary = (await client.get('/observability/overview?window=all')).json()
        assert summary['total_tokens'] == 200 and summary['feedback']['like'] == 1
        assert summary['cost_usd'] is None
        assert (await client.get('/observability/traces?window=all&agent=agent:a&q=question-1')).json()['total'] == 1
        assert (await client.get('/observability/traces?q=%25')).json()['total'] == 0
        detail = await client.get('/observability/traces/'+ids[0])
        assert detail.status_code == 200 and detail.json()['answer'] == 'answer 0'
        assert detail.json()['events'][0]['metadata']['token_usage']['total_tokens'] == 100
        assert 'NEVER-SHOW' not in detail.text and 'sk-test-secret' not in detail.text
        assert (await client.get('/observability/traces/'+ids[2])).status_code == 404


@pytest.mark.asyncio
async def test_remote_lookup_is_never_called_for_another_owner(monkeypatch):
    calls = []
    async def read(trace_id, **kw):
        calls.append(trace_id)
        return {'state':'available','observations':[]}
    monkeypatch.setattr(router, 'read_observations', read)
    async with harness(monkeypatch) as (client, _, users, owner, ids, run_id):
        assert (await client.get('/observability/traces/'+ids[2]+'/observations')).status_code == 404
        assert not calls
        assert (await client.get('/observability/traces/'+ids[0]+'/observations')).status_code == 200
        assert calls == ['1'*32]
        assert (await client.get(f'/observability/evaluations/{run_id}/items/0')).status_code == 200
        assert calls[-1] == uuid.uuid5(run_id,'0').hex
        assert (await client.get(f'/observability/evaluations/{run_id}/items/-1')).status_code == 404
        owner[0] = users[1]
        assert (await client.get(f'/observability/evaluations/{run_id}/items/0')).status_code == 404
        assert len(calls) == 2


def test_remote_projection_secret_mask_content_mode_and_wrong_trace(monkeypatch):
    cfg = get_settings()
    monkeypatch.setattr(cfg, 'EASYRAG_OBSERVABILITY_CONTENT_MODE', 'redacted')
    raw = {'id':'a'*32, 'observations':[{'id':'b'*16, 'traceId':'a'*32, 'name':'llm.chat', 'type':'GENERATION',
        'input':{'api_key':'PRIVATE', 'messages':[{'role':'system','content':'HIDDEN-PROMPT'}]},
        'output':'answer sk-private-value', 'usageDetails':{'input':10}, 'metadata':{'password':'PRIVATE','durationMs':20}}]}
    projected = project_observations(raw, 'a'*32)
    assert 'PRIVATE' not in str(projected) and 'HIDDEN-PROMPT' not in str(projected) and 'sk-private-value' not in str(projected)
    assert projected['observations'][0]['token_usage']['input'] == 10
    monkeypatch.setattr(cfg, 'EASYRAG_OBSERVABILITY_CONTENT_MODE', 'metadata_only')
    assert project_observations(raw,'a'*32)['observations'][0]['output'] is None
    with pytest.raises(ValueError): project_observations(raw,'c'*32)
    assert len(display_content('x'*10000, limit=50)) == 50


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -1, 2, '0.9'])
def test_invalid_judge_scores_rejected(bad):
    with pytest.raises(ValueError):
        turn_judge.validate_result({'answer_relevance':bad, 'evidence_support':None,'explanation':'note'})


@pytest.mark.asyncio
async def test_judge_persistence_idempotency_outbox_and_concurrent_admission(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def invoke(profile, messages):
        calls.append(messages)
        started.set()
        await release.wait()
        return {'answer_relevance':.9,'evidence_support':.8,'explanation':'依据检索资料回答'}
    monkeypatch.setattr(turn_judge,'invoke_judge', invoke)
    from backend.server.routers import chat_router
    async def model(*args):
        from types import SimpleNamespace
        return SimpleNamespace(id='owner-model', name='Judge', model='test')
    monkeypatch.setattr(chat_router,'_resolve_request_model', model)
    async with harness(monkeypatch) as (client, factory, users, owner, ids, __):
        body = {'model_id':'owner-model','request_id':str(uuid.uuid4())}
        task = asyncio.create_task(client.post(f'/observability/traces/{ids[0]}/judge',json=body))
        await asyncio.wait_for(started.wait(), timeout=5)
        assert (await client.post(f'/observability/traces/{ids[0]}/judge',json=body)).status_code == 409
        release.set()
        response = await task
        assert response.status_code == 200, response.text
        assert response.json()['result']['answer_relevance'] == .9
        assert (await client.post(f'/observability/traces/{ids[0]}/judge',json=body)).status_code == 200
        assert len(calls) == 1 and 'sk-test-secret' not in str(calls)
        async with factory() as session:
            scores = (await session.execute(select(ObservabilityScore))).scalars().all()
            assert len(scores) == 2 and all(json.loads(s.payload_json)['trace_id'] == '1'*32 for s in scores)
        detail = (await client.get(f'/observability/traces/{ids[0]}')).json()
        assert detail['judgement']['result']['evidence_available'] and len(detail['scores']) == 2
        assert (await client.post(f'/observability/traces/{ids[0]}/judge',json={**body,'request_id':str(uuid.uuid4())})).status_code == 429
        owner[0] = users[1]
        assert (await client.post(f'/observability/traces/{ids[0]}/judge',json=body)).status_code == 404


@pytest.mark.asyncio
async def test_judge_failure_and_no_evidence_do_not_publish_fabricated_scores(monkeypatch):
    from backend.server.routers import chat_router
    async def model(*args):
        from types import SimpleNamespace
        return SimpleNamespace(id='owner-model', name='Judge', model='test')
    monkeypatch.setattr(chat_router,'_resolve_request_model', model)
    async def bad(*args): return {'answer_relevance':9,'evidence_support':.9,'explanation':'bad'}
    monkeypatch.setattr(turn_judge,'invoke_judge', bad)
    async with harness(monkeypatch) as (client, factory, _, __, ids, ___):
        body={'model_id':'owner-model','request_id':str(uuid.uuid4())}
        assert (await client.post(f'/observability/traces/{ids[1]}/judge',json=body)).status_code == 502
        async with factory() as session:
            row=await session.get(TurnJudgement,ids[1])
            assert row.state == 'failed'
            row.started_at = datetime.now(timezone.utc)-timedelta(seconds=130)
            row.state='running'; row.lease_until=row.started_at
            await session.commit()
        async def good(*args): return {'answer_relevance':.8,'evidence_support':.9,'explanation':'no tool evidence'}
        monkeypatch.setattr(turn_judge,'invoke_judge',good)
        result = await client.post(f'/observability/traces/{ids[1]}/judge',json={**body,'request_id':str(uuid.uuid4())})
        assert result.status_code == 200 and result.json()['result']['evidence_support'] is None


@pytest.mark.asyncio
async def test_v2_remote_query_is_bounded_scoped_parses_io_and_handles_failures(monkeypatch):
    cfg = get_settings()
    for key, value in {'LANGFUSE_ENABLED':True, 'LANGFUSE_BASE_URL':'http://langfuse:3000',
                       'LANGFUSE_PUBLIC_KEY':'pk-test', 'LANGFUSE_SECRET_KEY':'sk-test',
                       'EASYRAG_OBSERVABILITY_CONTENT_MODE':'redacted'}.items():
        monkeypatch.setattr(cfg, key, value)
    trace_id, user_id = 'a'*32, str(uuid.uuid4())
    status, payload, seen = [200], [{'data':[{'id':'b'*16, 'traceId':trace_id, 'userId':user_id,
        'input':'{"api_key":"PRIVATE","question":"hello"}', 'output':'"answer"',
        'metadata':{'durationMs':float('nan')}, 'usageDetails':{'input':10}}], 'meta':{}}], []
    def handler(request):
        seen.append(request)
        return httpx.Response(status[0], content=json.dumps(payload[0]).encode())
    original_client = httpx.AsyncClient
    monkeypatch.setattr(observation_reader.httpx, 'AsyncClient', lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    read = lambda: observation_reader.read_observations(trace_id, user_id=user_id,
        started_at='2026-11-01T02:00:00Z', completed_at='2026-11-01T02:10:00Z')
    reply = await read()
    assert reply['state'] == 'available' and reply['observations'][0]['output'] == 'answer'
    assert reply['observations'][0]['input']['api_key'] == '[REDACTED]'
    assert reply['observations'][0]['metadata']['durationMs'] is None
    params = seen[0].url.params
    assert seen[0].url.path == '/api/public/v2/observations'
    assert params['traceId'] == trace_id and params['userId'] == user_id and params['limit'] == '501'
    assert params['fromStartTime'] == '2026-11-01T01:55:00+00:00'
    assert params['toStartTime'] == '2026-11-01T03:10:00+00:00'
    payload[0] = {'data':[], 'meta':{}}
    assert (await read())['state'] == 'pending_index'
    status[0] = 401
    assert (await read())['state'] == 'misconfigured'
    status[0] = 503
    assert (await read())['state'] == 'unavailable'
    count = len(seen)
    assert (await observation_reader.read_observations('invalid'))['state'] == 'invalid_link'
    assert (await observation_reader.read_observations(trace_id, sampled=False))['state'] == 'not_sampled'
    assert len(seen) == count


@pytest.mark.asyncio
async def test_eval_persists_actual_execution_times(monkeypatch):
    from backend.services.evaluation_service import save_run
    async with harness(monkeypatch) as (_, factory, __, ___, ____, _____):
        metrics = {'hit_rate':1, 'mrr':1, 'avg_score':.8, 'details':[],
            'execution_started_at':'2026-11-01T02:00:00Z', 'execution_finished_at':'2026-11-01T02:30:00Z'}
        async with factory() as session:
            run = await save_run(session, 'actual duration', metrics, 3)
            assert run.finished_at-run.started_at == timedelta(minutes=30)
            assert run.started_at.hour == 2 and run.started_at.minute == 0
            from backend.server.routers.evaluation_router import _to_summary
            run.metrics_json = json.dumps({**metrics, 'details':[{'retrieval_error':'EmbeddingError'},{}]})
            assert _to_summary(run).failed_items == 1


@pytest.mark.asyncio
async def test_judge_rejects_unauthorized_model_before_any_call(monkeypatch):
    from fastapi import HTTPException
    from backend.server.routers import chat_router
    async def forbidden(*args): raise HTTPException(404, '模型不存在')
    async def never(*args): raise AssertionError('must not invoke model')
    monkeypatch.setattr(chat_router, '_resolve_request_model', forbidden)
    monkeypatch.setattr(turn_judge, 'invoke_judge', never)
    async with harness(monkeypatch) as (client, factory, _, __, ids, ___):
        reply=await client.post(f'/observability/traces/{ids[0]}/judge',json={'model_id':'foreign','request_id':str(uuid.uuid4())})
        assert reply.status_code == 404
        async with factory() as session:
            assert await session.get(TurnJudgement, ids[0]) is None
