import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from minio.error import S3Error

from app.agents.events import use_request_trace
from app.services.artifact_context import use_artifact_owner
from app.tools.create_file import create_file
from app.tools.create_file import TOOL
from app.tools.registry import ToolRegistry
from backend.services import artifact_service as service
from backend.server.routers import artifact_router as routes


@pytest.fixture
def storage(monkeypatch):
    objects = {}

    class Body(io.BytesIO):
        def release_conn(self):
            pass

    class Store:
        def put_object(self, bucket, name, data, length, content_type, metadata):
            objects[name] = (data.read(), SimpleNamespace(
                object_name=name, size=length, content_type=content_type,
                metadata={'x-amz-meta-' + k: v for k, v in metadata.items()}))

        def stat_object(self, bucket, name):
            if name not in objects:
                raise S3Error('NoSuchKey', 'missing', '', '', '', None)
            return objects[name][1]

        def get_object(self, bucket, name):
            return Body(objects[name][0])

        def list_objects(self, bucket, prefix, **kwargs):
            return [value[1] for key, value in objects.items() if key.startswith(prefix)]

    monkeypatch.setattr(service, 'get_minio_client', lambda: Store())
    monkeypatch.setattr(service, 'ensure_bucket', lambda _: None)
    return objects


@pytest.mark.parametrize('filename', ['../x.md', 'a\\x.txt', 'bad\n.csv', '.hidden.txt', 'x.exe', 'x.txt.', 'a/b.html'])
def test_invalid_filename(filename):
    with pytest.raises(ValueError):
        service.prepare_file(filename, 'content')


def test_invalid_content():
    for filename, content in [('x.json', '{invalid'), ('x.txt', ''), ('x.md', 'x' * (service.MAX_BYTES + 1))]:
        with pytest.raises(ValueError):
            service.prepare_file(filename, content)


@pytest.mark.parametrize('filename,content', [
    ('report.md', '# Report'), ('notes.txt', 'notes'), ('table.csv', 'name,value\na,1'),
    ('data.json', '{"name": "中文"}'), ('page.html', '<h1>Report</h1>'),
])
def test_supported_formats(filename, content):
    data, mime = service.prepare_file(filename, content)
    assert data.decode('utf-8') == content
    assert mime == service.MIME_TYPES['.' + filename.rsplit('.', 1)[-1]]


def test_list_metadata_without_header_prefix():
    info = SimpleNamespace(object_name='artifacts/id', size=4, metadata={'filename': '%E6%8A%A5%E5%91%8A.md'})
    result = service.describe(info, 'conversation')
    assert result['filename'] == '报告.md'
    assert result['content_type'] == 'text/markdown'


def test_generate_persist_and_isolate(storage):
    user, conversation = uuid4(), uuid4()
    registry = ToolRegistry()
    registry.register(TOOL)
    with use_artifact_owner(user, conversation), use_request_trace() as trace:
        artifact = json.loads(registry.invoke('create_file', filename='分析报告.md', content='# 报告\n正文'))
    assert next(e for e in trace.events if e['metadata'].get('artifact'))['metadata']['artifact'] == artifact
    assert artifact['filename'] == '分析报告.md'
    assert service.list_files(user, conversation)[0]['filename'] == artifact['filename']
    assert service.list_files(uuid4(), conversation) == []
    info, data = service.read_file(user, conversation, artifact['id'])
    assert data.decode() == '# 报告\n正文'
    assert info['size'] == len(data)
    with pytest.raises(ValueError, match='authenticated'):
        create_file('x.md', 'hello')


def test_failed_storage_does_not_emit_file(monkeypatch):
    def fail(*args):
        raise RuntimeError('offline')
    monkeypatch.setattr(service, 'create_file', fail)
    with use_artifact_owner(uuid4(), uuid4()), use_request_trace() as trace:
        with pytest.raises(RuntimeError):
            create_file('x.md', 'hello')
    assert trace.events == []


def test_authenticated_download_and_conversation_ownership(storage, monkeypatch):
    user, conversation = uuid4(), uuid4()
    artifact = service.create_file(user, conversation, '报告.html', '<script>alert(1)</script>')

    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(routes, 'get_session', Session)
    monkeypatch.setattr(routes, 'get_conversation', AsyncMock(return_value=SimpleNamespace(user_id=user)))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    assert client.get(artifact['download_url']).status_code in (401, 403)
    app.dependency_overrides[routes.get_current_user] = lambda: SimpleNamespace(id=user)
    response = client.get(artifact['download_url'])
    assert response.status_code == 200
    assert response.content == b'<script>alert(1)</script>'
    assert response.headers['content-disposition'].startswith('attachment;')
    assert 'filename*=UTF-8' in response.headers['content-disposition']
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert len(client.get(f'/artifacts/{conversation}').json()['artifacts']) == 1
    assert client.get(artifact['download_url'] + '/preview').json()['kind'] == 'html'
    assert client.get(f'/artifacts/{conversation}/{uuid4()}').status_code == 404
    app.dependency_overrides[routes.get_current_user] = lambda: SimpleNamespace(id=uuid4())
    assert client.get(artifact['download_url']).status_code == 404
    assert client.get(f'/artifacts/{conversation}').status_code == 404
    assert client.get(artifact['download_url'] + '/preview').status_code == 404


@pytest.mark.parametrize('filename,content,preview_kind', [
    ('报告.doc', '# 标题', 'html'),
    ('表格.xls', '名称,金额\n甲,100', 'sheets'),
    ('报告.pdf', '# 标题\n\n内容', 'pdf'),
])
def test_office_generation_list_download_and_preview(storage, monkeypatch, filename, content, preview_kind):
    user, conversation = uuid4(), uuid4()
    artifact = service.create_file(user, conversation, filename, content)

    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(routes, 'get_session', Session)
    monkeypatch.setattr(routes, 'get_conversation', AsyncMock(return_value=SimpleNamespace(user_id=user)))
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_current_user] = lambda: SimpleNamespace(id=user)
    client = TestClient(app)
    assert artifact['filename'] == service.normalize_filename(filename)
    assert service.list_files(user, conversation)[0]['filename'] == artifact['filename']
    downloaded = client.get(artifact['download_url'])
    assert downloaded.content.startswith(b'%PDF-' if preview_kind == 'pdf' else b'PK')
    preview = client.get(artifact['download_url'] + '/preview')
    assert preview.status_code == 200
    assert preview.headers['cache-control'] == 'private, no-store'
    if preview_kind == 'pdf':
        assert preview.headers['content-type'] == 'application/pdf'
        assert preview.content == downloaded.content
    else:
        assert preview.json()['kind'] == preview_kind
    app.dependency_overrides[routes.get_current_user] = lambda: SimpleNamespace(id=uuid4())
    assert client.get(artifact['download_url'] + '/preview').status_code == 404
