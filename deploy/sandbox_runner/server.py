"""Private Docker control plane for disposable code and MCP containers.

This service is reachable only on the compose control network. Requests select
fixed profiles; callers cannot supply images, mounts, Docker flags or commands.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import hmac
import json
import os
import subprocess
import threading
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator


TOKEN = os.environ.get('SANDBOX_RUNNER_TOKEN', '')
PYTHON_IMAGE = os.environ.get('SANDBOX_PYTHON_IMAGE', 'easyrag/sandbox-python:local')
FILESYSTEM_IMAGE = os.environ.get('SANDBOX_FILESYSTEM_IMAGE', 'easyrag/mcp-filesystem:local')
POSTGRES_IMAGE = os.environ.get('SANDBOX_POSTGRES_IMAGE', 'easyrag/mcp-postgres:local')
RAIL_IMAGE = os.environ.get('SANDBOX_RAIL_IMAGE', 'easyrag/mcp-12306:local')
WORKSPACE_VOLUME = os.environ.get('SANDBOX_WORKSPACE_VOLUME', 'easyrag-sandbox-workspace')
POSTGRES_READONLY_URL = os.environ.get('SANDBOX_POSTGRES_READONLY_URL', '')
POSTGRES_NETWORK = os.environ.get('SANDBOX_POSTGRES_NETWORK', 'easyrag-sandbox-db')
EGRESS_NETWORK = os.environ.get('SANDBOX_EGRESS_NETWORK', 'easyrag-sandbox-egress')
PYTHON_SLOTS = threading.BoundedSemaphore(4)
MCP_SLOTS = threading.BoundedSemaphore(8)


def authorize(x_runner_token: str = Header(default='')) -> None:
    if not TOKEN or not hmac.compare_digest(x_runner_token, TOKEN):
        raise HTTPException(403, 'Forbidden')


def docker_base(name: str, image: str, *, memory: str = '256m') -> list[str]:
    return [
        'docker', 'run', '--rm', '--interactive', '--name', name, '--pull', 'never',
        '--network', 'none', '--memory', memory, '--cpus', '1', '--pids-limit', '64',
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--user', '10001:10001', '--env', 'HOME=/tmp',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=32m,mode=1777',
        image,
    ]


def remove_container(name: str) -> None:
    subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10, check=False)


def remove_orphans() -> None:
    """Recover containers left by a runner process killed without lifespan cleanup."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=10, check=False,
        )
        for name in result.stdout.splitlines():
            if name.startswith(('easyrag-py-', 'easyrag-mcp-')):
                remove_container(name)
    except (OSError, subprocess.TimeoutExpired):
        pass


class InputFile(BaseModel):
    filename: str = Field(min_length=1, max_length=180)
    content_b64: str = Field(max_length=7_000_000)


class PythonRequest(BaseModel):
    code: str = Field(min_length=1, max_length=262144)
    inputs: list[InputFile] = Field(default_factory=list, max_length=3)

    @model_validator(mode='after')
    def limit_request(self) -> 'PythonRequest':
        if sum(len(item.content_b64) for item in self.inputs) > 14_000_000:
            raise ValueError('Input artifacts exceed request budget')
        return self


def execute_python(request: PythonRequest) -> dict[str, Any]:
    if not PYTHON_SLOTS.acquire(blocking=False):
        raise HTTPException(429, 'Python sandbox capacity reached')
    try:
        return _execute_python_locked(request)
    finally:
        PYTHON_SLOTS.release()


def _execute_python_locked(request: PythonRequest) -> dict[str, Any]:
    name = f'easyrag-py-{uuid4().hex}'
    command = docker_base(name, PYTHON_IMAGE)
    command[-1:-1] = [
        '--tmpfs', '/work:rw,nosuid,nodev,size=16m,mode=1777',
        '--tmpfs', '/output:rw,nosuid,nodev,size=16m,mode=1777',
    ]
    try:
        completed = subprocess.run(
            command, input=(json.dumps(request.model_dump()) + '\n').encode(),
            capture_output=True, timeout=60, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, 'Sandbox timed out') from exc
    except OSError as exc:
        raise HTTPException(503, 'Docker runner unavailable') from exc
    finally:
        try:
            remove_container(name)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if completed.returncode:
        raise HTTPException(503, 'Sandbox process failed')
    try:
        result = json.loads(completed.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(502, 'Invalid sandbox response') from exc
    if not isinstance(result, dict) or not isinstance(result.get('files'), list):
        raise HTTPException(502, 'Invalid sandbox response')
    return result


class McpCreate(BaseModel):
    profile: str = Field(pattern='^(filesystem|postgres|rail12306)$')


class McpCall(BaseModel):
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


class McpSession:
    def __init__(self, name: str, command: list[str]):
        if not MCP_SLOTS.acquire(blocking=False):
            raise HTTPException(429, 'MCP sandbox capacity reached')
        self.name = name
        try:
            self.queue: asyncio.Queue[tuple[str, Any, asyncio.Future[Any]]] = asyncio.Queue()
            self.ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self.task = asyncio.create_task(self._serve(command))
        except Exception:
            MCP_SLOTS.release()
            raise

    async def _serve(self, command: list[str]) -> None:
        """Own the MCP context from enter through exit in this one task."""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(stdio_client(
                    StdioServerParameters(command='docker', args=command[1:])
                ))
                client = await stack.enter_async_context(ClientSession(*streams))
                await asyncio.wait_for(client.initialize(), timeout=20)
                tools = await asyncio.wait_for(client.list_tools(), timeout=20)
                if not self.ready.done():
                    self.ready.set_result(tools)
                while True:
                    action, payload, future = await self.queue.get()
                    if action == 'stop':
                        if not future.done():
                            future.set_result(None)
                        break
                    try:
                        result = await client.call_tool(payload.tool, payload.arguments)
                        if not future.done():
                            future.set_result(result)
                    except Exception as exc:
                        if not future.done():
                            future.set_exception(exc)
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            while not self.queue.empty():
                _, _, future = self.queue.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError('MCP sandbox stopped'))
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            await asyncio.to_thread(remove_container, self.name)
            MCP_SLOTS.release()

    async def call(self, request: McpCall) -> Any:
        if self.task.done():
            raise RuntimeError('MCP sandbox stopped')
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(('call', request, future))
        return await asyncio.wait_for(future, timeout=45)

    async def close(self) -> None:
        try:
            if not self.task.done():
                future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
                self.queue.put_nowait(('stop', None, future))
                try:
                    await asyncio.wait_for(future, timeout=10)
                except asyncio.TimeoutError:
                    self.task.cancel()
            try:
                await asyncio.wait_for(self.task, timeout=10)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                self.task.cancel()
        finally:
            await asyncio.to_thread(remove_container, self.name)


sessions: dict[str, McpSession] = {}
sessions_lock = threading.Lock()


def mcp_docker_command(profile: str, name: str) -> list[str]:
    if profile == 'filesystem':
        command = docker_base(name, FILESYSTEM_IMAGE)
        command[-1:-1] = [
            '--mount', f'type=volume,source={WORKSPACE_VOLUME},target=/workspace,readonly',
        ]
        return command
    if profile == 'postgres':
        if not POSTGRES_READONLY_URL:
            raise HTTPException(503, 'Read-only Postgres credentials are not configured')
        command = docker_base(name, POSTGRES_IMAGE)
        command[command.index('--network') + 1] = POSTGRES_NETWORK
        return command + ['mcp-server-postgres', POSTGRES_READONLY_URL]
    if profile == 'rail12306':
        command = docker_base(name, RAIL_IMAGE)
        command[command.index('--network') + 1] = EGRESS_NETWORK
        command[-1:-1] = [
            '--env', 'HTTPS_PROXY=http://sandbox-egress-proxy:8080',
            '--env', 'HTTP_PROXY=http://sandbox-egress-proxy:8080',
            '--env', 'NODE_USE_ENV_PROXY=1',
            '--env', 'NO_PROXY=',
            '--env', 'no_proxy=',
        ]
        return command
    raise HTTPException(422, 'Unsupported MCP profile')


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(remove_orphans)
    yield
    with sessions_lock:
        remaining = list(sessions.values())
        sessions.clear()
    for item in remaining:
        await item.close()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.get('/health')
def health() -> dict[str, bool]:
    if not TOKEN:
        raise HTTPException(503, 'Runner token is not configured')
    try:
        result = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'],
                                capture_output=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(503, 'Docker daemon unavailable') from exc
    if result.returncode:
        raise HTTPException(503, 'Docker daemon unavailable')
    return {'ok': True}


@app.post('/v1/python', dependencies=[Depends(authorize)])
async def run_python(request: PythonRequest) -> dict[str, Any]:
    return await asyncio.to_thread(execute_python, request)


@app.post('/v1/mcp/sessions', dependencies=[Depends(authorize)])
async def create_mcp_session(request: McpCreate) -> dict[str, Any]:
    name = f'easyrag-mcp-{uuid4().hex}'
    command = mcp_docker_command(request.profile, name)
    item = McpSession(name, command)
    try:
        tools = await asyncio.wait_for(item.ready, timeout=45)
    except BaseException as exc:
        await item.close()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(503, f'MCP sandbox failed: {type(exc).__name__}') from exc
    session_id = uuid4().hex
    with sessions_lock:
        sessions[session_id] = item
    return {'session_id': session_id, 'tools': [
        {'name': tool.name, 'description': tool.description or '',
         'input_schema': getattr(tool, 'input_schema', None) or getattr(tool, 'inputSchema', {})}
        for tool in tools.tools
    ]}


def get_session(session_id: str) -> McpSession:
    with sessions_lock:
        item = sessions.get(session_id)
    if item is None:
        raise HTTPException(404, 'MCP session not found')
    return item


@app.post('/v1/mcp/sessions/{session_id}/call', dependencies=[Depends(authorize)])
async def call_mcp(session_id: str, request: McpCall) -> dict[str, Any]:
    item = get_session(session_id)
    try:
        result = await item.call(request)
    except asyncio.TimeoutError as exc:
        with sessions_lock:
            sessions.pop(session_id, None)
        await item.close()
        raise HTTPException(504, 'MCP tool timed out') from exc
    except Exception as exc:
        raise HTTPException(502, f'MCP tool failed: {type(exc).__name__}') from exc
    content = []
    remaining = 64 * 1024
    for part in result.content:
        if getattr(part, 'type', None) != 'text' or remaining <= 0:
            continue
        value = str(getattr(part, 'text', ''))[:remaining]
        content.append({'type': 'text', 'text': value})
        remaining -= len(value)
    return {'isError': bool(getattr(result, 'is_error', getattr(result, 'isError', False))),
            'content': content}


@app.delete('/v1/mcp/sessions/{session_id}', dependencies=[Depends(authorize)])
async def delete_mcp(session_id: str) -> dict[str, bool]:
    with sessions_lock:
        item = sessions.pop(session_id, None)
    if item is not None:
        await item.close()
    return {'closed': True}
