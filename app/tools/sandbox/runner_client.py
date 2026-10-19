"""Backend client for the private sandbox runner API."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx

from app.core.config import get_settings


def runner_config() -> tuple[str, dict[str, str]]:
    settings = get_settings()
    url = settings.SANDBOX_RUNNER_URL.rstrip('/')
    token = settings.SANDBOX_RUNNER_TOKEN
    if not url or not token:
        raise RuntimeError('Sandbox runner is not configured')
    return url, {'X-Runner-Token': token}


def execute_python(code: str, inputs: list[dict[str, str]] | None = None) -> dict[str, Any]:
    url, headers = runner_config()
    with httpx.Client(base_url=url, headers=headers, timeout=70, trust_env=False) as client:
        response = client.post('/v1/python', json={'code': code, 'inputs': inputs or []})
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get('files'), list):
        raise ValueError('Invalid sandbox runner response')
    return result


class RunnerMcpSession:
    """MCP surface backed by a Docker container in the runner service."""

    def __init__(self, profile: str):
        self.profile = profile
        self.session_id: str | None = None
        self.tools: list[dict[str, Any]] = []
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> 'RunnerMcpSession':
        url, headers = runner_config()
        self.client = httpx.AsyncClient(base_url=url, headers=headers, timeout=60, trust_env=False)
        try:
            response = await self.client.post('/v1/mcp/sessions', json={'profile': self.profile})
            response.raise_for_status()
            data = response.json()
            self.session_id = str(data['session_id'])
            self.tools = list(data['tools'])
            return self
        except Exception:
            if self.session_id:
                try:
                    await self.client.delete(f'/v1/mcp/sessions/{self.session_id}')
                except Exception:
                    pass
            await self.client.aclose()
            self.client = None
            raise

    async def __aexit__(self, *_: Any) -> None:
        if self.client is not None:
            try:
                if self.session_id:
                    await self.client.delete(f'/v1/mcp/sessions/{self.session_id}')
            finally:
                await self.client.aclose()
                self.client = None
                self.session_id = None

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=[SimpleNamespace(**tool) for tool in self.tools])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.client is None or self.session_id is None:
            raise RuntimeError('MCP sandbox session is closed')
        response = await self.client.post(
            f'/v1/mcp/sessions/{self.session_id}/call',
            json={'tool': name, 'arguments': arguments},
        )
        response.raise_for_status()
        result = response.json()
        return SimpleNamespace(content=result['content'], isError=result['isError'])
