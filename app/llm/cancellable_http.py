"""Interrupt model HTTP I/O without closing clients shared by other runs.

Sync OpenAI/LangChain callers use one process-local async I/O loop. Both waiting
for headers and reading a silent stream are cancellable, even before the model
emits its first content token. Cancellation closes this response's connection;
it is control flow (BaseException), so SDK retries cannot restart the request.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Event, Lock, Thread
from typing import Any, Coroutine

import httpx

from app.agents.cancellation import AgentRunCancelled, current_cancel_event

POLL_SECONDS = 0.02
_loop = None
_loop_lock = Lock()


def _run(coro: Coroutine[Any, Any, Any]):
    global _loop
    with _loop_lock:
        if _loop is None:
            ready = Event()
            def serve():
                global _loop
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                ready.set()
                _loop.run_forever()
            Thread(target=serve, name="model-http-io", daemon=True).start()
            ready.wait()
    future: Future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


async def _interruptible(awaitable, signal):
    if signal is None:
        return await awaitable
    operation = asyncio.ensure_future(awaitable)
    try:
        while True:
            if signal.is_set():
                raise AgentRunCancelled()
            if operation.done():
                return operation.result()
            await asyncio.wait((operation,), timeout=POLL_SECONDS)
    except BaseException:
        operation.cancel()
        result, = await asyncio.gather(operation, return_exceptions=True)
        # Headers may have arrived in the same tick as cancellation. Don't lose
        # the returned response and leave its pool connection checked out.
        if isinstance(result, httpx.Response):
            await result.aclose()
        raise


def _copy_request(request):
    return httpx.Request(
        request.method, request.url, headers=request.headers,
        content=request.read(), extensions=dict(request.extensions),
    )


def _response(original, stream):
    return httpx.Response(
        original.status_code, headers=original.headers,
        stream=stream, extensions=original.extensions,
    )


class _SyncBody(httpx.SyncByteStream):
    def __init__(self, response, signal):
        self.response, self.signal = response, signal
        self.closed = False

    def __iter__(self):
        source = self.response.aiter_raw()
        try:
            while True:
                try:
                    yield _run(_interruptible(anext(source), self.signal))
                except StopAsyncIteration:
                    return
        finally:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            _run(self.response.aclose())


class CancellableHTTPTransport(httpx.BaseTransport):
    def __init__(self, *, trust_env=False):
        self.trust_env = trust_env
        self.client = None

    async def _send(self, request, signal):
        # Accessed exclusively on the dedicated I/O loop; pool reuse stays safe
        # for simultaneous requests and doesn't depend on callers' event loops.
        if self.client is None:
            self.client = httpx.AsyncClient(trust_env=self.trust_env, timeout=None)
        return await _interruptible(self.client.send(request, stream=True), signal)

    def handle_request(self, request):
        signal = current_cancel_event()
        original = _run(self._send(_copy_request(request), signal))
        return _response(original, _SyncBody(original, signal))

    def close(self):
        if self.client is not None:
            _run(self.client.aclose())


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self, response, signal):
        self.response, self.signal = response, signal

    async def __aiter__(self):
        source = self.response.aiter_raw()
        try:
            while True:
                try:
                    yield await _interruptible(anext(source), self.signal)
                except StopAsyncIteration:
                    return
        finally:
            await self.aclose()

    async def aclose(self):
        await self.response.aclose()


class CancellableAsyncHTTPTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, trust_env=False):
        self.client = httpx.AsyncClient(trust_env=trust_env, timeout=None)

    async def handle_async_request(self, request):
        signal = current_cancel_event()
        original = await _interruptible(self.client.send(request, stream=True), signal)
        return _response(original, _AsyncBody(original, signal))

    async def aclose(self):
        await self.client.aclose()


def sync_model_http_client(*, trust_env=False):
    # Proxy selection belongs to the inner client, otherwise httpx environment
    # mounts would bypass our transport and silently remove cancellation support.
    return httpx.Client(transport=CancellableHTTPTransport(trust_env=trust_env), trust_env=False)


def async_model_http_client(*, trust_env=False):
    return httpx.AsyncClient(transport=CancellableAsyncHTTPTransport(trust_env=trust_env), trust_env=False)
