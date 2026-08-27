"""OpenTelemetry 可选集成（2026-08-26，阶段 5）。

- 安装了 ``opentelemetry-api`` → ``trace_span``/``get_tracer`` 返回真实
  tracer（需自行配置 exporter/TracerProvider 才能导出）；
- 未安装 → no-op 等价物：``trace_span`` 直接 yield None，``instrument_app``
  原样返回应用，零依赖零开销。

接入点（见各调用方）：
- ``registry.invoke``      → span ``tool.invoke.<name>``
- ``task`` 工具            → span ``subagent.<type>``
- ``spawn_tasks`` 调度     → span ``spawn_tasks`` / ``spawn_task.<key>``
- ASGI                     → ``instrument_app`` 请求根 span
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:  # opentelemetry-api 为可选依赖（未安装 → 全部 no-op）
    from opentelemetry import trace as _otel_trace

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境是否安装
    _otel_trace = None
    OTEL_AVAILABLE = False

_TRACER_NAME = "easyrag"


class _NoOpSpan:
    """未安装 OpenTelemetry 时的占位 span（上下文管理器 + 静默记录）。"""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


_NO_OP_SPAN = _NoOpSpan()


def get_tracer() -> Any:
    """返回 OpenTelemetry tracer；未安装返回 None（调用方先判 ``OTEL_AVAILABLE``）。"""
    if not OTEL_AVAILABLE:
        return None
    return _otel_trace.get_tracer(_TRACER_NAME)


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[Optional[Any]]:
    """开启一个追踪 span；未安装 OpenTelemetry 时为 no-op。

    异常时记录 exception 并以 ERROR 状态结束（随后照常抛出）。
    """
    if not OTEL_AVAILABLE:
        yield None
        return
    span = get_tracer().start_span(name)
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(_otel_trace.Status(_otel_trace.StatusCode.ERROR, str(exc)[:200]))
        raise
    finally:
        span.end()


class OTelASGIMiddleware:
    """ASGI 根 span 中间件（自实现最小版，无需额外安装插件）。

    每个请求一个 ``http.request`` span，携带 method/path；响应状态码回写
    span 属性。OTel 未安装时不应实例化（由 ``instrument_app`` 保证）。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not OTEL_AVAILABLE:
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "")
        path = scope.get("path", "")
        with trace_span(f"{method} {path}", **{"http.method": method, "http.path": path}) as span:
            async def send_wrapper(message: dict) -> None:
                if message.get("type") == "http.response.start" and span is not None:
                    span.set_attribute("http.status_code", message.get("status", 0))
                await send(message)

            await self.app(scope, receive, send_wrapper)


def instrument_app(app: Any) -> Any:
    """为 FastAPI 应用挂请求根 span（OTel 未安装时原样返回）。

    必须在 CORS 之后 add（Starlette 后加的中间件更靠外），保证根 span
    覆盖全部请求（含被 CORS 短路的预检请求）。
    """
    if not OTEL_AVAILABLE:
        return app
    app.add_middleware(OTelASGIMiddleware)
    return app
