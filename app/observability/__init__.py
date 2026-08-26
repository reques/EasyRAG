"""可观测性（2026-08-26，阶段 5）— OpenTelemetry 可选依赖。

未安装 OpenTelemetry 时本模块全部为 no-op（零开销、零依赖），
安装 ``opentelemetry-api``（+ ``opentelemetry-instrumentation-asgi``
可选）后自动启用真实 span：

- ``trace_span``：registry.invoke / task / spawn_tasks 的执行 span；
- ``instrument_app``：ASGI 请求根 span（CORS 更外层，覆盖全部请求）。

统一事件流（``app/agents/events.py``）负责业务事件，本模块只负责
分布式追踪，两者互不依赖。
"""
from app.observability.tracing import (
    OTEL_AVAILABLE,
    OTelASGIMiddleware,
    get_tracer,
    instrument_app,
    trace_span,
)

__all__ = [
    "OTEL_AVAILABLE",
    "OTelASGIMiddleware",
    "get_tracer",
    "instrument_app",
    "trace_span",
]
