"""Process-scoped optional client. Export never participates in business I/O."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from urllib.parse import urlsplit

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_client = None
_initialized = False
_state = "disabled"
_lock = threading.Lock()


def valid_base_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        return bool(url.scheme in {"http", "https"} and url.hostname and not url.username
                    and not url.password and not url.query and not url.fragment)
    except ValueError:
        return False


def initialize():
    global _client, _initialized, _state
    with _lock:
        if _initialized:
            return _client
        _initialized = True
        cfg = get_settings()
        if not cfg.LANGFUSE_ENABLED:
            _state = "disabled"
            return None
        if not (cfg.LANGFUSE_PUBLIC_KEY and cfg.LANGFUSE_SECRET_KEY and valid_base_url(cfg.LANGFUSE_BASE_URL)):
            _state = "misconfigured"
            logger.warning("Langfuse disabled: complete keys and an explicit http(s) base URL are required")
            return None
        try:
            from langfuse import Langfuse
            from langfuse.span_filter import is_langfuse_span
            from opentelemetry.sdk.trace import TracerProvider
            from app.observability.redaction import mask_otel_spans
            _client = Langfuse(
                public_key=cfg.LANGFUSE_PUBLIC_KEY,
                secret_key=cfg.LANGFUSE_SECRET_KEY,
                base_url=cfg.LANGFUSE_BASE_URL.rstrip("/"),
                environment=cfg.LANGFUSE_TRACING_ENVIRONMENT,
                release=cfg.APP_VERSION,
                tracer_provider=TracerProvider(),
                # Sampling is decided and persisted at admission, not sampled a
                # second time by the SDK. The processor queue is bounded by OTel.
                sample_rate=1.0, timeout=2, flush_at=64, flush_interval=2,
                should_export_span=is_langfuse_span,
                mask_otel_spans=mask_otel_spans,
                debug=False,
            )
            _state = "initialized"
        except Exception as exc:
            _state = "unavailable"
            logger.warning("Langfuse initialization skipped (%s)", type(exc).__name__)
        return _client


def get_client():
    # The default disabled path doesn't initialize/import the SDK.
    if not get_settings().LANGFUSE_ENABLED:
        return None
    return initialize()


def state() -> str:
    get_client()
    return _state if get_settings().LANGFUSE_ENABLED else "disabled"


async def shutdown():
    global _client, _state
    client, _client = _client, None
    if client is None:
        return
    _state = "stopped"
    done = threading.Event()

    def close():
        try:
            client.shutdown()
        except Exception:
            logger.warning("Langfuse shutdown failed")
        finally:
            done.set()

    # Do not put a possibly slow exporter on asyncio's default executor: loop
    # shutdown waits for that executor even after wait_for has expired.
    threading.Thread(target=close, name="langfuse-shutdown", daemon=True).start()
    deadline = time.monotonic() + get_settings().EASYRAG_OBSERVABILITY_SHUTDOWN_TIMEOUT_SECONDS
    while not done.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(.02)
