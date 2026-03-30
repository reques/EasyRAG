"""Application entry point.

Usage:
    python run.py              # production-style, no reload
    python run.py --reload     # development hot-reload
    uvicorn run:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import socket
import sys

# ── 代理修复：清除系统代理，防止 OpenAI SDK / httpx 通过本地代理访问外网 API
# Windows 注册表代理 (如 Clash/V2Ray 127.0.0.1:7897) 会被 httpx trust_env 读取
for _pv in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "ALL_PROXY", "all_proxy"):
    os.environ.pop(_pv, None)
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
# ─────────────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import RedirectResponse, HTMLResponse

from app.core.config import get_settings
from app.core.logger import get_logger

cfg = get_settings()
logger = get_logger(__name__)

# Use unpkg CDN which is more accessible in China than cdn.jsdelivr.net
_SWAGGER_JS  = "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"
_SWAGGER_CSS = "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
_REDOC_JS    = "https://unpkg.com/redoc@latest/bundles/redoc.standalone.js"


def _is_port_in_use(host: str, port: int) -> bool:
    """Return True if *port* is already bound on *host*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting %s v%s", cfg.APP_NAME, cfg.APP_VERSION)
    logger.info("LLM        : %s @ %s", cfg.LLM_MODEL, cfg.LLM_BASE_URL)
    logger.info("VectorStore: %s", cfg.VECTOR_STORE_TYPE)
    logger.info("Embedding  : %s", cfg.EMBEDDING_TYPE)
    logger.info("Docs       : http://%s:%d/docs", cfg.HOST, cfg.PORT)
    yield
    logger.info("Shutting down %s", cfg.APP_NAME)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Disable built-in docs so we can provide custom CDN URLs
    application = FastAPI(
        title=cfg.APP_NAME,
        version=cfg.APP_VERSION,
        description="Multi-step Agent QA system based on LangGraph + FastAPI",
        lifespan=lifespan,
        docs_url=None,    # disable default /docs (we serve custom below)
        redoc_url=None,   # disable default /redoc
    )

    # ── CORS ────────────────────────────────────────────────────────────
    origins = [o.strip() for o in cfg.CORS_ORIGINS.split(",")]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────
    from app.api.routes import router
    from app.api.kb_routes import router as kb_router

    application.include_router(router, prefix="/api/v1")
    application.include_router(kb_router, prefix="/api/v1")

    # ── Built-in redirects & custom docs ────────────────────────────────
    @application.get("/", include_in_schema=False)
    def root_redirect():
        """Redirect root to Swagger UI."""
        return RedirectResponse(url="/docs")

    @application.get("/docs", include_in_schema=False)
    def custom_swagger_ui() -> HTMLResponse:
        """Swagger UI served with unpkg CDN (works inside China)."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=cfg.APP_NAME + " - Swagger UI",
            swagger_js_url=_SWAGGER_JS,
            swagger_css_url=_SWAGGER_CSS,
        )

    @application.get("/redoc", include_in_schema=False)
    def custom_redoc() -> HTMLResponse:
        """ReDoc served with unpkg CDN."""
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=cfg.APP_NAME + " - ReDoc",
            redoc_js_url=_REDOC_JS,
        )

    return application


app = create_app()


if __name__ == "__main__":
    # ── Pre-flight: detect port conflict before uvicorn tries to bind ──
    if _is_port_in_use(cfg.HOST, cfg.PORT):
        print(
            f"\n[ERROR] Port {cfg.PORT} is already in use.\n"
            "  → Stop the old process first, e.g.:\n"
            f"      netstat -ano | findstr :{cfg.PORT}   (Windows)\n"
            f"      kill -9 <PID>                        (Linux/macOS)\n"
            "  Or change PORT in .env and restart.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Reload flag: pass --reload on the CLI to enable hot-reload ─────
    reload = "--reload" in sys.argv

    uvicorn.run(
        "run:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=reload,                  # False by default; pass --reload to enable
        log_level=cfg.LOG_LEVEL.lower(),
        timeout_graceful_shutdown=2,    # release port faster on Windows
        access_log=True,
    )
