"""Application-wide logger setup.

Provides `get_logger(name)` for per-module loggers all sharing the same
handler configuration. Uses stdlib logging - no extra runtime dependencies.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT = "all_in_rag"


def _ensure_root_handler(level: int) -> None:
    """Configure the root 'all_in_rag' logger once."""
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.propagate = False


@lru_cache(maxsize=None)
def get_logger(name: str = _ROOT) -> logging.Logger:
    """Return a child logger under the 'all_in_rag' hierarchy.

    Example::

        logger = get_logger(__name__)
    """
    # Resolve log level lazily so .env is already loaded
    from app.core.config import get_settings  # local import avoids circular dep
    cfg = get_settings()
    level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
    _ensure_root_handler(level)

    log = logging.getLogger(name if name != _ROOT else _ROOT)
    log.setLevel(level)
    return log


# Convenience alias kept for any legacy code that does `from app.core.logger import logger`
logger = get_logger(_ROOT)
