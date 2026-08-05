"""Datetime tool – returns current time or formats timestamps."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.core.exceptions import ToolExecutionError
from app.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_FMT = "%Y-%m-%d %H:%M:%S"


def datetime_tool(
    fmt: Optional[str] = None,
    tz: str = "local",
    timestamp: Optional[float] = None,
) -> str:
    """Return the current date/time as a formatted string.

    Args:
        fmt:       strftime format string. Defaults to "%Y-%m-%d %H:%M:%S".
        tz:        "local" for local time, "utc" for UTC.
        timestamp: Optional Unix timestamp (seconds). If None, uses now().

    Returns:
        Formatted datetime string plus a human-readable label.

    Raises:
        ToolExecutionError: on invalid format string.
    """
    logger.debug("datetime_tool: fmt=%s tz=%s ts=%s", fmt, tz, timestamp)
    effective_fmt = fmt or _DEFAULT_FMT

    try:
        if timestamp is not None:
            dt = datetime.fromtimestamp(float(timestamp))
        elif tz.lower() == "utc":
            dt = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            dt = datetime.now()

        result = dt.strftime(effective_fmt)
        label = "UTC" if tz.lower() == "utc" else "local time"
        return f"Current {label}: {result}"
    except ValueError as exc:
        raise ToolExecutionError(f"Invalid format string '{effective_fmt}': {exc}") from exc
    except Exception as exc:
        raise ToolExecutionError(f"datetime_tool failed: {exc}") from exc


def get_weekday(timestamp: Optional[float] = None) -> str:
    """Return the current weekday name (e.g. 'Monday')."""
    dt = datetime.fromtimestamp(float(timestamp)) if timestamp else datetime.now()
    return dt.strftime("%A")


def days_between(date1: str, date2: str, fmt: str = "%Y-%m-%d") -> str:
    """Return the number of days between two date strings."""
    try:
        d1 = datetime.strptime(date1, fmt)
        d2 = datetime.strptime(date2, fmt)
        delta = abs((d2 - d1).days)
        return f"Days between {date1} and {date2}: {delta}"
    except ValueError as exc:
        raise ToolExecutionError(f"Invalid date format: {exc}") from exc


# ── 插件导出（discover_tools 自动注册）─────────────────────────────────────
from app.tools.registry import ToolDefinition


def _check() -> bool:
    return True  # datetime 无外部依赖，总是可用


TOOL = ToolDefinition(
    name="datetime_tool",
    description="Return the current date and time, optionally formatted.",
    fn=lambda fmt=None, tz="local", timestamp=None, **_: datetime_tool(
        fmt=fmt, tz=tz, timestamp=timestamp
    ),
    arg_schema={
        "fmt": ("string", "strftime format, e.g. '%Y-%m-%d'", False),
        "tz": ("string", "'local' or 'utc'", False),
        "timestamp": ("number", "Unix timestamp in seconds (optional)", False),
    },
    check_fn=_check,
)
