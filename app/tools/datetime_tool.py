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
