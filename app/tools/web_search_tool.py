"""Web search tool backed by the Tavily Search API.

Tavily is a search API optimised for LLM agents — it returns clean,
pre-extracted page content instead of raw HTML.

Docs: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import requests

from app.core.config import get_settings
from app.core.exceptions import ToolExecutionError
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

_TAVILY_API_URL = "https://api.tavily.com/search"
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_TIMEOUT = 15

# Marker wrapping a machine-readable JSON list of sources at the end of the
# tool output. The LLM sees it as a trailing comment; the tool_execution node
# strips it and stores the parsed list in state["sources"].
SOURCES_MARKER = "<!--SOURCES:"


def extract_sources(tool_output: str) -> List[Dict[str, str]]:
    """Parse the sources block out of web_search output.

    Returns a list of {"title", "url"} dicts, or [] if absent/unparseable.
    """
    match = re.search(re.escape(SOURCES_MARKER) + r"(.*?)-->", tool_output, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
        if isinstance(data, list):
            return [{"title": str(s.get("title", "")), "url": str(s.get("url", ""))} for s in data]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return []


def strip_sources_block(tool_output: str) -> str:
    """Remove the machine-readable sources block from tool output (for display)."""
    return re.sub(re.escape(SOURCES_MARKER) + r".*?-->", "", tool_output, flags=re.DOTALL).rstrip()


def _format_results(data: Dict[str, Any], max_results: int) -> str:
    """Turn the Tavily JSON response into a compact, LLM-friendly string."""
    lines: List[str] = []

    # Tavily sometimes returns a direct short answer — surface it first.
    answer = data.get("answer")
    if answer:
        lines.append(f"Summary answer: {answer}")
        lines.append("")

    results = data.get("results") or []
    if not results:
        return "No search results found."

    sources: List[Dict[str, str]] = []
    for i, item in enumerate(results[:max_results], 1):
        title = item.get("title") or "(no title)"
        url = item.get("url") or ""
        content = (item.get("content") or "").strip()
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    URL: {url}")
        if content:
            lines.append(f"    {content}")
        lines.append("")
        if url:
            sources.append({"title": title, "url": url})

    body = "\n".join(lines).strip()
    if sources:
        body += "\n\n" + SOURCES_MARKER + json.dumps(sources, ensure_ascii=False) + "-->"
    return body


def web_search(
    query: str,
    max_results: Optional[int] = None,
    search_depth: Optional[str] = None,
    include_answer: Optional[bool] = None,
    progress_callback: Optional[Callable[..., Any]] = None,
) -> str:
    """Search the web via Tavily and return formatted results.

    Args:
        query:          The search query string.
        max_results:    Number of results to return (default 5, max 10).
        search_depth:   "basic" (fast) or "advanced" (deeper, slower).
        include_answer: Whether to include Tavily's short summary answer.
        progress_callback: 阶段 5 可选进度上报（由 registry.invoke 注入）。

    Returns:
        A formatted string with the summary answer (if any) and the top
        results — title, URL, and extracted content snippet for each.

    Raises:
        ToolExecutionError: when the API key is missing or the request fails.
    """
    api_key = cfg.TAVILY_API_KEY
    if not api_key:
        raise ToolExecutionError(
            "Tavily search is not configured. "
            "Set TAVILY_API_KEY in your .env file."
        )

    max_results = max_results or cfg.TAVILY_MAX_RESULTS or _DEFAULT_MAX_RESULTS
    max_results = max(1, min(int(max_results), 10))
    search_depth = search_depth or cfg.TAVILY_SEARCH_DEPTH or "basic"
    include_answer = cfg.TAVILY_INCLUDE_ANSWER if include_answer is None else include_answer

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
    }

    logger.info("web_search: query=%r max_results=%d depth=%s", query, max_results, search_depth)
    if progress_callback:
        progress_callback("联网搜索：请求 Tavily API…", percent=30)

    try:
        resp = requests.post(_TAVILY_API_URL, json=payload, timeout=_DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise ToolExecutionError(f"Tavily request failed: {exc}") from exc

    if resp.status_code != 200:
        raise ToolExecutionError(
            f"Tavily API returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise ToolExecutionError(f"Failed to parse Tavily response: {exc}") from exc

    if progress_callback:
        progress_callback("联网搜索：整理结果…", percent=90)
    return _format_results(data, max_results)


# ── 插件导出（discover_tools 自动注册）─────────────────────────────────────
from app.tools.registry import ToolDefinition


def _check() -> bool:
    # web_search 依赖 Tavily API key，未配置则不可用（不出现在 schema/react prompt）
    return bool(cfg.TAVILY_API_KEY)


TOOL = ToolDefinition(
    name="web_search",
    description="Search the web for current/real-time information: news, weather, recent events, prices, or anything not in the knowledge base. Returns titles, URLs and content snippets.",
    fn=lambda query, max_results=None, progress_callback=None, **_: web_search(
        query=query, max_results=max_results, progress_callback=progress_callback,
    ),
    arg_schema={
        "query": ("string", "The search query, e.g. 'latest AI news today'", True),
        "max_results": ("number", "Max results to return (1-10, default 5)", False),
    },
    check_fn=_check,
    # 网络工具：瞬时失败（抖动/限流）重试一次，指数退避（2026-08-26 阶段 1）。
    # 超时不重试（重试只会翻倍等待）。
    timeout_s=45,
    max_retries=1,
    # 阶段 2：工具发现元数据（适用场景关键词 + 能力标签）
    metadata={
        "scenarios": ["联网搜索", "网络搜索", "上网查", "新闻", "天气", "实时", "最新", "近期", "行情", "价格"],
        "tags": ["search", "web", "internet"],
    },
)
