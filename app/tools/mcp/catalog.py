"""ModelScope MCP 广场目录客户端与安装计划推导。

用途：让 MCP 控制台可以浏览/搜索 https://www.modelscope.cn/mcp 上的 MCP 服务，
并把选中的服务一键转成本地 MCP Server 定义（stdio 或远程 URL）。

分工：
- ``search_catalog`` / ``get_catalog_detail``：直接调用 ModelScope OpenAPI
  （``PUT/GET {endpoint}/mcp/servers``，检索不需要凭据），带 TTL 缓存。
- ``build_install_plan`` 等纯函数：把握手用的 ``server_config`` 规范化成
  「可以直接提交给 MCPServerCreate 的安装计划」，不依赖网络，便于单测。

ModelScope 返回的关键字段（2026-10 实测）::

    {"success": true, "data": {
        "id": "@amap/amap-maps", "name": "高德地图", "description": "...",
        "logo_url": "https://...", "view_count": 429669, "categories": ["location-services"],
        "tags": [], "author": "amap", "source_url": "https://www.npmjs.com/package/...",
        "operational_urls": [{"url": "https://mcp.../sse", "transport_type": "sse"}],
        "server_config": [{"mcpServers": {"amap-maps": {
            "command": "npx", "args": ["-y", "@amap/amap-maps-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": ""}}}}],
    }}
"""
from __future__ import annotations

import re
import shutil
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from app.core.config import get_settings
from app.core.logger import get_logger
from app.tools.mcp.models import TRUSTED_STDIO_BINARIES

logger = get_logger(__name__)

# 目录条目里出现的进程启动器 → 运行环境自检提示
_RUNTIME_HINTS = {
    "npx": "需要 Node.js 与 npx（镜像已内置），首次运行会从 npm registry 拉取包。",
    "node": "需要 Node.js（镜像已内置）。",
    "uvx": "需要 uv/uvx（uv tool run）。镜像未内置时请先安装 uv（pip install uv）。",
    "uv": "需要 uv（镜像未内置时请先安装）。",
    "python": "使用后端自带解释器；若为第三方包，需先 pip install 该包。",
    "python3": "使用后端自带解释器；若为第三方包，需先 pip install 该包。",
    "py": "使用后端自带解释器；若为第三方包，需先 pip install 该包。",
    "docker": "需要容器内可用的 docker CLI（默认未提供）。",
}

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def slugify_catalog_id(catalog_id: str) -> str:
    """``@amap/amap-maps`` → ``amap-amap-maps``，保证符合后端 server_id 规则。

    后端 ``MCPServerBase.server_id`` 只允许 ``[a-zA-Z0-9_-]``，而广场 id 带
    ``@`` 与 ``/``，这里统一折叠成连字符（控制台仍允许用户改写）。
    """
    raw = str(catalog_id or "").strip().lstrip("@")
    slug = _SLUG_STRIP_RE.sub("-", raw.lower()).strip("-")
    return slug or "mcp-server"


def infer_transport(url: str) -> str:
    """按端点后缀推断传输协议：``/sse`` → sse，其余 → streamable_http。"""
    base = str(url or "").split("?")[0].rstrip("/").lower()
    return "sse" if base.endswith("/sse") else "streamable_http"


def extract_server_entry(server_config: Optional[Sequence[Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """从广场的 ``server_config``（标准 mcpServers 结构）取出第一个可用条目。

    返回 ``{"name", "command", "args", "env", "url"}``；无法识别时返回 None。
    """
    for block in server_config or ():
        if not isinstance(block, Mapping):
            continue
        servers = block.get("mcpServers") or block.get("mcp_servers") or {}
        if not isinstance(servers, Mapping):
            continue
        for name, entry in servers.items():
            if not isinstance(entry, Mapping):
                continue
            command = str(entry.get("command") or "").strip()
            url = str(entry.get("url") or "").strip()
            if not command and not url:
                continue
            args = [str(item) for item in (entry.get("args") or [])]
            env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
            return {
                "name": str(name),
                "command": command,
                "args": args,
                "env": env,
                "url": url,
            }
    return None


def required_env_keys(entry: Optional[Mapping[str, Any]]) -> List[str]:
    """需要用户填写的环境变量名。

    广场里 ``env`` 的**值一律为空串**（占位，如 ``{"AMAP_MAPS_API_KEY": ""}``），
    因此非空值说明发布者已经给了默认值，不必强制用户输入。
    """
    if not entry:
        return []
    env = entry.get("env") or {}
    if not isinstance(env, Mapping):
        return []
    return sorted(str(key) for key, value in env.items() if not str(value or "").strip())


def default_env_values(entry: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """发布者预置了默认值（非空）的环境变量，可直接随安装写入。"""
    if not entry:
        return {}
    env = entry.get("env") or {}
    if not isinstance(env, Mapping):
        return {}
    return {str(k): str(v) for k, v in env.items() if str(v or "").strip()}


def binary_name(command: str) -> str:
    """取可执行文件基名（与 manager 的 stdio 白名单判定保持一致）。"""
    first = str(command or "").strip().split()[0] if str(command or "").strip() else ""
    return first.lower().replace(".exe", "").split("\\")[-1].split("/")[-1]


def runtime_status(command: str, *, resolve: Callable[[str], Optional[str]] = shutil.which) -> Dict[str, Any]:
    """运行环境自检：该启动器是否受信任、是否已安装、需要什么前提。"""
    name = binary_name(command)
    if not name:
        return {"binary": "", "trusted": False, "available": False, "hint": "未提供启动命令"}
    trusted = name in TRUSTED_STDIO_BINARIES or name in ("uvx",)
    available = bool(resolve(name)) if trusted else False
    hint = _RUNTIME_HINTS.get(name, "")
    if trusted and not available:
        hint = hint or f"运行环境缺少 {name}，连接时会失败。"
    return {"binary": name, "trusted": trusted, "available": available, "hint": hint}


def build_install_plan(
    detail: Mapping[str, Any],
    *,
    resolve: Callable[[str], Optional[str]] = shutil.which,
) -> Dict[str, Any]:
    """把广场详情规范化为安装计划（视图直接消费，无需再解析 server_config）。

    ``kind`` 三种取值：
      - ``stdio``           本地子进程（command/args 来自 server_config）
      - ``remote``          远程端点（operational_urls 或 server_config.url）
      - ``deploy_required`` 广场只在托管侧提供（需要 MODELSCOPE_API_KEY 部署后才能接入）
    """
    detail = detail or {}
    catalog_id = str(detail.get("id") or "")
    entry = extract_server_entry(detail.get("server_config"))

    plan: Dict[str, Any] = {
        "catalog_id": catalog_id,
        "name": str(detail.get("chinese_name") or detail.get("name") or catalog_id),
        "description": str(detail.get("description") or ""),
        "author": str(detail.get("author") or ""),
        "logo_url": str(detail.get("logo_url") or ""),
        "source_url": str(detail.get("source_url") or ""),
        "categories": [str(item) for item in (detail.get("categories") or [])],
        "tags": [str(item) for item in (detail.get("tags") or [])],
        "view_count": int(detail.get("view_count") or 0),
        "server_id_suggestion": slugify_catalog_id(catalog_id),
        "kind": "deploy_required",
        "transport": None,
        "stdio_command": None,
        "url": None,
        "env_schema": [],
        "env_defaults": {},
        "runtime": None,
    }

    # 1. 已部署的托管端点优先（operational_urls 是完整 MCP 端点，无需拼接）
    for item in detail.get("operational_urls") or ():
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        declared = str(item.get("transport_type") or "").strip().lower()
        transport = declared if declared in ("sse", "streamable_http") else infer_transport(url)
        plan.update({"kind": "remote", "transport": transport, "url": url})
        return plan

    # 2. server_config 里的远程 URL
    if entry and entry.get("url"):
        url = str(entry["url"])
        plan.update({"kind": "remote", "transport": infer_transport(url), "url": url})
        return plan

    # 3. server_config 里的本地命令
    if entry and entry.get("command"):
        command = [entry["command"], *entry.get("args", [])]
        plan.update({
            "kind": "stdio",
            "transport": "stdio",
            "stdio_command": command,
            "env_schema": required_env_keys(entry),
            "env_defaults": default_env_values(entry),
            "runtime": runtime_status(entry["command"], resolve=resolve),
        })
        return plan

    # 4. 广场只给了托管侧信息 → 需要部署（保留 env 提示，便于用户去广场配置）
    return plan


def merge_env(env_defaults: Mapping[str, str], required: Iterable[str], user_env: Mapping[str, str]) -> Dict[str, str]:
    """合并环境变量：默认值 → 必填项 → 用户输入（用户值非空才覆盖）。"""
    merged: Dict[str, str] = {str(k): str(v) for k, v in (env_defaults or {}).items()}
    for key in required:
        merged.setdefault(str(key), "")
    for key, value in (user_env or {}).items():
        text = str(value or "").strip()
        if text:
            merged[str(key)] = text
    return {k: v for k, v in merged.items() if str(v).strip()}


def missing_env_keys(plan: Mapping[str, Any], user_env: Mapping[str, str]) -> List[str]:
    """用户尚未填写的必填环境变量。"""
    provided = {str(k) for k, v in (user_env or {}).items() if str(v or "").strip()}
    return [key for key in plan.get("env_schema") or [] if key not in provided]


def plan_to_create_payload(
    plan: Mapping[str, Any],
    *,
    server_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    transport: Optional[str] = None,
    name: Optional[str] = None,
    timeout_s: int = 30,
    allowed_tools: Optional[Sequence[str]] = None,
    capabilities: Optional[Sequence[str]] = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    """安装计划 + 用户输入 → ``MCPServerCreate`` 入参（纯字典，便于单测）。"""
    kind = plan.get("kind")
    resolved_transport = transport or plan.get("transport")
    if kind == "remote" and resolved_transport == "sse":
        resolved_transport = "sse"
    if kind == "remote" and resolved_transport not in ("sse", "streamable_http"):
        resolved_transport = infer_transport(str(plan.get("url") or ""))

    payload: Dict[str, Any] = {
        "server_id": server_id or plan.get("server_id_suggestion"),
        "name": name or plan.get("name"),
        "description": plan.get("description") or "",
        "transport": resolved_transport,
        "timeout_s": int(timeout_s),
        "enabled": bool(enabled),
        # 广场来源默认最小权限：不写 ["*"]，由调用方显式声明
        "allowed_tools": list(allowed_tools or ["*"]),
        "capabilities": list(capabilities or []),
    }

    if kind == "stdio":
        payload["stdio_command"] = list(plan.get("stdio_command") or [])
        merged = merge_env(plan.get("env_defaults") or {}, plan.get("env_schema") or [], env or {})
        if merged:
            payload["env"] = merged
    else:
        # 远程服务的 env 对连接无意义，只接受端点与可选令牌
        payload["url"] = plan.get("url")
    payload["source"] = f"modelscope:{plan.get('catalog_id')}" if plan.get("catalog_id") else ""
    return payload


# ── ModelScope 目录客户端（带 TTL 缓存）────────────────────────────────────

class CatalogError(RuntimeError):
    """目录接口不可用（网络/上游返回异常）。"""


class CatalogClient:
    """ModelScope MCP 广场只读客户端。"""

    def __init__(self, *, endpoint: Optional[str] = None, timeout: Optional[float] = None,
                 cache_ttl: Optional[float] = None):
        settings = get_settings()
        self.endpoint = (endpoint or settings.MODELSCOPE_MCP_ENDPOINT).rstrip("/")
        self.timeout = float(timeout if timeout is not None else settings.MCP_CATALOG_TIMEOUT)
        self.cache_ttl = float(cache_ttl if cache_ttl is not None else settings.MCP_CATALOG_CACHE_TTL)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ── 缓存 ─────────────────────────────────────────────────────────────
    def _cache_get(self, key: str) -> Any:
        with self._lock:
            hit = self._cache.get(key)
        if not hit:
            return None
        stored_at, value = hit
        if self.cache_ttl > 0 and (time.time() - stored_at) > self.cache_ttl:
            with self._lock:
                self._cache.pop(key, None)
            return None
        return value

    def _cache_put(self, key: str, value: Any) -> None:
        if self.cache_ttl <= 0:
            return
        with self._lock:
            self._cache[key] = (time.time(), value)
            if len(self._cache) > 200:  # 简单容量保护
                oldest = sorted(self._cache.items(), key=lambda item: item[1][0])[:50]
                for name, _ in oldest:
                    self._cache.pop(name, None)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    # ── 请求 ─────────────────────────────────────────────────────────────
    async def _request(self, method: str, path: str, *, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method,
                    url,
                    json=json_body,
                    headers={"Accept": "application/json", "User-Agent": "EasyRAG-MCP-Console"},
                )
        except httpx.HTTPError as exc:
            raise CatalogError(f"MCP 广场不可达：{exc}") from exc
        if response.status_code >= 400:
            raise CatalogError(f"MCP 广场返回 HTTP {response.status_code}：{response.text[:200]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise CatalogError("MCP 广场返回了非 JSON 响应") from exc
        if payload.get("success") is False:
            raise CatalogError(str(payload.get("message") or "MCP 广场请求失败"))
        return payload.get("data") or {}

    async def search(self, *, search: str = "", page: int = 1, page_size: int = 12) -> Dict[str, Any]:
        """关键词检索广场服务（分页上限 page_number × page_size ≤ 100）。"""
        page_size = max(1, min(int(page_size or 12), 100))
        page = max(1, int(page or 1))
        if page * page_size > 100:
            page = max(1, 100 // page_size)
        key = f"search:{search.strip().lower()}:{page}:{page_size}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        data = await self._request(
            "PUT",
            "/mcp/servers",
            json_body={"page_number": page, "page_size": page_size, "search": search or None},
        )
        items = []
        for raw in data.get("mcp_server_list") or ():
            if not isinstance(raw, Mapping):
                continue
            locales = raw.get("locales") or {}
            zh = locales.get("zh") or {}
            items.append({
                "id": str(raw.get("id") or ""),
                "name": str(zh.get("name") or raw.get("chinese_name") or raw.get("name") or raw.get("id") or ""),
                "description": str(zh.get("description") or raw.get("description") or ""),
                "categories": [str(item) for item in (raw.get("categories") or [])],
                "tags": [str(item) for item in (raw.get("tags") or [])],
                "logo_url": str(raw.get("logo_url") or ""),
                "view_count": int(raw.get("view_count") or 0),
                "author": str(raw.get("publisher") or ""),
            })
        result = {
            "items": items,
            "total": int(data.get("total_count") or len(items)),
            "page": page,
            "page_size": page_size,
        }
        self._cache_put(key, result)
        return result

    async def detail(self, catalog_id: str) -> Dict[str, Any]:
        """服务详情 + 安装计划。"""
        catalog_id = str(catalog_id or "").strip()
        if not catalog_id:
            raise CatalogError("缺少服务标识")
        key = f"detail:{catalog_id}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        data = await self._request("GET", f"/mcp/servers/{catalog_id}")
        result = {"raw": data, "plan": build_install_plan(data)}
        self._cache_put(key, result)
        return result


_client: Optional[CatalogClient] = None


def get_catalog_client() -> CatalogClient:
    global _client
    if _client is None:
        _client = CatalogClient()
    return _client
