"""MCP 外部工具服务管理器 — 统一启停 + 工具注册 + 权限过滤。

设计要点：
  - 每个 MCP server 连接跑在**独立常驻事件循环线程**里（stdio 子进程 / HTTP 客户端
    都是 async context manager，必须常驻在 loop 内持有）。
  - 连接建立后 list_tools，把每个工具桥接成同步 ToolDefinition 注册进全局
    ToolRegistry（名字带 `mcp_<server>_<tool>` 前缀，避免与内置工具冲突）。
  - 工具 fn 是同步签名（registry 契约），内部用 run_coroutine_threadsafe 把
    call_tool 提交到该 server 的常驻 loop 执行。
  - 权限两层：
      1. server 级 allowed_tools 白名单（config 声明）
      2. Worker 侧 tool_names 白名单（既有机制，天然生效）
  - start(name) / stop(name) / status() 支持统一启停；stop 同时从 registry 注销工具。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional

from app.core.exceptions import ToolExecutionError
from app.core.logger import get_logger
from app.tools.mcp.config import MCPServerConfig, load_mcp_servers
from app.tools.registry import ToolDefinition, get_tool_registry

logger = get_logger(__name__)

# 单次工具调用的超时（含网络往返），秒
MCP_CALL_TIMEOUT_S = 120


def _mcp_tool_name(server_name: str, tool_name: str) -> str:
    """MCP 工具注册到全局 registry 时的名字：mcp_<server>_<tool>。"""
    return f"mcp_{server_name}_{tool_name}"


def _mcp_tool_metadata(tool_name: str, description: str) -> Dict[str, Any]:
    """注册时从工具名/描述提取发现元数据（阶段 2，v1 轻量规则）。

    tags 取自工具名的单词段（供 ``@tag`` 绑定与词匹配）；scenarios 取
    description 的首两个短句（供任务描述子串匹配）。"""
    import re as _re

    tags = [
        w for w in _re.split(r"[^a-z0-9]+", tool_name.lower()) if len(w) > 2
    ]
    scenarios = [
        seg.strip()[:60]
        for seg in _re.split(r"[。.;；\n]", description or "")
        if seg.strip()
    ][:2]
    return {"scenarios": scenarios, "tags": tags}


class MCPServerHandle:
    """单个 MCP server 的运行句柄：常驻 loop 线程 + 连接会话 + 已注册工具。"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session = None          # mcp ClientSession
        # 保持对 async context manager 的引用！stdio_client / ClientSession 是
        # asynccontextmanager：若作为函数局部变量随返回被 GC，生成器会收到
        # GeneratorExit，子进程被关闭、流被断开 —— 后续调用报 Connection closed。
        self._transport_cm = None
        self._session_cm = None
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self.registered_tools: List[str] = []   # registry 内的完整工具名
        self.started_at: Optional[float] = None
        self._stop_requested = threading.Event()

    # ── 状态 ────────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive() and self.session is not None

    def to_status(self) -> Dict[str, Any]:
        tools = []
        try:
            reg = get_tool_registry()
            tools = [
                {"name": t.name.removeprefix(f"mcp_{self.config.name}_"), "enabled": True}
                for t in reg.list_all() if t.name in self.registered_tools
            ]
        except Exception:
            pass
        return {
            "name": self.config.name,
            "transport": self.config.transport,
            "enabled": self.config.enabled,
            "running": self.running,
            "error": self._error,
            "started_at": self.started_at,
            "tools": tools,
            "allowed_tools": self.config.allowed_tools,
        }

    # ── 常驻线程主体 ────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            loop.run_until_complete(self._connect_and_serve())
        except Exception as exc:
            logger.error("[mcp:%s] connection loop failed: %s", self.config.name, exc)
            self._error = str(exc)
        finally:
            self._ready.set()
            try:
                loop.close()
            except Exception:
                pass

    async def _connect_and_serve(self) -> None:
        cfg = self.config
        try:
            if cfg.is_stdio:
                session = await self._connect_stdio()
            else:
                session = await self._connect_http()
            self.session = session

            # 拉取工具清单，权限过滤后注册
            tools_result = await session.list_tools()
            available = [t for t in tools_result.tools if cfg.allows(t.name)]
            logger.info(
                "[mcp:%s] connected, %d/%d tools allowed",
                cfg.name, len(available), len(tools_result.tools),
            )
            self._register_tools(available)
            self.started_at = time.time()
            self._ready.set()

            # 保持 loop 存活：持续响应直到 stop 信号
            while not self._stop_requested.is_set():
                await asyncio.sleep(0.2)
        except Exception as exc:
            logger.error("[mcp:%s] connect failed: %s", cfg.name, exc)
            self._error = str(exc)
            self._ready.set()
        finally:
            await self._cleanup_session()

    async def _connect_stdio(self):
        import sys

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        command = list(self.config.command) if self.config.command else ["python"]
        # 关键：stdio 子进程必须用当前解释器（装有 mcp 包的环境），
        # 而不是 PATH 里的 python —— 否则子进程 import mcp 直接失败。
        if command[0].lower() in ("python", "python3", "py"):
            command[0] = sys.executable
        params = StdioServerParameters(
            command=command[0],
            args=command[1:],
            cwd=self.config.cwd,
            env=self.config.env or None,
        )
        transport_cm = stdio_client(params)
        self._transport_cm = transport_cm   # 防 GC（见 __init__ 注释）
        read, write = await transport_cm.__aenter__()
        session_cm = ClientSession(read, write)
        self._session_cm = session_cm       # 防 GC
        session = await session_cm.__aenter__()
        await session.initialize()
        return session

    async def _connect_http(self):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        transport_cm = streamable_http_client(self.config.url)
        self._transport_cm = transport_cm   # 防 GC（见 __init__ 注释）
        read, write = await transport_cm.__aenter__()
        session_cm = ClientSession(read, write)
        self._session_cm = session_cm       # 防 GC
        session = await session_cm.__aenter__()
        await session.initialize()
        return session

    async def _cleanup_session(self) -> None:
        try:
            if self.session is not None:
                await self.session.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("[mcp:%s] session close: %s", self.config.name, exc)
        self.session = None
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("[mcp:%s] session cm close: %s", self.config.name, exc)
        self._session_cm = None
        try:
            if self._transport_cm is not None:
                await self._transport_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("[mcp:%s] transport cm close: %s", self.config.name, exc)
        self._transport_cm = None

    # ── 工具注册 ────────────────────────────────────────────────────────
    def _register_tools(self, mcp_tools) -> None:
        """把 MCP 工具桥接为同步 ToolDefinition 注册进全局 registry。"""
        reg = get_tool_registry()
        cfg = self.config

        for t in mcp_tools:
            full_name = _mcp_tool_name(cfg.name, t.name)
            arg_schema: Dict[str, Any] = {}
            props = getattr(t, "inputSchema", {}) or {}
            for arg_name, meta in (props.get("properties") or {}).items():
                # 跳过下划线开头的参数（不符合 Pydantic/JSON Schema 规范）
                if arg_name.startswith("_"):
                    logger.debug(
                        "[mcp:%s] skip parameter %r in tool %r (leading underscore)",
                        cfg.name, arg_name, t.name
                    )
                    continue
                arg_schema[arg_name] = (
                    str(meta.get("type", "string")),
                    str(meta.get("description", "")),
                    arg_name in (props.get("required") or []),
                )

            def make_fn(tool_name: str, session_ref: "MCPServerHandle") -> Any:
                def fn(**kwargs: Any) -> str:
                    return session_ref.call_tool_sync(tool_name, kwargs)

                return fn

            reg.register(
                ToolDefinition(
                    name=full_name,
                    description=getattr(t, "description", "") or f"MCP tool {t.name}",
                    fn=make_fn(t.name, self),
                    arg_schema=arg_schema,
                    check_fn=lambda: self.session is not None,
                    # MCP 桥接工具自带 120s 超时（call_tool_sync），外层不再包裹
                    timeout_s=0,
                    metadata=_mcp_tool_metadata(
                        t.name, getattr(t, "description", "") or ""
                    ),
                )
            )
            self.registered_tools.append(full_name)
        logger.info("[mcp:%s] registered tools: %s", cfg.name, self.registered_tools)

    # ── 同步桥接（registry 契约）────────────────────────────────────────
    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """同步调用 MCP 工具：提交到常驻 loop，等待结果。"""
        if self.session is None or self.loop is None:
            raise ToolExecutionError(f"MCP server '{self.config.name}' is not running")
        if not self.config.allows(tool_name):
            raise ToolExecutionError(
                f"Tool '{tool_name}' is not allowed on MCP server '{self.config.name}'"
            )

        async def _call():
            result = await self.session.call_tool(tool_name, arguments)
            if getattr(result, "isError", False):
                raise ToolExecutionError(_text_content(result) or f"MCP tool '{tool_name}' failed")
            return _text_content(result) or f"(tool '{tool_name}' returned no text)"

        fut = asyncio.run_coroutine_threadsafe(_call(), self.loop)
        try:
            return fut.result(timeout=MCP_CALL_TIMEOUT_S)
        except asyncio.TimeoutError:
            fut.cancel()
            raise ToolExecutionError(f"MCP tool '{tool_name}' timed out after {MCP_CALL_TIMEOUT_S}s")
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"MCP tool '{tool_name}' error: {exc}") from exc


def _text_content(result) -> str:
    """从 CallToolResult 提取文本内容（支持 TextContent 及 dict 形态）。"""
    parts: List[str] = []
    for block in getattr(result, "content", []) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        else:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
    return "".join(parts)


# ── 全局单例管理器 ────────────────────────────────────────────────────────

class MCPManager:
    """统一管理所有 MCP server 连接的启停与工具注册。"""

    def __init__(self, servers: Optional[List[MCPServerConfig]] = None):
        self.servers: Dict[str, MCPServerHandle] = {}
        for cfg in servers if servers is not None else load_mcp_servers():
            self.servers[cfg.name] = MCPServerHandle(cfg)

    @classmethod
    def from_defaults(cls) -> "MCPManager":
        return cls()

    # ── 启停 ────────────────────────────────────────────────────────────
    def start(self, name: str, wait: bool = True, timeout: float = 30.0) -> Dict[str, Any]:
        """启动指定 server 连接。已运行则幂等返回。"""
        handle = self.servers.get(name)
        if handle is None:
            raise KeyError(f"MCP server '{name}' not configured")
        if handle.running:
            return handle.to_status()

        handle._error = None
        handle._stop_requested.clear()
        handle.started_at = None
        handle.registered_tools = []
        handle.thread = threading.Thread(
            target=handle._run_loop, name=f"mcp-{name}", daemon=True
        )
        handle.thread.start()

        if wait:
            if not handle._ready.wait(timeout):
                raise TimeoutError(f"MCP server '{name}' did not become ready in {timeout}s")
            if handle.session is None:
                raise RuntimeError(f"MCP server '{name}' failed to start: {handle._error}")
        return handle.to_status()

    def start_all(self, wait: bool = True) -> Dict[str, Any]:
        """启动所有 enabled 的 server。"""
        results = {}
        for name, handle in self.servers.items():
            if handle.config.enabled:
                try:
                    results[name] = self.start(name, wait=wait)
                except Exception as exc:
                    results[name] = {"name": name, "running": False, "error": str(exc)}
        return results

    def stop(self, name: str) -> Dict[str, Any]:
        """停止指定 server：发信号 → 等线程退出 → 注销工具。"""
        handle = self.servers.get(name)
        if handle is None:
            raise KeyError(f"MCP server '{name}' not configured")

        # 从 registry 注销本 server 注册的工具
        if handle.registered_tools:
            reg = get_tool_registry()
            for tool_name in handle.registered_tools:
                try:
                    reg.unregister(tool_name)
                except Exception:
                    pass
            handle.registered_tools = []

        handle._stop_requested.set()
        if handle.thread and handle.thread.is_alive():
            handle.thread.join(timeout=10)
        handle.started_at = None
        return handle.to_status()

    def stop_all(self) -> None:
        for name in list(self.servers):
            try:
                self.stop(name)
            except Exception as exc:
                logger.warning("[mcp] stop %s: %s", name, exc)

    # ── 查询 ────────────────────────────────────────────────────────────
    def status(self) -> List[Dict[str, Any]]:
        return [h.to_status() for h in self.servers.values()]

    def get(self, name: str) -> MCPServerHandle:
        handle = self.servers.get(name)
        if handle is None:
            raise KeyError(f"MCP server '{name}' not configured")
        return handle


# 全局单例（lazy init，首次访问时加载配置）
_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
