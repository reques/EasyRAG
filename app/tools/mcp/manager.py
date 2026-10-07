"""MCP 外部工具服务管理器 — 对齐 Yuxi 企业级架构重构。

核心设计：
  1. 传输协议支持：SSE（远程优先）、Streamable HTTP、受控 Stdio（本地受信任白名单）；
  2. 异步生命周期管理：基于 contextlib.AsyncExitStack 统一管理会话与流，杜绝资源泄漏；
  3. 双向命名空间与工具路由：
     - 主命名空间：{server_id}__{tool_name}（Yuxi / LangGraph 标准）
     - 兼容别名：mcp_{server_id}_{tool_name}（EasyRAG 既有 ToolRegistry 兼容）
  4. 动态配置与热插拔：支持运行时 register / unregister / update / sync_tools；
  5. 异常防御：对超时、连接丢失、远程执行错误提供格式化字符串兜底，促进 LLM 自主重试。
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.exceptions import ToolExecutionError
from app.core.logger import get_logger
from app.tools.mcp.config import (
    MCPServerConfig,
    load_mcp_server_definitions,
    save_mcp_server_definitions,
)
from app.tools.mcp.models import (
    MCPServerCreate,
    MCPServerDefinition,
    MCPServerStatus,
    MCPServerUpdate,
    MCPToolInfo,
    MCPTransportType,
    TRUSTED_STDIO_BINARIES,
)
from app.tools.registry import ToolDefinition, get_tool_registry

logger = get_logger(__name__)

# 默认命名空间分隔符
TOOL_NAMESPACE_SEP = "__"
LEGACY_PREFIX = "mcp_"


def _mcp_tool_name(server_id: str, tool_name: str) -> str:
    """Yuxi 标准命名空间格式：{server_id}__{tool_name}。"""
    return f"{server_id}{TOOL_NAMESPACE_SEP}{tool_name}"


def _legacy_mcp_tool_name(server_id: str, tool_name: str) -> str:
    """EasyRAG 历史兼容格式：mcp_{server_id}_{tool_name}。"""
    return f"{LEGACY_PREFIX}{server_id}_{tool_name}"


def _mcp_tool_metadata(
    tool_name: str,
    description: str,
    capabilities: Optional[List[str]] = None,
    server_id: str = "",
) -> Dict[str, Any]:
    """从工具名和描述提取能力元数据，支持按 server_id 与标签路由。"""
    tags = [w for w in re.split(r"[^a-z0-9]+", tool_name.lower()) if len(w) > 2]
    scenarios = [
        seg.strip()[:60]
        for seg in re.split(r"[。.;；\n]", description or "")
        if seg.strip()
    ][:2]
    return {
        "server_id": server_id,
        "scenarios": scenarios,
        "tags": tags,
        "capabilities": list(capabilities or ["none"]),
    }


def _text_content(result: Any) -> str:
    """从 CallToolResult 提取文本内容。"""
    parts: List[str] = []
    for block in getattr(result, "content", []) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        else:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
    return "\n".join(parts)


class MCPServerHandle:
    """单个 MCP server 的运行句柄：AsyncExitStack 会话管理 + 常驻 loop 线程 + 工具注册。"""

    def __init__(self, definition: MCPServerDefinition):
        self.config: MCPServerDefinition = definition
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session: Any = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.registered_tools: List[str] = []       # registry 中的注册名
        self.tool_infos: Dict[str, MCPToolInfo] = {}  # namespaced_name -> MCPToolInfo

    @property
    def server_id(self) -> str:
        return self.config.server_id

    @property
    def running(self) -> bool:
        return (
            self.thread is not None
            and self.thread.is_alive()
            and self.session is not None
            and self.config.status == MCPServerStatus.CONNECTED
        )

    def to_status(self) -> Dict[str, Any]:
        """向后兼容的运行状态字典。"""
        tools_list = []
        for namespaced, info in self.tool_infos.items():
            tools_list.append({"name": info.raw_name, "namespaced_name": namespaced, "enabled": True})
        return {
            "name": self.config.name,
            "server_id": self.config.server_id,
            "transport": self.config.transport.value if hasattr(self.config.transport, "value") else str(self.config.transport),
            "enabled": self.config.enabled,
            "running": self.running,
            "status": self.config.status.value if hasattr(self.config.status, "value") else str(self.config.status),
            "error": self._error or self.config.last_error,
            "started_at": self.started_at,
            "tools": tools_list,
            "allowed_tools": self.config.allowed_tools,
        }

    # ── 常驻线程与异步生命周期 ───────────────────────────────────────────
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            loop.run_until_complete(self._connect_and_serve())
        except Exception as exc:
            logger.error("[mcp:%s] loop crashed: %s", self.server_id, exc)
            self._error = str(exc)
            self.config.status = MCPServerStatus.ERROR
            self.config.last_error = str(exc)
        finally:
            self._ready.set()
            try:
                loop.close()
            except Exception:
                pass

    async def _connect_and_serve(self) -> None:
        stack = AsyncExitStack()
        self._exit_stack = stack
        self.config.status = MCPServerStatus.CONNECTING
        self.config.last_error = None

        try:
            session = await self._establish_connection(stack)
            self.session = session
            self.config.status = MCPServerStatus.CONNECTED
            self.config.running = True
            self.started_at = time.time()

            # 拉取工具并挂载
            await self._refresh_and_register_tools()
            self._ready.set()

            # 保持 loop 存活直至收到退出信号
            while not self._stop_requested.is_set():
                await asyncio.sleep(0.2)

        except Exception as exc:
            logger.error("[mcp:%s] connection failed: %s", self.server_id, exc)
            self._error = str(exc)
            self.config.status = MCPServerStatus.ERROR
            self.config.last_error = str(exc)
            self.config.running = False
            self._ready.set()
        finally:
            await self._cleanup()

    async def _establish_connection(self, stack: AsyncExitStack) -> Any:
        from mcp import ClientSession

        cfg = self.config
        transport_type = cfg.transport

        headers = dict(cfg.headers or {})
        if cfg.auth_token:
            headers["Authorization"] = f"Bearer {cfg.auth_token}"

        # 1. 优先支持远程 SSE
        if transport_type == MCPTransportType.SSE:
            from mcp.client.sse import sse_client
            if not cfg.url:
                raise ValueError(f"MCP Server '{cfg.server_id}' 采用 SSE 模式，必须提供有效 url")
            read_stream, write_stream = await stack.enter_async_context(
                sse_client(cfg.url, headers=headers)
            )

        # 2. 远程 Streamable HTTP
        elif transport_type == MCPTransportType.STREAMABLE_HTTP:
            from mcp.client.streamable_http import streamable_http_client
            if not cfg.url:
                raise ValueError(f"MCP Server '{cfg.server_id}' 采用 Streamable HTTP 模式，必须提供有效 url")
            read_stream, write_stream = await stack.enter_async_context(
                streamable_http_client(cfg.url, headers=headers)
            )

        # 3. 本地受限 Stdio
        elif transport_type == MCPTransportType.STDIO:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            command = list(cfg.stdio_command or [])
            if not command:
                raise ValueError(f"MCP Server '{cfg.server_id}' 采用 stdio 模式，必须提供 stdio_command")

            # 安全防御：校验可执行文件是否在安全白名单内
            bin_name = command[0].lower().replace(".exe", "").split("\\")[-1].split("/")[-1]
            if bin_name not in TRUSTED_STDIO_BINARIES:
                raise PermissionError(
                    f"安全拒绝：程序 '{command[0]}' 不在受信任白名单 {TRUSTED_STDIO_BINARIES} 中"
                )

            # Python 环境隔离：确保使用主进程相同解释器
            if bin_name in ("python", "python3", "py"):
                command[0] = sys.executable

            # 子进程环境：与父进程合并（直接传 env 会丢掉 PATH/NODE_* 等，
            # 导致 npx/node 找不到），再叠加服务自己声明的 API Key 等变量。
            params_kwargs: Dict[str, Any] = {"command": command[0], "args": command[1:]}
            extra_env = {str(k): str(v) for k, v in (cfg.env or {}).items() if str(v)}
            if extra_env:
                params_kwargs["env"] = {**os.environ, **extra_env}
            if cfg.cwd:
                params_kwargs["cwd"] = cfg.cwd
            params = StdioServerParameters(**params_kwargs)
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))

        else:
            raise ValueError(f"不支持的传输协议: {transport_type}")

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    async def _cleanup(self) -> None:
        self.config.status = MCPServerStatus.DISCONNECTED
        self.config.running = False
        self.started_at = None
        self._unregister_tools()
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.debug("[mcp:%s] exit stack closed with: %s", self.server_id, exc)
            finally:
                self._exit_stack = None
                self.session = None

    # ── 工具发现与注册 ──────────────────────────────────────────────────
    async def _refresh_and_register_tools(self) -> None:
        """从远程 Session 拉取工具并挂载到 ToolRegistry。"""
        if not self.session:
            return

        self._unregister_tools()
        tools_res = await self.session.list_tools()
        cfg = self.config

        allowed = [
            t for t in tools_res.tools
            if not cfg.allowed_tools or "*" in cfg.allowed_tools or t.name in cfg.allowed_tools
        ]

        reg = get_tool_registry()
        for t in allowed:
            namespaced_name = _mcp_tool_name(cfg.server_id, t.name)
            legacy_name = _legacy_mcp_tool_name(cfg.server_id, t.name)
            input_schema = getattr(t, "inputSchema", {}) or {}

            # 记录元数据
            info = MCPToolInfo(
                namespaced_name=namespaced_name,
                raw_name=t.name,
                server_id=cfg.server_id,
                description=getattr(t, "description", "") or f"MCP tool from {cfg.server_id}",
                input_schema=input_schema,
                capabilities=cfg.capabilities,
            )
            self.tool_infos[namespaced_name] = info

            # 解析参数签名
            arg_schema: Dict[str, Any] = {}
            props = input_schema.get("properties") or {}
            for arg_name, meta in props.items():
                if arg_name.startswith("_"):
                    continue
                arg_schema[arg_name] = (
                    str(meta.get("type", "string")),
                    str(meta.get("description", "")),
                    arg_name in (input_schema.get("required") or []),
                )

            # 构造同步调用闭包
            def make_call_fn(raw_tool_name: str, handle_ref: MCPServerHandle):
                def fn(**kwargs: Any) -> str:
                    return handle_ref.call_tool_sync(raw_tool_name, kwargs)
                return fn

            meta_dict = _mcp_tool_metadata(
                t.name,
                getattr(t, "description", "") or "",
                cfg.capabilities,
                server_id=cfg.server_id,
            )

            # 1. 注册标准双下划线命名空间工具：{server_id}__{tool_name}
            reg.register(
                ToolDefinition(
                    name=namespaced_name,
                    description=getattr(t, "description", "") or f"MCP tool {t.name}",
                    fn=make_call_fn(t.name, self),
                    arg_schema=arg_schema,
                    check_fn=lambda: self.running,
                    timeout_s=0,
                    metadata=meta_dict,
                )
            )
            self.registered_tools.append(namespaced_name)

            # 2. 注册兼容别名工具：mcp_{server_id}_{tool_name}
            reg.register(
                ToolDefinition(
                    name=legacy_name,
                    description=getattr(t, "description", "") or f"MCP tool {t.name}",
                    fn=make_call_fn(t.name, self),
                    arg_schema=arg_schema,
                    check_fn=lambda: self.running,
                    timeout_s=0,
                    metadata=meta_dict,
                )
            )
            self.registered_tools.append(legacy_name)

        cfg.registered_tools_count = len(self.tool_infos)
        logger.info(
            "[mcp:%s] registered %d tools (primary: %s__*, alias: mcp_%s_*)",
            cfg.server_id, len(self.tool_infos), cfg.server_id, cfg.server_id,
        )

    def _unregister_tools(self) -> None:
        """从全局 registry 中注销工具。"""
        if not self.registered_tools:
            return
        reg = get_tool_registry()
        for name in self.registered_tools:
            try:
                reg.unregister(name)
            except Exception:
                pass
        self.registered_tools.clear()
        self.tool_infos.clear()
        self.config.registered_tools_count = 0

    # ── 执行派发与健壮性异常拦截 ─────────────────────────────────────────
    def call_tool_sync(self, raw_tool_name: str, arguments: Dict[str, Any]) -> str:
        """同步派发调用：提交到常驻 loop，内置超时与友好错误拦截。"""
        if not self.running or self.session is None or self.loop is None:
            return f"[MCP Error] 服务 '{self.server_id}' 未连接或已离线，无法执行 '{raw_tool_name}'"

        timeout_s = self.config.timeout_s or 30

        async def _call():
            result = await self.session.call_tool(raw_tool_name, arguments)
            if getattr(result, "isError", False):
                err = _text_content(result) or f"MCP 工具 '{raw_tool_name}' 返回异常"
                return f"[MCP Tool Error] {err}"
            return _text_content(result) or f"(工具 '{raw_tool_name}' 执行完成但无文本返回)"

        fut = asyncio.run_coroutine_threadsafe(_call(), self.loop)
        try:
            return fut.result(timeout=timeout_s)
        except asyncio.TimeoutError:
            fut.cancel()
            return f"[MCP Timeout] 工具 '{raw_tool_name}' 在 {timeout_s} 秒后超时，请简化请求或稍后重试。"
        except Exception as exc:
            logger.error("[mcp:%s] invoke %s error: %s", self.server_id, raw_tool_name, exc)
            return f"[MCP Execution Error] 调用失败: {exc}。请检查参数是否符合要求。"


# ── 全局统一 MCP 管理器 ───────────────────────────────────────────────────

class MCPManager:
    """企业级 MCP 服务聚合管理器：全生命周期管理、动态热插拔、配置持久化。"""

    def __init__(self):
        self.servers: Dict[str, MCPServerHandle] = {}
        self._lock = threading.Lock()
        # 初始化加载预置服务定义
        for defn in load_mcp_server_definitions():
            self.servers[defn.server_id] = MCPServerHandle(defn)

    # ── 动态配置热挂载与 CRUD ───────────────────────────────────────────
    def register(self, item: MCPServerCreate | MCPServerDefinition, persist: bool = True) -> MCPServerDefinition:
        """【动态创建】注册新 MCP Server。已存在则平滑替换并重连。"""
        with self._lock:
            sid = item.server_id
            if sid in self.servers:
                self.stop(sid)

            if isinstance(item, MCPServerDefinition):
                defn = item
            else:
                defn = MCPServerDefinition(**item.model_dump())

            handle = MCPServerHandle(defn)
            self.servers[sid] = handle

            if defn.enabled:
                self.start(sid, wait=False)

            if persist:
                self.persist_all()
            return defn

    def unregister(self, server_id: str, persist: bool = True) -> None:
        """【动态删除】卸载并安全断开指定 MCP Server。"""
        with self._lock:
            if server_id not in self.servers:
                raise KeyError(f"MCP server '{server_id}' not found")
            self.stop(server_id)
            del self.servers[server_id]

            if persist:
                self.persist_all()

    def update(self, server_id: str, patch: MCPServerUpdate, persist: bool = True) -> MCPServerDefinition:
        """【动态更新】修改 MCP Server 配置并按需重连。"""
        with self._lock:
            handle = self.servers.get(server_id)
            if not handle:
                raise KeyError(f"MCP server '{server_id}' not found")

            # 停止原连接
            was_running = handle.running
            self.stop(server_id)

            data = handle.config.model_dump()
            update_data = patch.model_dump(exclude_unset=True)
            data.update(update_data)
            data["updated_at"] = time.time()

            new_defn = MCPServerDefinition(**data)
            new_handle = MCPServerHandle(new_defn)
            self.servers[server_id] = new_handle

            if new_defn.enabled and was_running:
                self.start(server_id, wait=False)

            if persist:
                self.persist_all()
            return new_defn

    def persist_all(self) -> None:
        """将当前内存中的所有服务配置持久化到配置文件。"""
        definitions = [h.config for h in self.servers.values()]
        try:
            save_mcp_server_definitions(definitions)
        except Exception as exc:
            logger.error("Failed to persist MCP definitions: %s", exc)

    # ── 启停管理 ────────────────────────────────────────────────────────
    def start(self, server_id: str, wait: bool = True, timeout: float = 30.0) -> Dict[str, Any]:
        """启动指定 server 连接。幂等安全。"""
        handle = self.servers.get(server_id)
        if handle is None:
            raise KeyError(f"MCP server '{server_id}' not configured")
        if handle.running:
            return handle.to_status()

        handle._error = None
        handle._stop_requested.clear()
        handle._ready.clear()
        handle.started_at = None
        handle.registered_tools = []
        handle.tool_infos.clear()

        handle.thread = threading.Thread(
            target=handle._run_loop, name=f"mcp-{server_id}", daemon=True
        )
        handle.thread.start()

        if wait:
            if not handle._ready.wait(timeout):
                raise TimeoutError(f"MCP server '{server_id}' did not become ready in {timeout}s")
            if handle.session is None and handle._error:
                raise RuntimeError(f"MCP server '{server_id}' failed: {handle._error}")
        return handle.to_status()

    def start_all(self, wait: bool = False) -> Dict[str, Any]:
        """随应用启动所有 enabled 的 server。"""
        results = {}
        for sid, handle in list(self.servers.items()):
            if handle.config.enabled:
                try:
                    results[sid] = self.start(sid, wait=wait)
                except Exception as exc:
                    results[sid] = {"server_id": sid, "running": False, "error": str(exc)}
        return results

    def stop(self, server_id: str) -> Dict[str, Any]:
        """优雅关闭指定 server：发信号 -> 断开 ExitStack -> 注销工具。"""
        handle = self.servers.get(server_id)
        if handle is None:
            raise KeyError(f"MCP server '{server_id}' not configured")

        handle._stop_requested.set()
        handle._unregister_tools()

        if handle.thread and handle.thread.is_alive():
            handle.thread.join(timeout=5)

        handle.started_at = None
        handle.config.running = False
        handle.config.status = MCPServerStatus.DISCONNECTED
        return handle.to_status()

    def stop_all(self) -> None:
        """应用关闭时安全清理全部会话。"""
        for sid in list(self.servers.keys()):
            try:
                self.stop(sid)
            except Exception as exc:
                logger.warning("[mcp] stop %s error: %s", sid, exc)

    def sync_tools(self, server_id: str) -> Dict[str, Any]:
        """重新拉取指定 Server 的工具列表。"""
        handle = self.servers.get(server_id)
        if not handle or not handle.running:
            raise RuntimeError(f"MCP server '{server_id}' 未在运行状态，无法同步工具")

        async def _refresh():
            await handle._refresh_and_register_tools()

        fut = asyncio.run_coroutine_threadsafe(_refresh(), handle.loop)
        fut.result(timeout=10)
        return handle.to_status()

    # ── 查询与工具过滤 (Tool Routing) ───────────────────────────────────
    def status(self) -> List[Dict[str, Any]]:
        return [h.to_status() for h in self.servers.values()]

    def list_definitions(self) -> List[MCPServerDefinition]:
        return [h.config for h in self.servers.values()]

    def get_definition(self, server_id: str) -> MCPServerDefinition:
        handle = self.servers.get(server_id)
        if not handle:
            raise KeyError(f"MCP server '{server_id}' not found")
        return handle.config

    def get(self, server_id: str) -> MCPServerHandle:
        handle = self.servers.get(server_id)
        if not handle:
            raise KeyError(f"MCP server '{server_id}' not found")
        return handle

    def get_tools(self, server_ids: Optional[List[str]] = None) -> List[MCPToolInfo]:
        """获取带命名空间的工具列表（支持 Tool Routing 服务白名单过滤）。"""
        tools: List[MCPToolInfo] = []
        for sid, handle in self.servers.items():
            if not handle.running:
                continue
            if server_ids is not None and sid not in server_ids:
                continue
            tools.extend(handle.tool_infos.values())
        return tools


# 全局单例
_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
