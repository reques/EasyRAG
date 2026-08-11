"""演示 MCP server — 零外部依赖，验证 MCP 接入全链路。

提供两个简单工具：
  - echo(message): 原样返回输入（验证参数传递）
  - get_time(): 返回当前时间（验证无参调用）

双模式运行：
  python -m app.tools.mcp.demo_server            # stdio 模式（默认）
  python -m app.tools.mcp.demo_server --http     # Streamable HTTP 模式（:8900/mcp）

用 mcp SDK 低层 API（mcp.server.lowlevel.Server），不依赖 FastMCP 等高层封装，
保证 stage1-agent 环境装了 mcp 包即可运行。
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from typing import Any, Dict, List

from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.types import CallToolRequest, ListToolsRequest, TextContent, Tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp.demo")

SERVER_NAME = "demo"
SERVER_VERSION = "0.1.0"

# ── 工具定义 ──────────────────────────────────────────────────────────────

TOOLS: List[Tool] = [
    Tool(
        name="echo",
        description="原样返回传入的 message 参数",
        inputSchema={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "要回显的文本"}},
            "required": ["message"],
        },
    ),
    Tool(
        name="get_time",
        description="返回当前日期时间（ISO 格式）",
        inputSchema={"type": "object", "properties": {}},
    ),
]


def _handle_tool_call(name: str, args: Dict[str, Any]) -> str:
    """执行工具并返回文本结果。"""
    if name == "echo":
        return f"echo: {args.get('message', '')}"
    if name == "get_time":
        return datetime.datetime.now().isoformat()
    raise ValueError(f"Unknown tool: {name}")


# ── MCP server 装配 ────────────────────────────────────────────────────────

def build_server() -> Server:
    from mcp.types import CallToolResult, ListToolsResult, TextContent

    async def on_list_tools(ctx, params) -> ListToolsResult:
        return ListToolsResult(tools=TOOLS)

    async def on_call_tool(ctx, params) -> CallToolResult:
        name = params.name
        args = params.arguments or {}
        logger.info("call_tool: %s %s", name, args)
        try:
            result = _handle_tool_call(name, args)
            return CallToolResult(content=[TextContent(type="text", text=result)], isError=False)
        except Exception as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {exc}")], isError=True
            )

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def _init_options() -> InitializationOptions:
    return InitializationOptions(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        capabilities={"tools": {}},
    )


def run_stdio() -> None:
    """stdio 模式：子进程通过 stdin/stdout 与父进程通信。"""
    import anyio

    from mcp.server.stdio import stdio_server

    server = build_server()

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, _init_options())

    anyio.run(_main)


def run_http(host: str = "127.0.0.1", port: int = 8900) -> None:
    """Streamable HTTP 模式：起一个独立 HTTP 服务。"""
    import uvicorn

    server = build_server()
    app = server.streamable_http_app()
    logger.info("demo MCP server (HTTP) listening on %s:%s/mcp", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(description="EasyRAG 演示 MCP server")
    parser.add_argument("--http", action="store_true", help="以 Streamable HTTP 模式运行（默认 stdio）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    args = parser.parse_args()

    if args.http:
        run_http(args.host, args.port)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
