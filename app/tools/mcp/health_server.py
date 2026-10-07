"""健康检查 MCP server — 检测 Postgres / Redis / Milvus / Neo4j 连通性。

提供四个工具：
  - check_postgres: 检查 PostgreSQL 连接状态、版本、活跃连接数
  - check_redis:    检查 Redis 连接状态、内存使用
  - check_milvus:   检查 Milvus 连接状态、集合列表及行数
  - check_neo4j:    检查 Neo4j 连接状态、节点/关系数量

双模式运行：
  python -m app.tools.mcp.health_server            # stdio 模式（默认）
  python -m app.tools.mcp.health_server --http      # Streamable HTTP 模式（:8901/mcp）

用 mcp SDK 低层 API（mcp.server.lowlevel.Server），不依赖 FastMCP 等高层封装。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.types import CallToolRequest, ListToolsRequest, TextContent, Tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp.health")

SERVER_NAME = "system-health"
SERVER_VERSION = "0.1.0"

# 默认连接超时（秒）
CONNECTION_TIMEOUT = 5

# ── 工具定义 ──────────────────────────────────────────────────────────────

TOOLS: List[Tool] = [
    Tool(
        name="check_postgres",
        description="检查 PostgreSQL 连接状态，返回版本号和活跃连接数",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="check_redis",
        description="检查 Redis 连接状态，执行 PING 并返回内存使用信息",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="check_milvus",
        description="检查 Milvus 连接状态，列出集合及其行数",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="check_neo4j",
        description="检查 Neo4j 连接状态，返回节点数和关系数",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ── 工具实现 ──────────────────────────────────────────────────────────────

def _check_postgres() -> str:
    """检查 PostgreSQL 健康状态。"""
    try:
        import psycopg2  # type: ignore
    except ImportError:
        return json.dumps({"status": "error", "message": "psycopg2 library not installed"}, ensure_ascii=False)

    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    dbname = os.getenv("POSTGRES_DB", "postgres")

    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname,
            connect_timeout=CONNECTION_TIMEOUT,
        )
        cur = conn.cursor()

        cur.execute("SELECT version();")
        version = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM pg_stat_activity;")
        active_connections = cur.fetchone()[0]

        cur.close()
        conn.close()

        return json.dumps({
            "status": "healthy",
            "host": host,
            "port": port,
            "database": dbname,
            "version": version,
            "active_connections": active_connections,
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "host": host, "port": port, "message": str(exc)}, ensure_ascii=False)


def _check_redis() -> str:
    """检查 Redis 健康状态。"""
    try:
        import redis  # type: ignore
    except ImportError:
        return json.dumps({"status": "error", "message": "redis library not installed"}, ensure_ascii=False)

    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", None)
    db = int(os.getenv("REDIS_DB", "0"))

    try:
        client = redis.Redis(
            host=host, port=port, password=password, db=db,
            socket_connect_timeout=CONNECTION_TIMEOUT, socket_timeout=CONNECTION_TIMEOUT,
        )

        ping_ok = client.ping()

        info = client.info("memory")
        memory_used = info.get("used_memory_human", "N/A")
        memory_peak = info.get("used_memory_peak_human", "N/A")
        memory_rss = info.get("used_memory_rss_human", "N/A")

        client.close()

        return json.dumps({
            "status": "healthy",
            "host": host,
            "port": port,
            "ping": "PONG" if ping_ok else "FAILED",
            "memory_used": memory_used,
            "memory_peak": memory_peak,
            "memory_rss": memory_rss,
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "host": host, "port": port, "message": str(exc)}, ensure_ascii=False)


def _check_milvus() -> str:
    """检查 Milvus 健康状态。"""
    try:
        from pymilvus import connections, utility  # type: ignore
    except ImportError:
        return json.dumps({"status": "error", "message": "pymilvus library not installed"}, ensure_ascii=False)

    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")

    alias = "_health_check"
    try:
        connections.connect(alias=alias, host=host, port=port, timeout=CONNECTION_TIMEOUT)

        collection_names = utility.list_collections(using=alias, timeout=CONNECTION_TIMEOUT)
        collections_info = []
        for name in collection_names:
            try:
                from pymilvus import Collection  # type: ignore
                col = Collection(name, using=alias)
                collections_info.append({"name": name, "row_count": col.num_entities})
            except Exception:
                collections_info.append({"name": name, "row_count": "N/A"})

        connections.disconnect(alias)

        return json.dumps({
            "status": "healthy",
            "host": host,
            "port": port,
            "collection_count": len(collection_names),
            "collections": collections_info,
        }, ensure_ascii=False)
    except Exception as exc:
        try:
            connections.disconnect(alias)
        except Exception:
            pass
        return json.dumps({"status": "error", "host": host, "port": port, "message": str(exc)}, ensure_ascii=False)


def _check_neo4j() -> str:
    """检查 Neo4j 健康状态。"""
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        return json.dumps({"status": "error", "message": "neo4j library not installed"}, ensure_ascii=False)

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")

    try:
        driver = GraphDatabase.driver(
            uri, auth=(user, password),
            connection_timeout=CONNECTION_TIMEOUT, max_transaction_retry_time=CONNECTION_TIMEOUT,
        )

        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

        driver.close()

        return json.dumps({
            "status": "healthy",
            "uri": uri,
            "user": user,
            "node_count": node_count,
            "relationship_count": rel_count,
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "uri": uri, "message": str(exc)}, ensure_ascii=False)


# ── 工具分发 ──────────────────────────────────────────────────────────────

def _handle_tool_call(name: str, args: Dict[str, Any]) -> str:
    """执行工具并返回文本结果。"""
    if name == "check_postgres":
        return _check_postgres()
    if name == "check_redis":
        return _check_redis()
    if name == "check_milvus":
        return _check_milvus()
    if name == "check_neo4j":
        return _check_neo4j()
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


def run_http(host: str = "127.0.0.1", port: int = 8901) -> None:
    """Streamable HTTP 模式：起一个独立 HTTP 服务。"""
    import uvicorn

    server = build_server()
    app = server.streamable_http_app()
    logger.info("health MCP server (HTTP) listening on %s:%s/mcp", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(description="EasyRAG 健康检查 MCP server")
    parser.add_argument("--http", action="store_true", help="以 Streamable HTTP 模式运行（默认 stdio）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8901)
    args = parser.parse_args()

    if args.http:
        run_http(args.host, args.port)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
