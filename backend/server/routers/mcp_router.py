"""MCP 外部工具服务管理 API — 统一启停与权限查询。

端点：
  GET  /mcp/servers                 所有 server 配置 + 运行状态
  POST /mcp/servers/{name}/start    启动指定 server（幂等）
  POST /mcp/servers/{name}/stop     停止指定 server（幂等）
  GET  /mcp/servers/{name}/tools    该 server 已注册（可被 AI 调用）的工具
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.tools.mcp.manager import get_mcp_manager

logger = get_logger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])


def _get_manager():
    return get_mcp_manager()


@router.get("/servers")
async def list_servers():
    """列出所有配置的 MCP server 及运行状态。"""
    return {"servers": _get_manager().status()}


@router.post("/servers/{name}/start")
async def start_server(name: str):
    """启动指定 MCP server（连接 + 注册工具）。已运行则幂等返回。"""
    try:
        status = _get_manager().start(name, wait=True, timeout=30)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return status


@router.post("/servers/{name}/stop")
async def stop_server(name: str):
    """停止指定 MCP server（断开连接 + 注销工具）。"""
    try:
        status = _get_manager().stop(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")
    return status


@router.get("/servers/{name}/tools")
async def server_tools(name: str):
    """该 server 当前已注册（可被 AI 调用）的工具列表。"""
    manager = _get_manager()
    try:
        handle = manager.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")
    status = handle.to_status()
    return {"name": name, "running": status["running"], "tools": status["tools"]}
