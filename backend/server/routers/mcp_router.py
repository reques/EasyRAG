"""MCP 外部工具服务管理 API — 企业级动态 CRUD、统一启停与工具路由查询。

端点清单：
  GET    /mcp/servers                 获取所有 server 配置与实时运行状态
  POST   /mcp/servers                 【新增】动态添加并热挂载新的 MCP Server (支持 SSE / HTTP)
  GET    /mcp/servers/{server_id}     【新增】获取指定 MCP Server 的详细配置
  PUT    /mcp/servers/{server_id}     【新增】动态修改 MCP Server 配置并重连
  DELETE /mcp/servers/{server_id}     【新增】动态卸载并释放指定 MCP Server
  POST   /mcp/servers/{name}/start    启动指定 server（幂等）
  POST   /mcp/servers/{name}/stop     停止指定 server（幂等）
  POST   /mcp/servers/{name}/sync     【新增】动态重新拉取并同步指定 server 的工具
  GET    /mcp/servers/{name}/tools    该 server 已注册的工具列表
  GET    /mcp/tools                   【新增】全量/白名单过滤的命名空间工具清单 (Tool Routing)
  GET    /mcp/catalog                 【新增】ModelScope MCP 广场检索（关键词 + 分页）
  GET    /mcp/catalog/{catalog_id}    【新增】广场服务详情 + 安装计划
  POST   /mcp/catalog/{catalog_id}/install 【新增】按安装计划把广场服务接入本地 Agent
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.tools.mcp.catalog import (
    CatalogError,
    get_catalog_client,
    missing_env_keys,
    plan_to_create_payload,
)
from app.tools.mcp.manager import get_mcp_manager
from app.tools.mcp.models import (
    MCPServerCreate,
    MCPServerDefinition,
    MCPServerUpdate,
    MCPToolInfo,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])

# 广场 id 允许的字符集（@publisher/name）；用于挡住路径穿越等畸形输入
_CATALOG_ID_RE = re.compile(r"^[A-Za-z0-9@/._\-]{1,200}$")


class MCPCatalogInstall(BaseModel):
    """从 MCP 广场安装服务的入参：安装计划提供默认值，这里只覆盖用户选择。"""

    server_id: Optional[str] = Field(default=None, description="留空则用广场 id 推导的命名空间前缀")
    name: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict, description="广场要求的 API Key 等环境变量")
    transport: Optional[str] = Field(default=None, description="覆盖安装计划推断的传输协议")
    url: Optional[str] = Field(default=None, description="计划未提供端点时手动指定（如托管部署后的 URL）")
    timeout_s: int = Field(default=30, ge=5, le=300)
    allowed_tools: Optional[List[str]] = Field(default=None, description="留空 = ['*'] 全量放行")
    capabilities: Optional[List[str]] = None
    enabled: bool = True
    overwrite: bool = Field(default=False, description="同名 server_id 已存在时是否覆盖")



def _get_manager():
    return get_mcp_manager()


# ── Server 配置管理 (CRUD) ────────────────────────────────────────────────

@router.get("/servers")
async def list_servers():
    """列出所有配置的 MCP server 及运行状态。"""
    mgr = _get_manager()
    return {
        "servers": mgr.status(),
        "definitions": mgr.list_definitions(),
    }


@router.post("/servers", response_model=MCPServerDefinition, status_code=status.HTTP_201_CREATED)
async def create_server(payload: MCPServerCreate):
    """【新增】动态添加并立即热挂载远程 MCP Server (SSE / HTTP / 受信任 Stdio)。"""
    mgr = _get_manager()
    try:
        return mgr.register(payload, persist=True)
    except Exception as exc:
        logger.error("Failed to register MCP server '%s': %s", payload.server_id, exc)
        raise HTTPException(status_code=400, detail=f"注册 MCP Server 失败: {exc}")


@router.get("/servers/{server_id}", response_model=MCPServerDefinition)
async def get_server(server_id: str):
    """【新增】获取指定 MCP Server 的完整配置。"""
    mgr = _get_manager()
    try:
        return mgr.get_definition(server_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")


@router.put("/servers/{server_id}", response_model=MCPServerDefinition)
async def update_server(server_id: str, patch: MCPServerUpdate):
    """【新增】更新指定 MCP Server 的配置参数并按需重连。"""
    mgr = _get_manager()
    try:
        return mgr.update(server_id, patch, persist=True)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"更新 MCP Server 失败: {exc}")


@router.delete("/servers/{server_id}", status_code=status.HTTP_200_OK)
async def delete_server(server_id: str):
    """【新增】动态卸载指定 MCP Server，注销工具并释放连接。"""
    mgr = _get_manager()
    try:
        mgr.unregister(server_id, persist=True)
        return {"success": True, "message": f"MCP server '{server_id}' 已成功卸载"}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")


# ── 启停与同步控制 ────────────────────────────────────────────────────────

@router.post("/servers/{name}/start")
async def start_server(name: str):
    """启动指定 MCP server（连接 + 注册工具）。已运行则幂等返回。"""
    try:
        return _get_manager().start(name, wait=True, timeout=30)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/servers/{name}/stop")
async def stop_server(name: str):
    """停止指定 MCP server（断开连接 + 注销工具）。"""
    try:
        return _get_manager().stop(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")


@router.post("/servers/{name}/sync")
async def sync_server_tools(name: str):
    """【新增】动态重新拉取并同步指定 MCP Server 的工具列表。"""
    try:
        return _get_manager().sync_tools(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── 工具发现与路由查询 (Tool Routing) ───────────────────────────────────

@router.get("/servers/{name}/tools")
async def server_tools(name: str):
    """该 server 当前已注册的工具列表。"""
    manager = _get_manager()
    try:
        handle = manager.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not configured")
    status_info = handle.to_status()
    return {"name": name, "running": status_info["running"], "tools": status_info["tools"]}


@router.get("/tools", response_model=List[MCPToolInfo])
async def list_all_tools(server_ids: Optional[str] = Query(None, description="逗号分隔的 server_id 白名单过滤")):
    """【新增】获取全部可调用的 MCP 工具，支持基于服务 ID 的按需白名单过滤。"""
    sids = [s.strip() for s in server_ids.split(",") if s.strip()] if server_ids else None
    return _get_manager().get_tools(server_ids=sids)


# ── MCP 广场（ModelScope 目录）────────────────────────────────────────────

def _catalog_client():
    return get_catalog_client()


def _validate_catalog_id(catalog_id: str) -> str:
    catalog_id = str(catalog_id or "").strip()
    if not _CATALOG_ID_RE.match(catalog_id):
        raise HTTPException(status_code=400, detail=f"非法的广场服务标识: {catalog_id!r}")
    return catalog_id


@router.get("/catalog")
async def search_catalog(
    search: str = Query("", description="关键词，留空返回热门服务"),
    page: int = Query(1, ge=1, le=100),
    page_size: int = Query(12, ge=1, le=100, description="ModelScope 限制 page × page_size ≤ 100"),
):
    """【新增】检索 ModelScope MCP 广场服务（只读代理，不需要凭据）。"""
    try:
        result = await _catalog_client().search(search=search, page=page, page_size=page_size)
    except CatalogError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return result


@router.post("/catalog/{catalog_id:path}/install", response_model=MCPServerDefinition)
async def install_catalog_server(catalog_id: str, payload: MCPCatalogInstall):
    """【新增】把广场服务接入本地：按安装计划构造定义、注册并热挂载。

    计划来源优先级：已部署的托管端点 → server_config 远程 URL → server_config 本地命令。
    计划缺失（需要先在 ModelScope 侧部署）时必须由前端显式提供 ``url``。
    """
    catalog_id = _validate_catalog_id(catalog_id)
    try:
        detail = await _catalog_client().detail(catalog_id)
    except CatalogError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    plan = detail.get("plan") or {}
    if plan.get("kind") == "deploy_required" and not payload.url:
        raise HTTPException(
            status_code=400,
            detail=(
                "该服务未提供可直接运行的 server_config，需要先在 ModelScope 侧部署为远程端点，"
                "或在安装表单里手动填写端点 URL。"
            ),
        )

    runtime = plan.get("runtime") or {}
    if plan.get("kind") == "stdio" and runtime and not runtime.get("trusted"):
        raise HTTPException(
            status_code=400,
            detail=f"服务声明的启动器 {runtime.get('binary')!r} 不在受信任白名单内，拒绝安装。",
        )

    missing = missing_env_keys(plan, payload.env)
    # 用超集计划（含用户手填 url）计算最终载荷
    effective_plan = dict(plan)
    if payload.url:
        effective_plan["url"] = payload.url
        if effective_plan.get("kind") == "deploy_required":
            effective_plan["kind"] = "remote"
    create_payload = plan_to_create_payload(
        effective_plan,
        server_id=payload.server_id,
        env=payload.env,
        transport=payload.transport,
        name=payload.name,
        timeout_s=payload.timeout_s,
        allowed_tools=payload.allowed_tools,
        capabilities=payload.capabilities,
        enabled=payload.enabled,
    )
    if missing and create_payload.get("transport") == "stdio":
        raise HTTPException(
            status_code=400,
            detail=f"该服务需要以下环境变量才能启动：{', '.join(missing)}",
        )

    mgr = _get_manager()
    server_id = create_payload.get("server_id") or ""
    if server_id in mgr.servers and not payload.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"MCP server '{server_id}' 已存在；如需替换请在安装时选择覆盖。",
        )

    try:
        definition = mgr.register(MCPServerCreate(**create_payload), persist=True)
    except Exception as exc:
        logger.error("Failed to install MCP catalog server '%s': %s", catalog_id, exc)
        raise HTTPException(status_code=400, detail=f"安装失败：{exc}")

    logger.info(
        "[mcp-catalog] installed %s as '%s' (transport=%s, kind=%s)",
        catalog_id, definition.server_id, definition.transport.value, plan.get("kind"),
    )
    return definition


@router.get("/catalog/{catalog_id:path}")
async def catalog_detail(catalog_id: str):
    """【新增】广场服务详情 + 可直接用于安装的计划（含必填环境变量与运行环境自检）。"""
    catalog_id = _validate_catalog_id(catalog_id)
    try:
        detail = await _catalog_client().detail(catalog_id)
    except CatalogError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    raw = detail.get("raw") or {}
    readme = str(raw.get("readme") or "")
    return {
        "plan": detail.get("plan") or {},
        "readme": readme[:20000],
        "already_installed": (detail.get("plan") or {}).get("server_id_suggestion") in _get_manager().servers,
    }
