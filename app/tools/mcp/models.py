"""MCP (Model Context Protocol) 生产级数据模型定义 (Pydantic v2)。

对齐 Yuxi (xerrors/Yuxi) 企业级架构：
1. 明确传输协议枚举：SSE、Streamable HTTP、受限 Stdio；
2. 动态服务配置模型：支持 URL、Headers、Bearer Token、超时与权限控制；
3. stdio 安全白名单校验：防止任意子进程命令注入 (RCE 防御)；
4. 命名空间隔离工具模型：{server_id}__{tool_name}。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MCPTransportType(str, Enum):
    """MCP 传输模式协议枚举。"""
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class MCPServerStatus(str, Enum):
    """MCP 服务连接与生命周期状态。"""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


# 受信任的本地可执行文件白名单（stdio 模式防御性校验）
TRUSTED_STDIO_BINARIES = {"python", "python3", "py", "node", "npx", "uv", "uvx"}


class MCPServerBase(BaseModel):
    """MCP Server 基础字段定义。"""
    server_id: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="服务唯一标识，用于工具命名空间隔离前缀 (例如: github, db_service)",
    )
    name: str = Field(..., description="服务展示名称")
    description: Optional[str] = Field(default="", description="服务描述")
    transport: MCPTransportType = Field(
        default=MCPTransportType.SSE,
        description="传输协议，生产推荐远程 sse 或 streamable_http",
    )

    # 远程模式参数 (SSE / HTTP)
    url: Optional[str] = Field(
        default=None,
        description="远程 MCP 服务端点 URL（SSE 或 Streamable HTTP 必填）",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="自定义 HTTP 请求头（如 API Key 等）",
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="可选身份凭据，会自动组装为 'Authorization: Bearer <token>' 请求头",
    )

    # stdio 限制：只允许系统预置的白名单受信任程序
    stdio_command: Optional[List[str]] = Field(
        default=None,
        description="仅允许配置预设白名单内的受信任命令，例如 ['python', '-m', '...']",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "stdio 子进程的额外环境变量（如 API Key）。"
            "运行时与父进程环境合并，不会丢失 PATH 等基础变量。"
        ),
    )
    cwd: Optional[str] = Field(
        default=None,
        description="stdio 子进程的工作目录；留空则继承后端进程的工作目录",
    )

    # 权限与治理
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["*"],
        description="允许暴露的工具名称白名单；['*'] 表示全量允许，[] 表示全量禁止",
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="沙箱权限标签列表（如 ['db.read', 'fs.read']），自动继承给该服务注册的工具",
    )
    enabled: bool = Field(default=True, description="是否随服务默认激活连接")
    timeout_s: int = Field(default=30, ge=5, le=300, description="单次 Tool 调用超时时间(秒)")
    source: Optional[str] = Field(
        default="",
        description="来源标记：MCP 广场安装时记录目录 id（如 modelscope:@amap/amap-maps），手工创建为空",
    )

    @field_validator("stdio_command")
    @classmethod
    def validate_stdio_safety(cls, cmd: Optional[List[str]], info):
        if not cmd:
            return cmd
        bin_name = cmd[0].lower().replace(".exe", "").split("\\")[-1].split("/")[-1]
        if bin_name not in TRUSTED_STDIO_BINARIES:
            raise ValueError(
                f"安全限制：stdio 模式仅允许执行受信任程序 {TRUSTED_STDIO_BINARIES}，拒绝执行: {cmd[0]}"
            )
        return cmd

    @field_validator("url")
    @classmethod
    def validate_remote_url(cls, v: Optional[str], info):
        transport = info.data.get("transport")
        if transport in (MCPTransportType.SSE, MCPTransportType.STREAMABLE_HTTP) and not v:
            raise ValueError(f"当传输模式为 {transport} 时，必须提供有效的 url")
        return v


class MCPServerCreate(MCPServerBase):
    """创建 MCP Server 接口入参模型。"""
    pass


class MCPServerUpdate(BaseModel):
    """更新 MCP Server 接口入参模型。"""
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    enabled: Optional[bool] = None
    timeout_s: Optional[int] = Field(default=None, ge=5, le=300)
    env: Optional[Dict[str, str]] = Field(default=None, description="覆盖 stdio 子进程的额外环境变量")


class MCPServerDefinition(MCPServerBase):
    """持久化的 MCP Server 完整定义及运行状态。"""
    status: MCPServerStatus = Field(default=MCPServerStatus.DISCONNECTED)
    running: bool = Field(default=False, description="当前是否保持活跃连接")
    last_error: Optional[str] = None
    registered_tools_count: int = Field(default=0, description="当前已成功注册的工具数量")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MCPToolInfo(BaseModel):
    """发现并经过命名空间隔离后的工具元数据。"""
    namespaced_name: str = Field(..., description="带命名空间的唯一工具名，例如: {server_id}__{tool_name}")
    raw_name: str = Field(..., description="MCP Server 原始声明的工具名")
    server_id: str = Field(..., description="所属 MCP Server 标识")
    description: str = Field(default="", description="工具功能描述")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="工具参数 JSON Schema")
    capabilities: List[str] = Field(default_factory=list, description="继承的沙箱权限能力标签")

