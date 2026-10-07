"""MCP 外部工具服务配置加载。

配置文件：项目根下的 `config/mcp_servers.json`（可被环境变量 MCP_SERVERS_FILE 覆盖）。

格式：
{
  "servers": [
    {
      "name": "demo-stdio",
      "transport": "stdio",                          // stdio | http
      "enabled": true,                               // 默认随应用启动
      "command": ["python", "-m", "app.tools.mcp.demo_server"],   // stdio: 子进程命令
      "cwd": null,                                   // stdio: 可选工作目录
      "env": {},                                     // stdio: 可选环境变量覆盖
      "url": "http://127.0.0.1:8900/mcp",            // http: 服务地址
      "allowed_tools": ["*"]                         // 权限白名单；["*"]=全部，[]=禁止全部
    }
  ]
}
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SERVERS_FILE = str(Path(__file__).resolve().parents[3] / "config" / "mcp_servers.json")

# 匹配 ${VAR} 或 ${VAR:-default} 语法
_ENV_VAR_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _expand_env_vars(val: Any) -> Any:
    """递归展开数据结构中字符串的 ``${VAR}`` / ``${VAR:-default}`` 环境变量引用。

    跨平台兼容：在 JSON 解析后递归处理字符串，避免 Windows 路径反斜杠导致 JSON 解码失败。
    """
    if isinstance(val, str):
        def _replacer(m: re.Match) -> str:
            name = m.group("name")
            default = m.group("default")
            res = os.environ.get(name)
            if res is not None:
                return res
            return default if default is not None else ""

        return _ENV_VAR_RE.sub(_replacer, val)
    elif isinstance(val, list):
        return [_expand_env_vars(item) for item in val]
    elif isinstance(val, dict):
        return {k: _expand_env_vars(v) for k, v in val.items()}
    return val


@dataclass
class MCPServerConfig:
    """单个 MCP server 的声明配置。"""

    name: str
    transport: str  # "stdio" | "http"
    enabled: bool = True
    # stdio 模式
    command: List[str] = field(default_factory=list)
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    # http 模式
    url: str = ""
    # 权限白名单：工具名列表；["*"] 表示该 server 的全部工具
    allowed_tools: List[str] = field(default_factory=lambda: ["*"])
    # Sandboxing capability labels inherited by every bridged MCP tool.
    capabilities: List[str] = field(default_factory=list)

    @property
    def is_stdio(self) -> bool:
        return self.transport == "stdio"

    def allows(self, tool_name: str) -> bool:
        """工具级权限判断：白名单含 "*" 或具体工具名才允许。"""
        if not self.allowed_tools:
            return False
        return "*" in self.allowed_tools or tool_name in self.allowed_tools


def load_mcp_servers(path: Optional[str] = None) -> List[MCPServerConfig]:
    """从 JSON 文件加载 MCP server 配置列表。文件不存在或损坏时返回空列表。"""
    file_path = path or os.environ.get("MCP_SERVERS_FILE") or DEFAULT_SERVERS_FILE
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        data = _expand_env_vars(data)
    except FileNotFoundError:
        logger.info("MCP servers file %s not found — no external tools configured", file_path)
        return []
    except json.JSONDecodeError as exc:
        logger.error("MCP servers file %s is invalid JSON: %s", file_path, exc)
        return []

    servers: List[MCPServerConfig] = []
    for item in data.get("servers", []):
        name = item.get("name", "")
        if not name:
            logger.warning("[mcp] skipping server entry without name: %s", item)
            continue
        servers.append(
            MCPServerConfig(
                name=name,
                transport=item.get("transport", "stdio"),
                enabled=bool(item.get("enabled", True)),
                command=[str(c) for c in item.get("command", [])],
                cwd=item.get("cwd"),
                env={str(k): str(v) for k, v in item.get("env", {}).items()},
                url=item.get("url", ""),
                allowed_tools=[str(t) for t in item.get("allowed_tools", ["*"])],
                capabilities=[str(c) for c in item.get("capabilities", [])],
            )
        )
    logger.info("MCP servers loaded from %s: %s", file_path, [s.name for s in servers])
    return servers


def load_mcp_server_definitions(path: Optional[str] = None) -> List["MCPServerDefinition"]:
    """加载配置并返回统一的 Pydantic MCPServerDefinition 模型列表。"""
    from app.tools.mcp.models import MCPServerDefinition, MCPTransportType

    file_path = path or os.environ.get("MCP_SERVERS_FILE") or DEFAULT_SERVERS_FILE
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        data = _expand_env_vars(data)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        logger.error("MCP servers file %s is invalid JSON: %s", file_path, exc)
        return []

    definitions: List[MCPServerDefinition] = []
    for item in data.get("servers", []):
        sid = str(item.get("server_id") or item.get("name") or "").strip()
        if not sid:
            continue
        transport_raw = str(item.get("transport", "sse")).lower()
        if transport_raw in ("streamable_http", "http"):
            transport = MCPTransportType.STREAMABLE_HTTP
        elif transport_raw == "stdio":
            transport = MCPTransportType.STDIO
        else:
            transport = MCPTransportType.SSE

        definitions.append(
            MCPServerDefinition(
                server_id=sid,
                name=str(item.get("name") or sid),
                description=str(item.get("description") or ""),
                transport=transport,
                url=str(item.get("url") or "") or None,
                headers=dict(item.get("headers") or {}),
                auth_token=item.get("auth_token"),
                stdio_command=[str(c) for c in (item.get("stdio_command") or item.get("command") or [])] or None,
                env={str(k): str(v) for k, v in (item.get("env") or {}).items()},
                cwd=item.get("cwd"),
                allowed_tools=[str(t) for t in item.get("allowed_tools", ["*"])],
                capabilities=[str(c) for c in item.get("capabilities", [])],
                enabled=bool(item.get("enabled", True)),
                timeout_s=int(item.get("timeout_s", 30)),
                source=str(item.get("source") or ""),
            )
        )
    return definitions


def save_mcp_server_definitions(servers: List["MCPServerDefinition"], path: Optional[str] = None) -> None:
    """持久化保存 MCPServerDefinition 列表到 JSON 配置文件中。"""
    file_path = path or os.environ.get("MCP_SERVERS_FILE") or DEFAULT_SERVERS_FILE
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for s in servers:
        items.append({
            "server_id": s.server_id,
            "name": s.name,
            "description": s.description,
            "transport": s.transport.value,
            "url": s.url,
            "headers": s.headers,
            "auth_token": s.auth_token,
            "command": s.stdio_command,
            "stdio_command": s.stdio_command,
            "env": dict(s.env or {}),
            "cwd": s.cwd,
            "allowed_tools": s.allowed_tools,
            "capabilities": s.capabilities,
            "enabled": s.enabled,
            "timeout_s": s.timeout_s,
            "source": s.source or "",
        })
    content = json.dumps({"servers": items}, indent=2, ensure_ascii=False) + "\n"
    target.write_text(content, encoding="utf-8")
    logger.info("Saved %d MCP server definitions to %s", len(items), file_path)
