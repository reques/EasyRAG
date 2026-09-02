"""MCP 外部工具服务配置加载。

配置文件：项目根 `mcp_servers.json`（可被环境变量 MCP_SERVERS_FILE 覆盖）。

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SERVERS_FILE = "mcp_servers.json"


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
            )
        )
    logger.info("MCP servers loaded from %s: %s", file_path, [s.name for s in servers])
    return servers
