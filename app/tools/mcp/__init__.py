"""MCP 外部工具服务接入（Model Context Protocol）。

- config.py       : 配置加载（mcp_servers.json）
- manager.py      : 统一启停 + 工具注册 + 权限过滤（MCPManager 单例）
- demo_server.py  : 零依赖演示 server（stdio / HTTP 双模式）
"""
