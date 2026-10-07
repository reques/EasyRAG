import asyncio
import os
from pydantic import ValidationError

from app.tools.mcp.models import (
    MCPServerCreate,
    MCPServerDefinition,
    MCPServerStatus,
    MCPServerUpdate,
    MCPToolInfo,
    MCPTransportType,
)
from app.tools.mcp.manager import (
    MCPManager,
    _mcp_tool_name,
    _legacy_mcp_tool_name,
    get_mcp_manager,
)
from app.tools.registry import ToolDefinition, get_tool_registry
from app.agents.deep.tools import registry_to_langchain_tools


def test_mcp_models_validation():
    # 1. SSE 模式下缺少 URL 抛异常
    try:
        MCPServerCreate(
            server_id="test_sse",
            name="Test SSE",
            transport=MCPTransportType.SSE,
            url=None,
        )
        assert False, "Should have raised ValidationError for missing URL in SSE"
    except ValidationError:
        pass

    # 2. stdio 模式执行非白名单命令被拦截 (RCE 防御)
    try:
        MCPServerCreate(
            server_id="evil_service",
            name="Evil Service",
            transport=MCPTransportType.STDIO,
            stdio_command=["bash", "-c", "rm -rf /"],
        )
        assert False, "Should have raised ValidationError for untrusted bash command"
    except ValidationError:
        pass

    try:
        MCPServerCreate(
            server_id="evil_powershell",
            name="Evil PS",
            transport=MCPTransportType.STDIO,
            stdio_command=["powershell.exe", "-Command", "whoami"],
        )
        assert False, "Should have raised ValidationError for untrusted powershell command"
    except ValidationError:
        pass

    # 3. 受信任命令正常放行
    valid_stdio = MCPServerCreate(
        server_id="safe_python",
        name="Safe Python",
        transport=MCPTransportType.STDIO,
        stdio_command=["python", "-m", "app.tools.mcp.health_server"],
    )
    assert valid_stdio.server_id == "safe_python"

    # 4. 远程 SSE 正常放行
    valid_sse = MCPServerCreate(
        server_id="remote_sse",
        name="Remote SSE",
        transport=MCPTransportType.SSE,
        url="https://api.example.com/sse",
        headers={"X-Custom": "val"},
        auth_token="token_123",
    )
    assert valid_sse.url == "https://api.example.com/sse"
    print("test_mcp_models_validation PASSED")


def test_tool_namespacing():
    assert _mcp_tool_name("github", "create_issue") == "github__create_issue"
    assert _legacy_mcp_tool_name("github", "create_issue") == "mcp_github_create_issue"
    print("test_tool_namespacing PASSED")


def test_mcp_manager_crud():
    mgr = MCPManager()

    # 1. 注册新远程 SSE 服务（enabled=False 避免实际外联）
    new_srv = MCPServerCreate(
        server_id="test_remote_crm",
        name="Test CRM",
        transport=MCPTransportType.SSE,
        url="https://crm.internal.corp/sse",
        enabled=False,
    )
    defn = mgr.register(new_srv, persist=False)
    assert defn.server_id == "test_remote_crm"
    assert "test_remote_crm" in mgr.servers

    # 2. 查询
    fetched = mgr.get_definition("test_remote_crm")
    assert fetched.name == "Test CRM"

    # 3. 更新
    updated = mgr.update(
        "test_remote_crm",
        MCPServerUpdate(name="Test CRM Enterprise", timeout_s=45),
        persist=False,
    )
    assert updated.name == "Test CRM Enterprise"
    assert updated.timeout_s == 45

    # 4. 卸载
    mgr.unregister("test_remote_crm", persist=False)
    assert "test_remote_crm" not in mgr.servers
    print("test_mcp_manager_crud PASSED")


def test_tool_routing_filter():
    reg = get_tool_registry()

    # 模拟两个不同 server_id 的 MCP 工具
    reg.register(
        ToolDefinition(
            name="db__query_user",
            description="Query user",
            fn=lambda **kw: "ok",
            arg_schema={},
            metadata={"server_id": "db", "capabilities": ["db.read"]},
        )
    )
    reg.register(
        ToolDefinition(
            name="github__list_repos",
            description="List repos",
            fn=lambda **kw: "ok",
            arg_schema={},
            metadata={"server_id": "github", "capabilities": ["net.out"]},
        )
    )

    # 1. 不加限制，获取全量
    all_tools = registry_to_langchain_tools()
    tool_names = [t.name for t in all_tools]
    assert "db__query_user" in tool_names
    assert "github__list_repos" in tool_names

    # 2. Tool Routing: 仅允许 db 服务工具
    db_only = registry_to_langchain_tools(mcp_server_ids=["db"])
    db_names = [t.name for t in db_only]
    assert "db__query_user" in db_names
    assert "github__list_repos" not in db_names

    # 3. Tool Routing: 仅允许 github 服务工具
    gh_only = registry_to_langchain_tools(mcp_server_ids=["github"])
    gh_names = [t.name for t in gh_only]
    assert "github__list_repos" in gh_names
    assert "db__query_user" not in gh_names

    print("test_tool_routing_filter PASSED")


if __name__ == "__main__":
    test_mcp_models_validation()
    test_tool_namespacing()
    test_mcp_manager_crud()
    test_tool_routing_filter()
    print("ALL MCP V2 ARCHITECTURE TESTS PASSED!")
