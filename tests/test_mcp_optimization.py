"""验证 MCP 服务优化配置及核心逻辑。"""
import json
import os
from pathlib import Path

from app.tools.mcp.config import _expand_env_vars, load_mcp_servers
from app.tools.sandbox.policy import PolicyEngine

try:
    from app.tools.mcp.health_server import TOOLS as HEALTH_TOOLS, _handle_tool_call, build_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def test_expand_env_vars():
    # 1. 默认值解析
    assert _expand_env_vars("${TEST_NON_EXISTENT_VAR:-default_val}") == "default_val"
    # 2. 环境变量覆盖默认值
    os.environ["TEST_EXISTING_VAR"] = "custom_val"
    try:
        assert _expand_env_vars("${TEST_EXISTING_VAR:-default_val}") == "custom_val"
        assert _expand_env_vars("${TEST_EXISTING_VAR}") == "custom_val"
    finally:
        del os.environ["TEST_EXISTING_VAR"]

    # 3. 递归结构展开（含 Windows 风格反斜杠路径）
    nested = {
        "cmd": ["npx", "-y", "${TEST_PATH_VAR:-C:\\Project\\volumes}"],
        "meta": {"sub": "${TEST_SUB:-hello}"},
    }
    expanded = _expand_env_vars(nested)
    assert expanded["cmd"][2] == "C:\\Project\\volumes"
    assert expanded["meta"]["sub"] == "hello"
    print("test_expand_env_vars PASSED")


def test_load_mcp_servers():
    servers = load_mcp_servers()
    server_names = [s.name for s in servers]
    assert "demo" not in server_names
    assert "filesystem" in server_names
    assert "postgres-query" in server_names
    assert "system-health" in server_names

    fs_srv = next(s for s in servers if s.name == "filesystem")
    assert fs_srv.enabled is True
    assert "fs.read" in fs_srv.capabilities
    assert "search_files" not in fs_srv.allowed_tools
    assert "read_file" in fs_srv.allowed_tools

    pg_srv = next(s for s in servers if s.name == "postgres-query")
    assert pg_srv.enabled is True
    assert "db.read" in pg_srv.capabilities
    assert pg_srv.allowed_tools == ["query"]

    health_srv = next(s for s in servers if s.name == "system-health")
    assert health_srv.enabled is False
    assert "ops.monitor" in health_srv.capabilities
    print("test_load_mcp_servers PASSED")


def test_sandbox_policy_mcp_rules():
    from app.tools.sandbox.context import SandboxContext

    policy_path = Path("config/sandbox_policy.json")
    assert policy_path.exists()

    engine = PolicyEngine.from_file(policy_path)
    ctx = SandboxContext(user_id="test_user", session_id="test_session")

    # mcp_filesystem_read_file 应在 rules 中明确放行
    verdict_fs = engine.check(
        tool_name="mcp_filesystem_read_file",
        metadata={"capabilities": ["fs.read"]},
        context=ctx,
    )
    assert verdict_fs.allowed is True
    assert verdict_fs.would_allow is True

    # mcp_postgres-query_query 应在 rules 中明确放行
    verdict_pg = engine.check(
        tool_name="mcp_postgres-query_query",
        metadata={"capabilities": ["db.read"]},
        context=ctx,
    )
    assert verdict_pg.allowed is True
    assert verdict_pg.would_allow is True

    # mcp_system-health_check_redis 应放行
    verdict_health = engine.check(
        tool_name="mcp_system-health_check_redis",
        metadata={"capabilities": ["ops.monitor"]},
        context=ctx,
    )
    assert verdict_health.allowed is True
    assert verdict_health.would_allow is True
    print("test_sandbox_policy_mcp_rules PASSED")


def test_health_server_definitions():
    if not HAS_MCP:
        print("mcp SDK not installed in host python environment, skipping health server runtime test (verified in Docker)")
        return
    server = build_server()
    assert server.name == "system-health"
    tool_names = [t.name for t in HEALTH_TOOLS]
    assert "check_postgres" in tool_names
    assert "check_redis" in tool_names
    assert "check_milvus" in tool_names
    assert "check_neo4j" in tool_names

    # 调用无外部依赖时的健康检查应返回 json 结构（即使报连接错误或缺库，也是标准 json 串）
    res_pg = _handle_tool_call("check_postgres", {})
    parsed_pg = json.loads(res_pg)
    assert "status" in parsed_pg

    res_redis = _handle_tool_call("check_redis", {})
    parsed_redis = json.loads(res_redis)
    assert "status" in parsed_redis
    print("test_health_server_definitions PASSED")


if __name__ == "__main__":
    test_expand_env_vars()
    test_load_mcp_servers()
    test_sandbox_policy_mcp_rules()
    test_health_server_definitions()
    print("ALL TESTS PASSED!")
