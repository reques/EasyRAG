"""MCP 配置持久化：``${VAR}`` 占位符必须活过一次安装/编辑。

背景（线上问题，2026-10）：``save_mcp_server_definitions`` 直接落盘内存里
**已展开**的值，于是从 MCP 广场安装一个服务后，整份配置被固化成具体值：
``${MCP_FS_ROOT:-./volumes}`` 变成 ``/app/volumes/workspace``、
``${POSTGRES_PASSWORD:-...}`` 变成口令明文，跨环境配置彻底失效。
"""
import json
from pathlib import Path

from app.tools.mcp.config import (
    load_mcp_server_definitions,
    save_mcp_server_definitions,
)

RAW = {
    "servers": [
        {
            "name": "filesystem",
            "transport": "stdio",
            "enabled": True,
            "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "${MCP_FS_ROOT:-./volumes}"],
            "capabilities": ["fs.read"],
            "allowed_tools": ["read_file"],
        },
        {
            "name": "postgres-query",
            "transport": "stdio",
            "enabled": True,
            "command": [
                "npx", "-y", "@modelcontextprotocol/server-postgres",
                "postgresql://${PG_USER:-easyrag}:${PG_PASSWORD:-easyrag_secret}@${PG_HOST:-localhost}:5432/easyrag",
            ],
            "allowed_tools": ["query"],
        },
    ]
}


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_placeholders_survive_an_unrelated_persist(tmp_path, monkeypatch):
    """加载后原样保存（比如新增了一个广场服务）不得把占位符展开写死。"""
    monkeypatch.setenv("MCP_FS_ROOT", "/app/volumes/workspace")
    monkeypatch.setenv("PG_PASSWORD", "real-secret")
    path = _write(tmp_path, RAW)

    definitions = load_mcp_server_definitions(str(path))
    assert definitions[0].stdio_command[-1] == "/app/volumes/workspace"  # 内存中是展开值
    save_mcp_server_definitions(definitions, str(path))

    saved = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["server_id"]: item for item in saved["servers"]}
    assert by_id["filesystem"]["command"][-1] == "${MCP_FS_ROOT:-./volumes}"
    assert by_id["filesystem"]["stdio_command"][-1] == "${MCP_FS_ROOT:-./volumes}"
    assert "${PG_PASSWORD:-easyrag_secret}" in by_id["postgres-query"]["command"][-1]
    assert "real-secret" not in path.read_text(encoding="utf-8")


def test_changed_values_are_written_and_new_entries_have_no_raw(tmp_path, monkeypatch):
    """真正改过的字段要落盘；广场新装的服务没有原始条目，直接写具体值。"""
    monkeypatch.delenv("PG_PASSWORD", raising=False)
    path = _write(tmp_path, RAW)
    definitions = load_mcp_server_definitions(str(path))

    definitions[0].stdio_command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data/kb"]
    from app.tools.mcp.models import MCPServerDefinition, MCPTransportType

    definitions.append(MCPServerDefinition(
        server_id="joooook-12306-mcp",
        name="12306",
        transport=MCPTransportType.STDIO,
        stdio_command=["npx", "-y", "12306-mcp"],
        env={"KEY": "value"},
        source="modelscope:@Joooook/12306-mcp",
    ))
    save_mcp_server_definitions(definitions, str(path))

    saved = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["server_id"]: item for item in saved["servers"]}
    assert by_id["filesystem"]["command"][-1] == "/data/kb"
    assert by_id["joooook-12306-mcp"]["env"] == {"KEY": "value"}
    assert by_id["joooook-12306-mcp"]["source"] == "modelscope:@Joooook/12306-mcp"
    # 未改动的 postgres-query 仍保留占位符
    assert "${PG_HOST:-localhost}" in by_id["postgres-query"]["command"][-1]


def test_env_values_keep_placeholders(tmp_path, monkeypatch):
    """env 里的 ${VAR} 同样不能被展开写死（API Key 走 .env 引用是最佳实践）。"""
    monkeypatch.setenv("AMAP_KEY", "abc123")
    payload = {
        "servers": [{
            "name": "amap", "transport": "stdio", "enabled": True,
            "command": ["npx", "-y", "@amap/amap-maps-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": "${AMAP_KEY:-}"},
        }]
    }
    path = _write(tmp_path, payload)
    definitions = load_mcp_server_definitions(str(path))
    assert definitions[0].env["AMAP_MAPS_API_KEY"] == "abc123"

    save_mcp_server_definitions(definitions, str(path))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["servers"][0]["env"]["AMAP_MAPS_API_KEY"] == "${AMAP_KEY:-}"
    assert "abc123" not in path.read_text(encoding="utf-8")


def test_reload_after_save_keeps_expansion_working(tmp_path, monkeypatch):
    """保存后再加载，展开结果与首次一致（幂等）。"""
    monkeypatch.setenv("MCP_FS_ROOT", "/app/volumes/workspace")
    path = _write(tmp_path, RAW)

    first = load_mcp_server_definitions(str(path))
    save_mcp_server_definitions(first, str(path))
    second = load_mcp_server_definitions(str(path))
    assert [d.stdio_command for d in first] == [d.stdio_command for d in second]
