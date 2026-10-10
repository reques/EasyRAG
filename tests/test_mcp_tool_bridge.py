"""MCP 工具桥接：inputSchema 解析与参数转发的回归测试。

背景（线上故障，2026-10）：
    mcp SDK 的 ``Tool`` 字段名是 ``input_schema``（别名为 ``inputSchema``），
    而 manager 只读 ``getattr(t, "inputSchema", {})`` → 永远拿到空 schema。
    后果：模型看到的所有 MCP 工具都没有参数（甚至被塞了一个 ``noop`` 占位），
    只能靠猜参数；12306 这类严格校验的服务端一律回“参数校验错误”。
"""
import asyncio
from typing import Any, Dict, List

import pytest

from app.agents.deep.tools import registry_to_langchain_tools
from app.tools.mcp.manager import MCPServerHandle, _legacy_mcp_tool_name, _mcp_tool_name
from app.tools.mcp.models import MCPServerDefinition, MCPTransportType
from app.tools.registry import get_tool_registry

# 与真实 @Joooook/12306-mcp 工具形状一致：有参数，且必填项明确。
TICKETS_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "description": "乘车日期，格式 yyyy-MM-dd"},
        "from_station": {"type": "string", "description": "出发站 telecode"},
        "to_station": {"type": "string", "description": "到达站 telecode"},
        "train_filter_flags": {"type": "string", "description": "车次筛选标志"},
    },
    "required": ["date", "from_station", "to_station"],
}


class _FakeTool:
    """模拟 mcp.types.Tool：真实 SDK 用的是字段名 input_schema。"""

    def __init__(self, name: str, description: str = "", input_schema: Dict[str, Any] | None = None,
                 legacy_alias: bool = False):
        self.name = name
        self.description = description
        if legacy_alias:
            # 老版本 SDK：只有 camelCase 别名
            self.inputSchema = input_schema or {}
        else:
            self.input_schema = input_schema or {}


class _FakeListToolsResult:
    def __init__(self, tools: List[_FakeTool]):
        self.tools = tools


class _FakeCallResult:
    def __init__(self, text: str = "ok"):
        self.isError = False
        self.content = [type("C", (), {"type": "text", "text": text})()]
        self.structuredContent = None


class _FakeSession:
    def __init__(self, tools: List[_FakeTool]):
        self._tools = tools
        self.calls: List[tuple] = []

    async def list_tools(self):
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return _FakeCallResult(f"called {name}")


def _handle(tools: List[_FakeTool], server_id: str = "ms12306") -> MCPServerHandle:
    definition = MCPServerDefinition(
        server_id=server_id,
        name=server_id,
        transport=MCPTransportType.STDIO,
        stdio_command=["python", "-c", "pass"],
    )
    handle = MCPServerHandle(definition)
    handle.session = _FakeSession(tools)
    return handle


@pytest.fixture(autouse=True)
def _treat_handles_as_connected(monkeypatch):
    """``running`` 决定 check_fn 与 registry 可见性；单元测试里直接置为已连接。"""
    monkeypatch.setattr(MCPServerHandle, "running", property(lambda self: True))


def _capture_calls(monkeypatch, handle: MCPServerHandle) -> List[Dict[str, Any]]:
    """捕获桥接层真正转发给 MCP 服务端的参数（绕过常驻 loop 的存活检查）。"""
    captured: List[Dict[str, Any]] = []

    def fake_call(raw_tool_name: str, arguments: Dict[str, Any]) -> str:
        captured.append({"tool": raw_tool_name, "arguments": arguments})
        return "ok"

    monkeypatch.setattr(handle, "call_tool_sync", fake_call)
    return captured


def _cleanup(*names: str) -> None:
    for name in names:
        get_tool_registry().unregister(name)


def test_input_schema_is_read_from_sdk_field_name():
    """核心回归：字段名 input_schema 必须被解析成 arg_schema。"""
    handle = _handle([_FakeTool("get-tickets", "查询 12306 余票", TICKETS_SCHEMA)])
    asyncio.run(handle._refresh_and_register_tools())

    try:
        tool = get_tool_registry().get(_mcp_tool_name("ms12306", "get-tickets"))
        assert set(tool.arg_schema) == {"date", "from_station", "to_station", "train_filter_flags"}
        assert tool.arg_schema["date"] == ("string", "乘车日期，格式 yyyy-MM-dd", True)
        assert tool.arg_schema["train_filter_flags"][2] is False

        # 暴露给模型的 schema 必须带上参数，而不是空 properties
        schema = tool.to_llm_schema()["function"]["parameters"]
        assert set(schema["properties"]) == set(tool.arg_schema)
        assert schema["required"] == ["date", "from_station", "to_station"]
    finally:
        _cleanup(_mcp_tool_name("ms12306", "get-tickets"), _legacy_mcp_tool_name("ms12306", "get-tickets"))


def test_legacy_camelcase_alias_still_supported():
    """老版本 SDK 只暴露 inputSchema 别名时也要能解析。"""
    handle = _handle([_FakeTool("get-station-code-of-citys", "查城市站码",
                                {"type": "object", "properties": {"citys": {"type": "string", "description": "城市"}},
                                 "required": ["citys"]}, legacy_alias=True)])
    asyncio.run(handle._refresh_and_register_tools())
    try:
        tool = get_tool_registry().get(_mcp_tool_name("ms12306", "get-station-code-of-citys"))
        assert tool.arg_schema == {"citys": ("string", "城市", True)}
    finally:
        _cleanup(_mcp_tool_name("ms12306", "get-station-code-of-citys"),
                 _legacy_mcp_tool_name("ms12306", "get-station-code-of-citys"))


def test_only_declared_arguments_reach_the_mcp_server(monkeypatch):
    """展示/控制参数（_action_summary、noop、模型臆造的键）不得进入 MCP 请求体。"""
    handle = _handle([_FakeTool("get-tickets", "查询余票", TICKETS_SCHEMA)])
    asyncio.run(handle._refresh_and_register_tools())
    captured = _capture_calls(monkeypatch, handle)
    try:
        tool = get_tool_registry().get(_mcp_tool_name("ms12306", "get-tickets"))
        tool.fn(
            date="2026-10-09",
            from_station="BJP",
            to_station="SHH",
            _action_summary="查一下余票",
            noop=None,
            invented="nope",
        )
        assert captured == [{
            "tool": "get-tickets",
            "arguments": {"date": "2026-10-09", "from_station": "BJP", "to_station": "SHH"},
        }]
    finally:
        _cleanup(_mcp_tool_name("ms12306", "get-tickets"), _legacy_mcp_tool_name("ms12306", "get-tickets"))


def test_parameterless_tool_forwards_nothing_and_advertises_no_fake_field(monkeypatch):
    """无参数工具：既不向模型暴露 noop 占位，也不把杂项参数发给服务端。"""
    handle = _handle([_FakeTool("get-current-date", "获取当前日期", {})])
    asyncio.run(handle._refresh_and_register_tools())
    captured = _capture_calls(monkeypatch, handle)
    try:
        name = _mcp_tool_name("ms12306", "get-current-date")
        tool = get_tool_registry().get(name)
        assert tool.arg_schema == {}

        langchain_tool = next(t for t in registry_to_langchain_tools([name]) if t.name == name)
        assert "noop" not in langchain_tool.args_schema.model_fields
        assert langchain_tool.args_schema.model_fields == {}

        tool.fn(noop=None, _action_summary="x")
        assert captured == [{"tool": "get-current-date", "arguments": {}}]
    finally:
        _cleanup(_mcp_tool_name("ms12306", "get-current-date"),
                 _legacy_mcp_tool_name("ms12306", "get-current-date"))


def test_array_and_object_arguments_stay_structured():
    """array/object 参数不能退化成 str，否则模型只能拼字符串发给服务端。"""
    handle = _handle([_FakeTool("bulk", "批量查询", {
        "type": "object",
        "properties": {
            "codes": {"type": "array", "description": "站码列表"},
            "filters": {"type": "object", "description": "筛选条件"},
        },
        "required": ["codes"],
    })])
    asyncio.run(handle._refresh_and_register_tools())
    try:
        name = _mcp_tool_name("ms12306", "bulk")
        langchain_tool = next(t for t in registry_to_langchain_tools([name]) if t.name == name)
        fields = langchain_tool.args_schema.model_fields
        assert fields["codes"].annotation == list
        assert "dict" in str(fields["filters"].annotation)
    finally:
        _cleanup(_mcp_tool_name("ms12306", "bulk"), _legacy_mcp_tool_name("ms12306", "bulk"))


def test_underscore_prefixed_server_properties_are_skipped():
    """服务端自带下划线参数（Pydantic 不允许作为字段名）应被跳过而不是崩溃。"""
    handle = _handle([_FakeTool("weird", "怪工具", {
        "type": "object",
        "properties": {"_internal": {"type": "string"}, "ok": {"type": "string", "description": "正常"}},
        "required": [],
    })])
    asyncio.run(handle._refresh_and_register_tools())
    try:
        tool = get_tool_registry().get(_mcp_tool_name("ms12306", "weird"))
        assert set(tool.arg_schema) == {"ok"}
    finally:
        _cleanup(_mcp_tool_name("ms12306", "weird"), _legacy_mcp_tool_name("ms12306", "weird"))
