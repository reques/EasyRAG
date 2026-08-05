"""hermes-verify: 工具插件化验证（非测试套件）。

针对 Task 1 工具插件化:
1. discover_tools 自动发现并注册 4 个工具（不再硬编码）
2. check_fn 自检：web_search 无 TAVILY_API_KEY 时 is_available=False
3. is_available 过滤：list_names/to_llm_schema/to_react_prompt 只含可用工具
4. invoke 对不可用工具抛 ToolExecutionError
5. 现有 4 工具功能回归（calculator/datetime/text_tool 实际调用）
6. 新增工具无需改 registry（放一个测试模块进去能被扫到）

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_tool_plugin.py
"""
import os
import sys

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
sys.path.insert(0, ROOT)

from app.core.exceptions import ToolExecutionError
from app.tools.registry import (
    ToolDefinition, ToolRegistry, discover_tools, get_tool_registry,
)

print("== 1. discover_tools 自动发现 ==")
reg = get_tool_registry()
names = reg.list_names(available_only=False)
check("discovers all 4 tools", set(names) == {"calculator", "datetime_tool", "text_tool", "web_search"},
      f"got {sorted(names)}")

print("== 2. check_fn 自检 ==")
from app.core.config import get_settings
cfg = get_settings()
ws = reg.get("web_search")
if cfg.TAVILY_API_KEY:
    check("web_search available (key set)", ws.is_available(), "TAVILY_API_KEY configured")
else:
    check("web_search unavailable (no key)", not ws.is_available(), "TAVILY_API_KEY empty")
check("calculator always available", reg.get("calculator").is_available())
check("datetime_tool always available", reg.get("datetime_tool").is_available())
check("text_tool always available", reg.get("text_tool").is_available())

print("== 3. is_available 过滤 ==")
avail = reg.list_names()
check("list_names filters by availability",
      all(reg.get(n).is_available() for n in avail), f"available={sorted(avail)}")
schema = reg.to_llm_schema()
check("to_llm_schema only available tools",
      all(reg.get(s["function"]["name"]).is_available() for s in schema),
      f"{len(schema)} tools in schema")
rp = reg.to_react_prompt()
check("to_react_prompt non-empty", bool(rp) and rp != "（无可用工具）")
check("to_react_prompt contains calculator", "calculator" in rp)
if not cfg.TAVILY_API_KEY:
    check("to_react_prompt excludes unavailable web_search", "web_search" not in rp)

print("== 4. invoke 自检拦截 ==")
# 造一个 check_fn=False 的工具验证 invoke 拦截
test_reg = ToolRegistry()
test_reg.register(ToolDefinition(
    name="broken_tool", description="t", fn=lambda: "x",
    check_fn=lambda: False,
))
try:
    test_reg.invoke("broken_tool")
    check("invoke blocks unavailable tool", False, "no exception raised")
except ToolExecutionError as e:
    check("invoke blocks unavailable tool", "not available" in str(e), str(e)[:60])

print("== 5. 现有工具功能回归 ==")
try:
    r = reg.invoke("calculator", expression="1+1")
    check("calculator 1+1=2", "= 2" in r or "=2" in r, r)
except Exception as e:
    check("calculator 1+1=2", False, str(e))
try:
    r = reg.invoke("datetime_tool", fmt="%Y")
    check("datetime_tool returns year", len(r) > 0 and "20" in r, r[:40])
except Exception as e:
    check("datetime_tool returns year", False, str(e))
try:
    r = reg.invoke("text_tool", operation="word_count", text="hello world foo")
    check("text_tool word_count", "3" in r, r[:40])
except Exception as e:
    check("text_tool word_count", False, str(e))

print("== 6. 新增工具免改 registry ==")
# 放一个临时工具模块到 app/tools/，验证 discover_tools 能扫到
tmp_tool = os.path.join(ROOT, "app", "tools", "_tmp_verify_tool.py")
with open(tmp_tool, "w", encoding="utf-8") as f:
    f.write(
        "from app.tools.registry import ToolDefinition\n"
        "def _ping() -> str:\n    return 'pong'\n"
        "TOOL = ToolDefinition(name='_tmp_ping', description='verify', fn=_ping, check_fn=lambda: True)\n"
    )
try:
    fresh = discover_tools()
    check("new module auto-discovered without editing registry",
          "_tmp_ping" in fresh.list_names(available_only=False))
    check("new tool invokable", fresh.invoke("_tmp_ping") == "pong")
finally:
    os.remove(tmp_tool)
    # 清 pycache 避免下次扫描残留
    cache = os.path.join(ROOT, "app", "tools", "__pycache__", "_tmp_verify_tool.cpython-311.pyc")
    if os.path.exists(cache):
        os.remove(cache)

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed (ad-hoc) ==")
sys.exit(0 if passed == len(results) else 1)
