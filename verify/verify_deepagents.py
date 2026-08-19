"""DeepAgents 集成验证脚本。

用法（easyrag 全量环境）：
    PYTHONPATH=. python verify/verify_deepagents.py            # 结构验证（不调 LLM）
    PYTHONPATH=. python verify/verify_deepagents.py --live "帮我研究一下 LangGraph 并总结"
                                                             # 真实 LLM 端到端（需要 .env 的 LLM key）

验证内容（结构模式）：
1. SubAgent 配置加载（默认 research/coding）
2. 主 Agent 构建（create_react_agent + task 工具 + 项目工具）
3. task 工具对未知类型报错、已知类型可路由
4. 工具转换（注册表 → StructuredTool）

--live 模式额外执行：主 Agent → task → SubAgent → 结果 → 主 Agent 完整链路。
"""
from __future__ import annotations

import sys
import time


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {name}: {exc}")
        raise


def verify_structure() -> None:
    print("[deepagents] 结构验证（无 LLM 调用）")

    from app.agents.deep.agent import build_main_agent
    from app.agents.deep.subagents import get_subagents, subagents_prompt
    from app.agents.deep.task_tool import build_task_tool

    def _subagents():
        subs = get_subagents()
        assert len(subs) >= 2, f"应至少 2 个内置 SubAgent: {subs}"
        names = [s.name for s in subs]
        assert "research-agent" in names and "coding-agent" in names
        assert all(s.description and s.system_prompt for s in subs)
        prompt = subagents_prompt()
        assert all(s.name in prompt for s in subs)

    check("SubAgent 配置加载", _subagents)

    def _build():
        agent = build_main_agent()
        # create_react_agent 编译图可被 invoke（不执行）
        assert agent is not None
        nodes = set(agent.get_graph().nodes.keys())
        assert "agent" in nodes and "tools" in nodes, f"节点异常: {nodes}"

    check("主 Agent 构建（create_react_agent + task）", _build)

    def _task_tool():
        tool = build_task_tool()
        assert tool.name == "task"
        schema = tool.args
        assert "description" in schema and "subagent_type" in schema
        assert "research-agent" in tool.description  # 名册注入工具描述
        try:
            tool.invoke({"description": "x", "subagent_type": "ghost"})
            raise AssertionError("未知类型应抛错")
        except ValueError:
            pass

    check("task 工具（名册注入 + 未知类型拒绝）", _task_tool)

    def _tools():
        from app.agents.deep.tools import registry_to_langchain_tools

        tools = registry_to_langchain_tools()
        names = [t.name for t in tools]
        assert "calculator" in names and "text_tool" in names
        calc = [t for t in tools if t.name == "calculator"][0]
        assert "3" in str(calc.invoke({"expression": "1+2"}))

    check("工具转换（注册表 → StructuredTool）", _tools)

    print("\n✅ 结构验证通过（可 --live 跑真实 LLM 端到端）")


def verify_live(query: str) -> None:
    print(f"[deepagents] 真实 LLM 端到端: {query!r}")
    from app.agents.deep.agent import get_main_agent

    agent = get_main_agent()
    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [("user", query)]},
        config={"recursion_limit": 20},
    )
    elapsed = time.perf_counter() - start
    messages = result["messages"]
    answer = messages[-1].content if messages else ""
    print(f"  耗时 {elapsed:.1f}s，消息 {len(messages)} 条")
    print(f"  最终回答:\n{answer[:600]}")
    tool_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
    if tool_msgs:
        print(f"  工具调用次数: {len(tool_msgs)}")
    assert answer.strip(), "回答为空"
    print("\n✅ 端到端通过（若未触发 task 委派，换一个更复杂的查询再试）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        query = " ".join(sys.argv[2:]) or "帮我研究一下 LangGraph 的 SubAgent 机制并给出总结"
        verify_live(query)
    else:
        verify_structure()
