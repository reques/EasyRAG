"""DeepAgents 集成测试：配置 / 工具转换 / task 委派 / 上下文隔离 / 错误传播。

全部用 mock 模型（不调用真实 LLM）。运行：easyrag env 下 pytest。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agents.deep.subagents import (
    DEFAULT_SUBAGENTS,
    SubAgentConfig,
    get_subagent_config,
    get_subagents,
    subagents_prompt,
)
from app.agents.deep.task_tool import build_task_tool
from app.agents.deep.tools import _to_structured


class MockToolCallingModel(BaseChatModel):
    """按预设脚本返回 tool_calls 或纯文本的测试模型。

    responses: 每个元素要么是 str（直接返回该文本），要么是
    {"tool_calls": [{"name", "args"}], "content": "..."}。
    超出脚本长度后重复最后一个响应。
    """

    responses: List[Any]
    calls: int = 0
    last_messages: Optional[List[Any]] = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        self.last_messages = list(messages)
        idx = min(self.calls - 1, len(self.responses) - 1)
        spec = self.responses[idx]
        if isinstance(spec, str):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=spec))])
        tool_calls = [
            {
                "name": tc["name"],
                "args": tc.get("args", {}),
                "id": f"call_{self.calls}_{i}",
            }
            for i, tc in enumerate(spec.get("tool_calls", []))
        ]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=spec.get("content", ""), tool_calls=tool_calls
                    )
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "mock-tool-calling"


# ── 1. SubAgent 配置 ──────────────────────────────────────────────────────


def test_default_subagents_loaded():
    subagents = get_subagents()
    assert len(subagents) >= 2
    names = {s.name for s in subagents}
    assert "research-agent" in names and "coding-agent" in names


def test_get_subagent_config_match_and_miss():
    cfg = get_subagent_config("research-agent")
    assert cfg is not None
    assert cfg.description and cfg.system_prompt and cfg.tools
    assert get_subagent_config("no-such-agent") is None


def test_subagents_prompt_lists_all():
    prompt = subagents_prompt()
    for s in DEFAULT_SUBAGENTS:
        assert s.name in prompt


# ── 2. 工具转换 ───────────────────────────────────────────────────────────


def test_tool_definition_to_structured():
    """注册表工具 → StructuredTool 转换 + 执行（走 registry.invoke）。"""
    from app.tools.registry import get_tool_registry

    registry = get_tool_registry()
    calc = registry.get("calculator")  # 纯函数工具，无外部依赖
    st = _to_structured(calc)
    assert st.name == "calculator"
    assert st.description == calc.description
    result = st.invoke({"expression": "1+2"})
    assert "3" in str(result)


def test_task_tool_unknown_subagent_raises():
    tool = build_task_tool(model=MockToolCallingModel(responses=["x"]))
    with pytest.raises(ValueError, match="未知子智能体类型"):
        tool.invoke({"description": "do something", "subagent_type": "ghost-agent"})


# ── 3. task 委派端到端（主 Agent → task → SubAgent → 结果 → 主 Agent）────


def _main_and_sub_models():
    """main: 第一轮调 task，第二轮给最终答案；sub: 返回研究结果。"""
    sub = MockToolCallingModel(responses=["【研究结果】经过检索，结论是 A。", "补充说明"])
    main = MockToolCallingModel(responses=[
        {
            "tool_calls": [
                {
                    "name": "task",
                    "args": {
                        "description": "研究一下 XXX 并总结",
                        "subagent_type": "research-agent",
                    },
                }
            ],
            "content": "",
        },
        "综合回答：基于子智能体结果，最终结论是 A。",
    ])
    return main, sub


def test_main_agent_delegates_and_continues():
    from app.agents.deep.agent import build_main_agent

    main, sub = _main_and_sub_models()
    agent = build_main_agent(model=main, subagent_model=sub)
    result = agent.invoke(
        {"messages": [("user", "帮我研究一下 XXX 并给出总结")]},
        config={"recursion_limit": 20},
    )
    messages = result["messages"]
    answer = messages[-1].content
    # 主 Agent 基于子智能体结果继续推理 → 最终回答
    assert "最终结论是 A" in answer
    # 子 Agent 确实被调用（task 委派发生）
    assert sub.calls >= 1


def test_subagent_context_isolated_from_main():
    """子 Agent 只看到 task description，看不到主 Agent 的完整历史。"""
    from app.agents.deep.agent import build_main_agent

    main, sub = _main_and_sub_models()
    agent = build_main_agent(model=main, subagent_model=sub)
    agent.invoke(
        {"messages": [("user", "帮我研究一下 XXX 并给出总结")]},
        config={"recursion_limit": 20},
    )
    # 子 Agent 收到的消息 = 系统 prompt + task description（无主 Agent 历史）
    assert sub.last_messages is not None
    texts = " ".join(str(getattr(m, "content", "")) for m in sub.last_messages)
    assert "研究一下 XXX 并总结" in texts  # task description 传入
    # 上下文隔离：主 Agent 的原始用户消息不泄漏给子 Agent
    assert "帮我研究一下" not in texts


def test_subagent_result_does_not_pollute_main_state():
    """子 Agent 结果只作为 ToolMessage 进入主 Agent 消息流，主 Agent state 干净。"""
    from app.agents.deep.agent import build_main_agent

    main, sub = _main_and_sub_models()
    agent = build_main_agent(model=main, subagent_model=sub)
    result = agent.invoke(
        {"messages": [("user", "帮我研究一下 XXX 并给出总结")]},
        config={"recursion_limit": 20},
    )
    messages = result["messages"]
    # 消息流包含：user + ai(task) + tool(子结果) + ai(最终) —— 无子 Agent 中间思考
    ai_msgs = [m for m in messages if getattr(m, "type", "") == "ai"]
    tool_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
    assert len(tool_msgs) == 1  # 只有一次 task 委派结果
    assert "研究结果" in tool_msgs[0].content
    assert any("最终结论是 A" in m.content for m in ai_msgs)


def test_subagent_error_propagates_to_main():
    """子 Agent 抛错 → task 工具报错 → 主 Agent 收到错误后仍可继续推理。"""
    from app.agents.deep.agent import build_main_agent

    class ExplodingModel(MockToolCallingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            self.last_messages = list(messages)
            raise RuntimeError("模拟子 Agent 崩溃")

    sub = ExplodingModel(responses=["x"])
    main = MockToolCallingModel(responses=[
        {
            "tool_calls": [
                {
                    "name": "task",
                    "args": {
                        "description": "研究一下 XXX",
                        "subagent_type": "research-agent",
                    },
                }
            ],
            "content": "",
        },
        "子智能体执行失败，我基于已有知识给出回答。",
    ])
    agent = build_main_agent(model=main, subagent_model=sub)
    result = agent.invoke(
        {"messages": [("user", "帮我研究一下 XXX")]},
        config={"recursion_limit": 20},
    )
    messages = result["messages"]
    answer = messages[-1].content
    # 错误传播：ToolMessage 含错误信息；主 Agent 继续生成最终回答
    tool_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
    assert tool_msgs, "应存在工具错误消息"
    assert "模拟子 Agent 崩溃" in str(tool_msgs[0].content)
    assert "基于已有知识" in answer


# ── 4. 独立 SubAgent 直接运行 ─────────────────────────────────────────────


def test_run_subagent_directly():
    from app.agents.deep.subagents import run_subagent

    sub = MockToolCallingModel(responses=["直接运行的结果"])
    cfg = SubAgentConfig(
        name="unit-test-agent",
        description="测试",
        system_prompt="你是测试子智能体",
        tools=(),
    )
    result = run_subagent(cfg, "执行任务", model=sub)
    assert result == "直接运行的结果"
