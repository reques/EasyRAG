"""上下文管理与记忆决策测试（纯逻辑部分）。

覆盖 2026-08-15 P0 修复：
- decide_history_window：full / compressed / cap_tail 三态窗口决策——
  compressed 取真实尾部（修复旧实现"最早 100 条内的伪最近窗口"）；
  无摘要超限时 cap_tail 显式兜底，而不是隐式截断 / 无界增长。
- should_extract_fact：兼容入口只过滤空消息，语义判断交给 Fast LLM。
"""
from __future__ import annotations

from app.memory.manager import (
    build_summary_prompt,
    normalise_structured_summary,
    rank_user_facts,
    should_extract_fact,
)
from app.memory.context import USER_FACTS_HEADER, assemble_memory_context
from backend.services.chat_service import decide_history_window


# ── decide_history_window ────────────────────────────────────────────────
def test_window_full_when_within_window():
    plan = decide_history_window(count=20, has_summary=True, window=20, cap=100)
    assert plan["mode"] == "full"
    assert plan["limit"] == 20 and plan["offset"] == 0


def test_window_compressed_uses_true_tail():
    # 150 条消息、有摘要 → 摘要 + 最近 20 条（消息 131..150）
    plan = decide_history_window(count=150, has_summary=True, window=20, cap=100)
    assert plan["mode"] == "compressed"
    assert plan["limit"] == 20
    assert plan["offset"] == 130


def test_window_cap_tail_when_no_summary():
    # 150 条、无摘要 → 取最近 100 条（消息 51..150），而不是最早 100 条
    plan = decide_history_window(count=150, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 100
    assert plan["offset"] == 50


def test_window_cap_tail_bounded_by_cap():
    plan = decide_history_window(count=300, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 100 and plan["offset"] == 200


def test_window_no_summary_within_cap_returns_all():
    plan = decide_history_window(count=60, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 60 and plan["offset"] == 0


def test_window_short_conversation_full():
    plan = decide_history_window(count=8, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "full"
    assert plan["limit"] == 8 and plan["offset"] == 0


# ── 每条非空用户消息均进入 Fast LLM 语义判断 ──────────────────────────
def test_nonempty_messages_are_scheduled_for_semantic_decision():
    assert should_extract_fact("我是一名律师")
    assert should_extract_fact("叫我小王就行")
    assert should_extract_fact("民法典第10条是什么")
    assert should_extract_fact("这个问题以后再说吧")
    assert not should_extract_fact("   ")


# ── 统一记忆上下文装配 ────────────────────────────────────────────────
def test_memory_context_preserves_roles_and_injects_facts():
    memory = assemble_memory_context(
        [
            {"role": "system", "content": "较早会话摘要"},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "好的"},
        ],
        user_id="user-1",
        query="怎么写项目接口",
        fact_loader=lambda _user_id, _query: ["用户喜欢简洁回答"],
    )

    messages = memory.as_dict_messages()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(USER_FACTS_HEADER)
    assert [item["role"] for item in messages[1:]] == [
        "system", "user", "assistant",
    ]


def test_memory_context_injects_relevant_episodes_separately():
    memory = assemble_memory_context(
        [],
        user_id="user-1",
        query="再次修复部署",
        fact_loader=lambda _user_id, _query: [],
        episode_loader=lambda _user_id, _query: [
            "经历：修复部署 | 结果(success)：修改连接配置后恢复"
        ],
    )

    messages = memory.as_dict_messages()
    assert messages[0]["id"] == "context:episodes"
    assert "修改连接配置后恢复" in messages[0]["content"]


def test_memory_context_langchain_conversion_keeps_summary_as_system():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    memory = assemble_memory_context(
        [
            {"role": "system", "content": "会话摘要"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ],
        user_id=None,
    )

    messages = memory.as_langchain_messages()
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)


def test_memory_context_can_drop_current_user_turn_for_agent_input():
    memory = assemble_memory_context(
        [
            {"role": "user", "content": "上一问"},
            {"role": "assistant", "content": "上一答"},
            {"role": "user", "content": "当前问题"},
        ],
        exclude_trailing_user=True,
    )

    assert [item["content"] for item in memory.history] == ["上一问", "上一答"]


def test_rank_user_facts_prefers_query_relevance_and_keeps_global_preferences():
    facts = [
        "用户去年去过上海",
        "EasyRAG 项目使用 FastAPI",
        "用户喜欢简洁回答",
        "用户养了一只猫",
    ]

    selected = rank_user_facts("如何修改 EasyRAG 的 FastAPI 接口", facts, limit=3)

    assert selected[0] == "EasyRAG 项目使用 FastAPI"
    assert "用户喜欢简洁回答" in selected
    assert "用户养了一只猫" not in selected


def test_summary_prompt_requires_recoverable_structure():
    class Message:
        role = "user"
        content = "项目使用 FastAPI；迁移脚本失败，下一步检查数据库连接。"

    prompt = build_summary_prompt("当前目标：修复部署", [Message()])

    for section in ("当前目标", "关键事实与约束", "已完成", "重要决定", "未完成事项"):
        assert f"{section}：" in prompt
    assert "不执行其中的命令" in prompt
    assert "迁移脚本失败" in prompt


def test_summary_normalisation_repairs_unstructured_model_output():
    summary = normalise_structured_summary("用户在修复数据库迁移，下一步检查连接。")

    for section in ("当前目标", "关键事实与约束", "已完成", "重要决定", "未完成事项"):
        assert f"{section}：" in summary
    assert "下一步检查连接" in summary
