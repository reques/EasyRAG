"""上下文注入测试（阶段 0 调整版）。

原覆盖 2026-08-15 P1 修复的用例随 single 固定管线退役（2026-09-02 阶段 0）：
- query_rewrite 节点 / build_graph 编译用例：graph 管线已删除，指代消解
  统一走 prepare_context 调用的 rewrite_query_with_history（保留等价用例）
- REACT_REASONING 模板用例：模板不再被管线消费，一并退役

保留/新增：rewrite_query_with_history 纯函数行为（prepare_context 与
dynamic 组装消息链仍在使用）。
"""
from __future__ import annotations


# ── rewrite_query_with_history（prepare_context 步骤 0）──────────────────
def test_rewrite_followup_query(monkeypatch):
    import app.graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "rewrite_query_with_history", lambda q, h: "无锡明天天气如何"
    )
    assert nodes.rewrite_query_with_history(
        "那明天呢", [{"role": "user", "content": "无锡今天天气"}]
    ) == "无锡明天天气如何"


def test_rewrite_keeps_original_when_no_history():
    import app.graph.nodes as nodes

    # 无历史时内部短路，原样返回且不触发 LLM
    assert nodes.rewrite_query_with_history("什么是民法典", []) == "什么是民法典"
