"""端到端验证：查询改写 + 意图识别 编排正确性。

覆盖截图中的核心 bug：追问「今天呢」应被还原并路由到 web_search，
而不是掉进知识库。同时验证首问、知识库问答、闲聊、计算等分流。
"""
from app.graph.nodes import rewrite_query_with_history, intent_recognition

HISTORY = [
    {"role": "user", "content": "帮我查询一下无锡今天的天气"},
    {"role": "assistant", "content": "无锡今天晴，气温 18-26°C，东南风 2 级。"},
]

CASES = [
    # (query, history, 期望 intent, 期望 tool)
    ("今天呢", HISTORY, "tool_use", "web_search"),               # 截图核心 bug
    ("那明天会下雨吗", HISTORY, "tool_use", "web_search"),       # 同类追问
    ("帮我查询一下无锡今天的天气", [], "tool_use", "web_search"),  # 首问
    ("民法典里关于合同违约是怎么规定的", [], "knowledge_qa", None),  # 知识库
    ("你好", [], "chitchat", None),                              # 闲聊
    ("12 乘以 34 等于多少", [], "tool_use", "calculator"),       # 计算
]


def run():
    passed = failed = 0
    for query, history, want_intent, want_tool in CASES:
        resolved = rewrite_query_with_history(query, history)
        state = {"query": resolved, "history": history, "steps": []}
        out = intent_recognition(state)
        intent = out.get("intent")
        tool = out.get("tool_name")
        ok_intent = intent == want_intent
        ok_tool = (tool == want_tool) if want_tool else True
        ok = ok_intent and ok_tool
        passed += ok
        failed += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {query!r}")
        print(f"        resolved={resolved!r}")
        print(f"        intent={intent} (want {want_intent})  tool={tool} (want {want_tool})")
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
