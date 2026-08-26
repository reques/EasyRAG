"""多智能体路由规则测试（_should_use_multi）。

2026-08-15 修订：规则 1 从"围绕法律手工列举的 domain_pairs"改为
"领域字典 ≥2 个不同领域命中"，保证上传知识库类型多样（金融/医疗/教育/
生活…）时跨领域查询仍能触发多智能体；新增领域只需在 _DOMAIN_KEYWORDS
加一行，任意跨领域组合天然命中。
"""
from __future__ import annotations

from app.services.agent_service import AgentService


def _multi(q: str) -> bool:
    return AgentService._should_use_multi(q)


# ── 旧行为保留：原有跨领域组合仍触发 ─────────────────────────────────────
def test_cross_domain_law_code():
    assert _multi("帮我写个计算劳动赔偿的脚本")


def test_cross_domain_retrieval_writing():
    assert _multi("查一下民法典并写一份摘要")


def test_cross_domain_analysis_code():
    assert _multi("解释这个算法并写代码实现")


# ── 多样性：非法律领域的跨域组合（新增能力）─────────────────────────────
def test_cross_domain_finance_analysis():
    assert _multi("分析一下最近股票走势并整理成报告")


def test_cross_domain_health_retrieval():
    assert _multi("帮我查一下这个药的用法，写一份服用说明")


def test_cross_domain_education_analysis():
    assert _multi("对比考研和留学，帮我制定学习计划")


def test_cross_domain_life_code():
    assert _multi("写一个整理菜谱的工具")


def test_cross_domain_code_writing():
    assert _multi("帮我写一个python脚本")


# ── 单领域 / 无关查询：不触发 ────────────────────────────────────────────
def test_single_domain_law_not_multi():
    assert not _multi("民法典第10条是什么")


def test_single_domain_health_not_multi():
    assert not _multi("这个药一天吃几次")


def test_single_domain_contract_not_multi():
    assert not _multi("帮我把合同里的赔偿条款梳理一下")


def test_plain_question_not_multi():
    assert not _multi("今天天气怎么样")


def test_greeting_not_multi():
    assert not _multi("你好")


# ── 规则 2：长查询 + 连词（无领域词也触发）──────────────────────────────
def test_long_query_with_connector():
    long_q = (
        "我想了解一下公司的整体运营情况，包括各部门的工作进展、最近的项目安排、"
        "团队的人员配置以及遇到的各种问题，需要把这些信息都整理清楚，"
        "然后再看看下个季度的整体计划应该怎么制定才更合理。"
    )
    assert len(long_q) > 80
    assert _multi(long_q)


def test_long_query_without_connector_not_multi():
    long_q = "这是一个非常长的没有连接词的普通查询" * 8
    assert len(long_q) > 80
    assert not _multi(long_q)


# ── 领域字典可扩展性：新增领域即生效（组合任意两个领域）─────────────────
def test_domain_keywords_cover_multiple_areas():
    domains = AgentService._DOMAIN_KEYWORDS
    assert len(domains) >= 8, "领域字典应覆盖多个领域（多样性）"
    for name in ("法律", "代码/计算", "金融/财经", "医疗/健康", "教育/学术"):
        assert name in domains
        assert domains[name], f"领域 {name} 关键词不应为空"


def test_any_two_domains_trigger():
    # 任取两个领域各一个关键词组合 → 都应触发（不再依赖手工 pair 枚举）
    a = AgentService._DOMAIN_KEYWORDS["金融/财经"][0]
    b = AgentService._DOMAIN_KEYWORDS["医疗/健康"][0]
    assert _multi(f"帮我同时处理{a}和{b}的问题")


# ── 阶段 3：multi / auto 命中改指 deepagents（Orchestrator 冻结）───────
def test_multi_mode_routes_to_deepagents(monkeypatch):
    """AGENT_MODE=multi → 转发 _run_deep（不再进 orchestrator）。"""
    from types import SimpleNamespace
    import app.services.agent_service as svc_mod

    calls = {}

    def _fake_run_deep(self, query, **kwargs):
        calls["deep"] = (query, kwargs)
        return {"final_answer": "deep", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", SimpleNamespace(AGENT_MODE="multi"))
    monkeypatch.setattr(AgentService, "_run_deep", _fake_run_deep)
    svc = object.__new__(AgentService)
    result = svc.run("帮我同时处理股票和药品的问题", session_id="s1")
    assert result["final_answer"] == "deep"
    assert calls["deep"][0] == "帮我同时处理股票和药品的问题"


def test_auto_complex_query_routes_to_deepagents(monkeypatch):
    """auto + 跨领域命中 → _run_deep；单领域仍走单 Agent 路径。"""
    from types import SimpleNamespace
    import app.services.agent_service as svc_mod

    deep_called = {"n": 0}

    def _fake_run_deep(self, query, **kwargs):
        deep_called["n"] += 1
        return {"final_answer": "deep", "is_fallback": False}

    monkeypatch.setattr(svc_mod, "cfg", SimpleNamespace(AGENT_MODE="auto"))
    monkeypatch.setattr(AgentService, "_run_deep", _fake_run_deep)
    svc = object.__new__(AgentService)
    svc._sessions = None
    # 跨领域（股票 + 药）→ 命中规则 → deepagents
    svc.run("分析一下股票，再查一下这个药的用法", session_id="s2")
    assert deep_called["n"] == 1


def test_auto_simple_query_stays_single(monkeypatch):
    """auto + 简单查询 → 不走 deep，进单 Agent 路径。"""
    from types import SimpleNamespace
    import app.services.agent_service as svc_mod

    deep_called = {"n": 0}

    def _fake_run_deep(self, query, **kwargs):
        deep_called["n"] += 1
        return {"final_answer": "deep"}

    monkeypatch.setattr(svc_mod, "cfg", SimpleNamespace(AGENT_MODE="auto"))
    monkeypatch.setattr(AgentService, "_run_deep", _fake_run_deep)
    monkeypatch.setattr(svc_mod, "get_graph", lambda: None)
    svc = object.__new__(AgentService)
    from app.services.agent_service import SessionStore
    svc._sessions = SessionStore()
    try:
        svc.run("今天天气怎么样", session_id="s3", history=[])
    except Exception:
        pass  # 单 Agent 路径会碰 graph（已 mock 为 None）——只需确认没进 deep
    assert deep_called["n"] == 0
