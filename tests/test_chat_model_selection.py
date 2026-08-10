"""Regression tests for the environment-backed conversation model selector."""
from __future__ import annotations

import pytest

from app.agents.orchestrator import Orchestrator
from app.agents.workers.base import TaskBrief, WorkerReport
from app.core.config import get_settings
from app.llm.client import get_active_chat_model_id, use_chat_model
from app.llm.models import (
    ChatModelProfile,
    ChatModelUnavailableError,
    UnknownChatModelError,
    get_chat_model_profile,
    list_chat_model_profiles,
)


def _settings(**updates):
    return get_settings().model_copy(update=updates)


def _profile(model_id: str) -> ChatModelProfile:
    return ChatModelProfile(
        id=model_id,
        name=model_id,
        provider="test",
        base_url="https://gateway.example/v1",
        api_key="test-key",
        model=model_id,
        temperature=0.0,
    )


def test_catalog_contains_only_the_four_requested_models():
    profiles = list_chat_model_profiles(_settings())

    assert [profile.id for profile in profiles] == [
        "minimax-m2.7",
        "deepseek-v4-flash",
        "qwen3.6-flash",
        "glm-5.2",
    ]
    public = profiles[0].to_public_dict(default_model_id="deepseek-v4-flash")
    assert "api_key" not in public
    assert "base_url" not in public
    assert "model" not in public


def test_shared_gateway_profiles_reuse_legacy_api_key():
    gateway = "https://gateway.example/v1"
    cfg = _settings(
        LLM_BASE_URL=gateway,
        LLM_API_KEY="test-shared-key",
        MINIMAX_BASE_URL=gateway,
        MINIMAX_API_KEY="",
        DEEPSEEK_BASE_URL=gateway,
        QWEN_BASE_URL=gateway,
        DASHSCOPE_API_KEY="",
        GLM_BASE_URL=gateway,
        ZHIPUAI_API_KEY="",
    )

    profiles = list_chat_model_profiles(cfg)

    assert all(profile.available for profile in profiles)
    assert {profile.api_key for profile in profiles} == {"test-shared-key"}


def test_direct_provider_without_key_is_known_but_unavailable():
    cfg = _settings(
        LLM_BASE_URL="https://gateway.example/v1",
        QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1",
        DASHSCOPE_API_KEY="",
    )

    with pytest.raises(ChatModelUnavailableError):
        get_chat_model_profile("qwen3.6-flash", settings=cfg)

    profile = get_chat_model_profile(
        "qwen3.6-flash", require_available=False, settings=cfg
    )
    assert profile.available is False


def test_unknown_model_is_rejected_by_allowlist():
    with pytest.raises(UnknownChatModelError):
        get_chat_model_profile(
            "attacker-controlled-model",
            settings=_settings(),
            require_available=False,
        )


def test_request_model_context_is_reset(monkeypatch):
    monkeypatch.setattr(
        "app.llm.models.get_chat_model_profile",
        lambda model_id: _profile(model_id),
    )

    assert get_active_chat_model_id() is None
    with use_chat_model("glm-5.2"):
        assert get_active_chat_model_id() == "glm-5.2"
    assert get_active_chat_model_id() is None


def test_parallel_workers_inherit_selected_model(monkeypatch):
    monkeypatch.setattr(
        "app.llm.models.get_chat_model_profile",
        lambda model_id: _profile(model_id),
    )

    class CapturingWorker:
        blackboard = None
        tool_callback = None

        def run_with_board(self, brief):
            return WorkerReport(
                task_id=brief.task_id,
                worker_name="rag",
                status="done",
                summary=get_active_chat_model_id() or "",
            )

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._workers = {"rag": CapturingWorker()}
    orchestrator._default_worker = "rag"
    orchestrator.blackboard = object()

    with use_chat_model("minimax-m2.7"):
        reports = orchestrator._dispatch_parallel(
            [TaskBrief(task_id="task-1", goal="test", worker_hint="rag")],
            [],
        )

    assert reports[0].summary == "minimax-m2.7"
