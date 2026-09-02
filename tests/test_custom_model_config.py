"""Security and behavior tests for persisted custom chat model profiles."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.client import get_active_chat_model_profile, use_chat_model
from backend.services.model_config_service import (
    ModelConfigValidationError,
    decrypt_api_key,
    encrypt_api_key,
    profile_from_custom_model,
    validate_and_normalize_base_url,
)


class CustomRecord:
    id = "11111111-1111-1111-1111-111111111111"
    public_id = "custom:11111111-1111-1111-1111-111111111111"
    name = "Local Qwen"
    provider_name = "Ollama"
    provider_type = "local"
    base_url = "http://localhost:11434/v1"
    model_name = "qwen3:32b"
    api_key_encrypted = None
    requires_api_key = False
    temperature = 0.0


def test_custom_api_key_is_encrypted_at_rest():
    encrypted = encrypt_api_key("secret-test-key")

    assert encrypted
    assert "secret-test-key" not in encrypted
    assert decrypt_api_key(encrypted) == "secret-test-key"


def test_local_openai_compatible_endpoint_can_run_without_key():
    profile = profile_from_custom_model(CustomRecord())

    assert profile.id.startswith("custom:")
    assert profile.available is True
    assert profile.source == "custom"
    public = profile.to_public_dict(default_model_id="deepseek-v4-flash")
    assert public["provider_type"] == "local"
    assert public["can_delete"] is True
    assert "api_key" not in public
    assert "base_url" not in public


def test_local_and_cloud_url_boundaries():
    assert validate_and_normalize_base_url(
        "http://localhost:11434/v1/", "local"
    ) == "http://localhost:11434/v1"
    assert validate_and_normalize_base_url(
        "https://api.example.com/v1/", "cloud"
    ) == "https://api.example.com/v1"

    with pytest.raises(ModelConfigValidationError):
        validate_and_normalize_base_url("http://api.example.com/v1", "cloud")
    with pytest.raises(ModelConfigValidationError):
        validate_and_normalize_base_url("https://127.0.0.1/v1", "cloud")
    with pytest.raises(ModelConfigValidationError):
        validate_and_normalize_base_url("https://api.example.com/v1", "local")


def test_custom_profile_flows_through_request_context_without_static_lookup():
    profile = profile_from_custom_model(CustomRecord())

    with use_chat_model(profile):
        assert get_active_chat_model_profile() is profile
    assert get_active_chat_model_profile() is None


def test_repository_queries_are_owner_scoped():
    source = (
        Path(__file__).parents[1]
        / "backend/repositories/model_config_repository.py"
    ).read_text(encoding="utf-8")

    assert "CustomModelConfig.owner_id == owner_id" in source
    assert "CustomModelConfig.id == record_id" in source
