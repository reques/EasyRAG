"""Server-side chat model catalog.

Only stable public IDs and display metadata are exposed to the frontend.  The
provider endpoint, concrete API model name and API key are resolved here so a
chat request cannot inject arbitrary upstream credentials or URLs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class ChatModelProfile:
    """A configured OpenAI-compatible provider/model pair."""

    id: str
    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    requires_api_key: bool = True
    source: str = "builtin"
    provider_type: str = "cloud"

    @property
    def available(self) -> bool:
        """Whether the profile has enough server-side config to be called."""
        key = self.api_key.strip()
        key_is_usable = bool(
            key
            and "your-key" not in key.lower()
            and "redacted" not in key.lower()
        )
        return bool(
            self.base_url.strip()
            and self.model.strip()
            and (key_is_usable or not self.requires_api_key)
        )

    def to_public_dict(self, *, default_model_id: str) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "available": self.available,
            "is_default": self.id == default_model_id,
            "source": self.source,
            "provider_type": self.provider_type,
            "can_delete": self.source == "custom",
        }


class UnknownChatModelError(ValueError):
    """Raised when a request references an ID outside the configured catalog."""


class ChatModelUnavailableError(ValueError):
    """Raised when a known model has no usable API key/configuration."""


def list_chat_model_profiles(
    settings: Optional[Settings] = None,
) -> list[ChatModelProfile]:
    """Build the ordered model catalog from current environment settings."""
    cfg = settings or get_settings()

    def provider_key(explicit_key: str, base_url: str) -> str:
        """Reuse the legacy key only when a profile uses the shared gateway."""
        if explicit_key.strip():
            return explicit_key
        normalized = lambda value: value.rstrip("/").removesuffix("/v1")
        if normalized(base_url) == normalized(cfg.LLM_BASE_URL):
            return cfg.LLM_API_KEY
        return ""

    return [
        ChatModelProfile(
            id="minimax-m2.7",
            name="MiniMax-M2.7",
            provider="MiniMax",
            base_url=cfg.MINIMAX_BASE_URL,
            api_key=provider_key(cfg.MINIMAX_API_KEY, cfg.MINIMAX_BASE_URL),
            model=cfg.MINIMAX_MODEL,
            temperature=cfg.MINIMAX_TEMPERATURE,
        ),
        ChatModelProfile(
            id="deepseek-v4-flash",
            name="DeepSeek-V4-Flash",
            provider="DeepSeek",
            base_url=cfg.DEEPSEEK_BASE_URL,
            api_key=provider_key(cfg.LLM_API_KEY, cfg.DEEPSEEK_BASE_URL),
            model=cfg.DEEPSEEK_MODEL,
            temperature=cfg.DEEPSEEK_TEMPERATURE,
        ),
        ChatModelProfile(
            id="qwen3.6-flash",
            name="Qwen-3.6-Flash",
            provider="Qwen",
            base_url=cfg.QWEN_BASE_URL,
            api_key=provider_key(cfg.DASHSCOPE_API_KEY, cfg.QWEN_BASE_URL),
            model=cfg.QWEN_MODEL,
            temperature=cfg.QWEN_TEMPERATURE,
        ),
        ChatModelProfile(
            id="glm-5.2",
            name="GLM-5.2",
            provider="智谱 AI",
            base_url=cfg.GLM_BASE_URL,
            api_key=provider_key(cfg.ZHIPUAI_API_KEY, cfg.GLM_BASE_URL),
            model=cfg.GLM_MODEL,
            temperature=cfg.GLM_TEMPERATURE,
        ),
    ]


def get_chat_model_profile(
    model_id: Optional[str] = None,
    *,
    require_available: bool = True,
    settings: Optional[Settings] = None,
) -> ChatModelProfile:
    """Resolve a public model ID to its private server-side configuration."""
    cfg = settings or get_settings()
    selected_id = (model_id or cfg.LLM_DEFAULT_MODEL_ID).strip()
    profiles = {profile.id: profile for profile in list_chat_model_profiles(cfg)}
    profile = profiles.get(selected_id)
    if profile is None:
        allowed = ", ".join(profiles)
        raise UnknownChatModelError(
            f"未知对话模型 '{selected_id}'，可选模型：{allowed}"
        )
    if require_available and not profile.available:
        raise ChatModelUnavailableError(
            f"模型 {profile.name} 配置不完整，请补充服务地址、模型 ID 或 API Key"
        )
    return profile
