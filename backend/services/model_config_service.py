"""Validation and credential protection for user-created model endpoints."""
from __future__ import annotations

import base64
import hashlib
import ipaddress
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings
from app.llm.models import ChatModelProfile
from backend.storage.postgres.models_model_config import CustomModelConfig


class ModelConfigValidationError(ValueError):
    """Raised for invalid or unreadable custom model configuration."""


def _credential_cipher() -> Fernet:
    cfg = get_settings()
    secret = (cfg.MODEL_CONFIG_ENCRYPTION_KEY or cfg.JWT_SECRET_KEY).strip()
    if not secret:
        raise ModelConfigValidationError("后端尚未配置模型凭据加密密钥")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_api_key(api_key: str) -> str | None:
    value = api_key.strip()
    if not value:
        return None
    return _credential_cipher().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_api_key(encrypted: str | None) -> str:
    if not encrypted:
        return ""
    try:
        return _credential_cipher().decrypt(encrypted.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeError) as exc:
        raise ModelConfigValidationError(
            "模型 API Key 无法解密，请删除后重新添加该模型"
        ) from exc


def validate_and_normalize_base_url(base_url: str, provider_type: str) -> str:
    """Validate endpoint shape and apply a conservative SSRF boundary."""
    raw = base_url.strip()
    if not raw:
        raise ModelConfigValidationError("请填写 API Base URL")
    try:
        parsed = urlsplit(raw)
        _ = parsed.port  # force invalid-port validation
    except ValueError as exc:
        raise ModelConfigValidationError("API Base URL 格式不正确") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ModelConfigValidationError("API Base URL 必须是完整的 http/https 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ModelConfigValidationError("API Base URL 不能包含账号、查询参数或片段")

    host = parsed.hostname.lower()
    if provider_type == "cloud":
        if parsed.scheme != "https":
            raise ModelConfigValidationError("云端模型必须使用 HTTPS 地址")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address and (address.is_private or address.is_loopback or address.is_link_local):
            raise ModelConfigValidationError("云端模型不能使用本机或私有网段地址")
    elif provider_type == "local":
        # Local endpoints may be loopback, LAN IPs, mDNS names or Docker
        # service names. Public domain names belong in the cloud option.
        allowed_name = (
            host in {"localhost", "host.docker.internal"}
            or host.endswith(".local")
            or "." not in host
        )
        try:
            address = ipaddress.ip_address(host)
            allowed_name = address.is_private or address.is_loopback or address.is_link_local
        except ValueError:
            pass
        if not allowed_name:
            raise ModelConfigValidationError(
                "本地模型地址仅支持 localhost、私有网段、.local 或 Docker 服务名"
            )
    else:
        raise ModelConfigValidationError("模型类型必须是 local 或 cloud")

    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def profile_from_custom_model(record: CustomModelConfig) -> ChatModelProfile:
    """Convert an owner-authorized DB record into a private runtime profile."""
    return ChatModelProfile(
        id=record.public_id,
        name=record.name,
        provider=record.provider_name,
        provider_type=record.provider_type,
        source="custom",
        base_url=record.base_url,
        api_key=decrypt_api_key(record.api_key_encrypted),
        model=record.model_name,
        temperature=record.temperature,
        requires_api_key=record.requires_api_key,
        supports_vision=bool(record.supports_vision),
    )
