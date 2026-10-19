"""Provider-reported context limits for configured chat models.

An absent or failed /models response is unknown, never the application-wide
CHAT_CONTEXT_WINDOW setting. The short cache avoids a network request for every
conversation while preserving the limit observed when a run starts.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import replace

import httpx

from app.llm.models import ChatModelProfile

_cache: dict[tuple[str, str, str, str], tuple[float, int | None, int | None]] = {}
_SUCCESS_TTL = 3600.0
_FAILURE_TTL = 300.0


def _cache_key(profile: ChatModelProfile) -> tuple[str, str, str, str]:
    key_hash = hashlib.sha256(profile.api_key.encode("utf-8")).hexdigest()[:16]
    return (profile.id, profile.base_url.rstrip("/"), profile.model, key_hash)


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


async def provider_context_window(profile: ChatModelProfile) -> ChatModelProfile:
    """Return a profile enriched only by metadata for its exact API model ID."""
    if not profile.available:
        return profile
    key = _cache_key(profile)
    cached = _cache.get(key)
    if cached and cached[0] > time.monotonic():
        window, output = cached[1:]
    else:
        window = output = None
        try:
            headers = {"Authorization": f"Bearer {profile.api_key}"} if profile.api_key else {}
            async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
                response = await client.get(f"{profile.base_url.rstrip('/')}/models", headers=headers)
                response.raise_for_status()
                data = response.json()
            models = data.get("data") if isinstance(data, dict) else None
            if isinstance(models, list):
                exact = next((item for item in models if isinstance(item, dict) and item.get("id") == profile.model), None)
                if exact:
                    window = _positive_int(exact.get("context_window"))
                    output = _positive_int(exact.get("max_output_tokens"))
        except Exception:
            # Metadata discovery is best effort; a provider's /models failure
            # must not prevent a chat request from reaching that provider.
            pass
        _cache[key] = (time.monotonic() + (_SUCCESS_TTL if window else _FAILURE_TTL), window, output)
    return replace(
        profile,
        context_window=window,
        max_output_tokens=output,
        context_source="provider" if window else "unknown",
    )


async def provider_context_windows(profiles: list[ChatModelProfile]) -> list[ChatModelProfile]:
    return list(await asyncio.gather(*(provider_context_window(profile) for profile in profiles)))
