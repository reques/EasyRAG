"""Small, deterministic policy engine for the tool execution boundary."""
from __future__ import annotations

import fnmatch
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger
from app.tools.sandbox.context import SandboxContext

logger = get_logger(__name__)
_VALID_MODES = {"disabled", "audit", "enforce"}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    would_allow: bool
    reason: str
    mode: str
    capabilities: Tuple[str, ...]


class PolicyEngine:
    """Evaluate the first matching rule, then the capability deny list."""

    def __init__(self, policy: Optional[Mapping[str, Any]] = None):
        self.policy: Dict[str, Any] = dict(policy or {})
        defaults = self.policy.get("defaults") or {}
        mode = str(defaults.get("mode", "audit")).lower()
        self.mode = mode if mode in _VALID_MODES else "audit"
        self.deny_capabilities = frozenset(str(item) for item in defaults.get("deny_capabilities", ()))

    @classmethod
    def from_file(cls, path: str | Path) -> "PolicyEngine":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("policy root must be an object")
            return cls(data)
        except FileNotFoundError:
            logger.warning("Sandbox policy file %s does not exist; using audit mode", path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.error("Sandbox policy file %s is invalid: %s; using audit mode", path, exc)
        return cls()

    @staticmethod
    def capabilities_for(metadata: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
        values = (metadata or {}).get("capabilities", ("none",))
        if isinstance(values, str):
            values = (values,)
        result = tuple(sorted({str(value) for value in values if str(value)}))
        return result or ("none",)

    def check(
        self,
        *,
        tool_name: str,
        metadata: Optional[Mapping[str, Any]],
        context: SandboxContext,
    ) -> Decision:
        capabilities = self.capabilities_for(metadata)
        if self.mode == "disabled":
            return Decision(True, True, "sandbox disabled", self.mode, capabilities)

        would_allow = True
        reason = "allowed by default"
        for rule in self.policy.get("rules") or ():
            if not isinstance(rule, Mapping) or not self._matches(rule, tool_name, context):
                continue
            if rule.get("allow") is False:
                would_allow, reason = False, "denied by matching rule"
                break
            denied = {str(value) for value in rule.get("deny", ())}
            blocked = sorted(denied.intersection(capabilities))
            if blocked:
                would_allow, reason = False, f"rule denies capability: {', '.join(blocked)}"
                break
            if rule.get("allow") is True:
                would_allow, reason = True, "allowed by matching rule"
                break

        if would_allow:
            blocked = sorted(self.deny_capabilities.intersection(capabilities))
            if blocked:
                would_allow, reason = False, f"default policy denies capability: {', '.join(blocked)}"

        allowed = would_allow or self.mode == "audit"
        if not would_allow and self.mode == "audit":
            reason = f"would deny: {reason}"
        return Decision(allowed, would_allow, reason, self.mode, capabilities)

    @staticmethod
    def _matches(rule: Mapping[str, Any], tool_name: str, context: SandboxContext) -> bool:
        pattern = str(rule.get("tool", "*"))
        if not fnmatch.fnmatchcase(tool_name, pattern):
            return False
        expected_context = rule.get("context") or {}
        if not isinstance(expected_context, Mapping):
            return False
        for key, expected in expected_context.items():
            actual = getattr(context, str(key), None)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


_engine_lock = threading.Lock()
_engine_cache: Dict[str, tuple[Optional[int], PolicyEngine]] = {}


def get_policy_engine() -> PolicyEngine:
    settings = get_settings()
    if not settings.SANDBOX_ENABLED:
        return PolicyEngine({"defaults": {"mode": "disabled"}})
    path = Path(settings.SANDBOX_POLICY_FILE)
    try:
        mtime_ns: Optional[int] = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    cache_key = str(path.resolve())
    with _engine_lock:
        cached = _engine_cache.get(cache_key)
        if cached is None or cached[0] != mtime_ns:
            _engine_cache[cache_key] = (mtime_ns, PolicyEngine.from_file(path))
        return _engine_cache[cache_key][1]
