"""Skill 目录索引 — 两来源磁盘扫描（builtin / personal）。

参照 Yuxi 的三来源模型（内置 / 共享 / 个人），本期落地两个：

| 来源 | 位置 | 管理权 |
|---|---|---|
| builtin | ``SKILLS_BUILTIN_DIR``（默认 ``./skills``，随代码发布） | 只读，不可编辑删除 |
| personal | ``SKILLS_PERSONAL_DIR/<user_id>/<slug>/``（默认 ``./volumes/user-skills``） | 归属用户 |

共享来源（shared）与远程安装留作扩展点，见规划文档
``docs/plans/2026-09-04-skill-management-refactor-yuxi.md`` §1.1。

**文件是真相**：Skill 内容只从磁盘读，PG ``custom_skill_configs`` 降级为
索引表（slug / owner / enabled / 展示元数据），不再存 instructions 与
tool_names_json。因此本模块不依赖数据库，可在任何线程/进程内直接调用。

**同名覆盖**：个人 Skill 的 slug 与内置相同时，个人版本覆盖内置（对齐
Yuxi）；删掉个人版本后内置版本自动恢复——因为每次扫描都是重新计算。

健壮性策略沿用 ``deep/subagents.py:_load_subagents_file``：单个坏文件只
warn 跳过，不因一个损坏的 SKILL.md 让整个 Skill 列表不可用。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.core.config import get_settings
from app.core.logger import get_logger
from app.skills.loader import (
    SKILL_FILENAME,
    SLUG_PATTERN,
    SkillDefinition,
    SkillLoadError,
    load_skill_directory,
)

logger = get_logger(__name__)

# 扫描结果缓存：{(source, owner_id): (expires_at, {slug: SkillDefinition})}
_scan_cache: Dict[tuple, tuple[float, Dict[str, SkillDefinition]]] = {}
_scan_lock = threading.RLock()


def _project_root() -> Path:
    """项目根目录（本文件位于 <root>/app/skills/registry.py）。"""
    return Path(__file__).resolve().parents[2]


def _resolve_dir(configured: str) -> Path:
    """把配置的相对路径按项目根解析为绝对路径。"""
    path = Path(configured).expanduser()
    return path if path.is_absolute() else (_project_root() / path).resolve()


def builtin_dir() -> Path:
    return _resolve_dir(get_settings().SKILLS_BUILTIN_DIR)


def personal_root() -> Path:
    return _resolve_dir(get_settings().SKILLS_PERSONAL_DIR)


def personal_dir(owner_id: str, slug: str) -> Path:
    """个人 Skill 目录路径。

    owner_id 与 slug 都经格式校验后拼接，防止 ``../`` 穿越写到别人的目录
    或项目代码树里。owner_id 是 UUID 字符串，slug 已由 SLUG_PATTERN 约束。
    """
    safe_owner = _safe_path_segment(owner_id, "owner_id")
    safe_slug = _safe_path_segment(slug, "slug")
    if not SLUG_PATTERN.match(safe_slug):
        raise SkillLoadError(f"slug {slug!r} 非法：只允许小写字母、数字和短横线")
    return personal_root() / safe_owner / safe_slug


def _safe_path_segment(value: str, field: str) -> str:
    """路径片段安全校验：拒绝分隔符、上级引用、空值与绝对路径。"""
    raw = str(value or "").strip()
    if not raw:
        raise SkillLoadError(f"{field} 不能为空")
    if raw in (".", "..") or "/" in raw or "\\" in raw or "\x00" in raw:
        raise SkillLoadError(f"{field} 含非法路径字符：{value!r}")
    if Path(raw).is_absolute():
        raise SkillLoadError(f"{field} 不能是绝对路径：{value!r}")
    return raw


def _scan_directory(
    root: Path, source: str, owner_id: Optional[str]
) -> Dict[str, SkillDefinition]:
    """扫描一个根目录下的所有 Skill 子目录（单层，不递归）。

    坏文件只 warn 跳过：缺 SKILL.md 的目录、frontmatter 非法的文件都不会
    中断扫描。同一 slug 在同一来源内重复出现时后者告警跳过（先到先得）。
    """
    found: Dict[str, SkillDefinition] = {}
    if not root.is_dir():
        logger.debug("[skills] scan skipped, not a directory: %s", root)
        return found

    try:
        entries = sorted(root.iterdir())
    except OSError as exc:
        logger.warning("[skills] cannot list %s: %s", root, exc)
        return found

    for entry in entries:
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if not (entry / SKILL_FILENAME).is_file():
            logger.debug("[skills] skip %s (no %s)", entry, SKILL_FILENAME)
            continue
        try:
            definition = load_skill_directory(entry, source=source, owner_id=owner_id)
        except SkillLoadError as exc:
            logger.warning("[skills] skip invalid skill %s: %s", entry, exc)
            continue
        except Exception as exc:  # 防御：yaml/编码等意外异常不应中断扫描
            logger.warning("[skills] skip skill %s (unexpected): %s", entry, exc)
            continue
        if definition.slug in found:
            logger.warning(
                "[skills] duplicate slug %r in %s (keeping %s)",
                definition.slug, entry, found[definition.slug].path,
            )
            continue
        found[definition.slug] = definition
    return found


def _cached_scan(
    root: Path, source: str, owner_id: Optional[str]
) -> Dict[str, SkillDefinition]:
    """带 TTL 缓存的扫描（TTL=0 时直通，便于开发调试与测试）。"""
    ttl = get_settings().SKILLS_SCAN_CACHE_TTL
    if ttl <= 0:
        return _scan_directory(root, source, owner_id)

    key = (source, owner_id, str(root))
    now = time.monotonic()
    with _scan_lock:
        cached = _scan_cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    result = _scan_directory(root, source, owner_id)
    with _scan_lock:
        _scan_cache[key] = (now + ttl, result)
    return result


def invalidate_cache(owner_id: Optional[str] = None) -> None:
    """清除扫描缓存（保存/删除个人 Skill 后调用）。

    owner_id 为 None 时清空全部（内置目录变更、测试隔离）。
    """
    with _scan_lock:
        if owner_id is None:
            _scan_cache.clear()
            return
        for key in [k for k in _scan_cache if k[1] == owner_id]:
            _scan_cache.pop(key, None)


def list_builtin_skills() -> List[SkillDefinition]:
    """内置 Skill（按 slug 排序，保证列表稳定）。"""
    return sorted(
        _cached_scan(builtin_dir(), "builtin", None).values(), key=lambda s: s.slug
    )


def list_personal_skills(owner_id: str) -> List[SkillDefinition]:
    """某用户的个人 Skill（按 slug 排序）。owner_id 为空时返回空列表。"""
    if not owner_id:
        return []
    try:
        root = personal_root() / _safe_path_segment(owner_id, "owner_id")
    except SkillLoadError as exc:
        logger.warning("[skills] invalid owner_id for personal scan: %s", exc)
        return []
    return sorted(
        _cached_scan(root, "personal", owner_id).values(), key=lambda s: s.slug
    )


def merge_available_skills(
    personal: Sequence[SkillDefinition] = (),
) -> List[SkillDefinition]:
    """内置 + 个人 Skill 合并为"用户可访问集合"（三层模型第一层）。

    ``personal`` 由调用方提供 —— **不在这里扫个人目录**，因为个人 Skill 的
    权威清单是 PG 索引表（``custom_skill_configs``，含 ``is_active``），
    只按目录扫会把索引里已移除的条目也翻出来。调用方（``chat_router``
    的 ``_accessible_skills``）先查索引、再按索引行读盘，然后交给本函数合并。

    同名 slug 时**个人版本覆盖内置**（对齐 Yuxi）；删掉个人版本后内置自动
    恢复 —— 因为每次都是重新计算，没有持久化的覆盖记录。

    排序：内置在前、个人在后，各自按 slug —— 前端列表顺序稳定，用户新建的
    Skill 出现在末尾。
    """
    merged: Dict[str, SkillDefinition] = {
        skill.slug: skill for skill in list_builtin_skills()
    }
    overridden = [s.slug for s in personal if s.slug in merged]
    for skill in personal:
        merged[skill.slug] = skill
    if overridden:
        logger.info(
            "[skills] personal skills override builtin: %s", ", ".join(overridden)
        )
    builtin_first = [s for s in merged.values() if s.source == "builtin"]
    personal_last = [s for s in merged.values() if s.source == "personal"]
    return sorted(builtin_first, key=lambda s: s.slug) + sorted(
        personal_last, key=lambda s: s.slug
    )


def get_skill(slug: str, owner_id: Optional[str] = None) -> Optional[SkillDefinition]:
    """按 slug 取一个可访问的 Skill（个人优先于内置）。找不到返回 None。"""
    if not slug:
        return None
    for skill in list_personal_skills(owner_id or ""):
        if skill.slug == slug:
            return skill
    return _cached_scan(builtin_dir(), "builtin", None).get(slug)


def validate_builtin_dependencies() -> List[str]:
    """启动自检：内置 Skill 的依赖是否可解析。返回问题列表（不抛异常）。

    工具依赖只 warn 不阻断 —— MCP 工具是运行时动态注册的，启动时必然查不到
    （见规划文档 §2.1）。Skill 依赖指向不存在的 slug 才是真配置错误。
    """
    problems: List[str] = []
    builtin = _cached_scan(builtin_dir(), "builtin", None)
    for skill in builtin.values():
        for dep in skill.skill_dependencies:
            if dep not in builtin:
                problems.append(
                    f"{skill.slug}: skill_dependencies 指向不存在的内置 Skill {dep!r}"
                )
    try:
        from app.tools.registry import get_tool_registry

        known = set(get_tool_registry().list_names(available_only=False))
    except Exception as exc:  # 注册表未初始化时跳过工具校验
        logger.debug("[skills] tool registry unavailable for validation: %s", exc)
        return problems
    for skill in builtin.values():
        for tool_name in skill.tool_dependencies:
            if tool_name not in known:
                logger.warning(
                    "[skills] %s declares unregistered tool %r "
                    "(ok if it is an MCP tool registered at runtime)",
                    skill.slug, tool_name,
                )
    return problems


__all__ = [
    "builtin_dir",
    "get_skill",
    "invalidate_cache",
    "merge_available_skills",
    "list_builtin_skills",
    "list_personal_skills",
    "personal_dir",
    "personal_root",
    "validate_builtin_dependencies",
]
