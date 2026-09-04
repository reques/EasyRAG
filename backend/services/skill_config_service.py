"""个人 Skill 的文件读写与 DB 索引同步（2026-09-04 Skill 重构）。

**文件是真相**：Skill 内容（正文 + tool_dependencies）存
``volumes/user-skills/<owner_id>/<slug>/SKILL.md``；PG ``custom_skill_configs``
降级为索引表 —— 保留 slug / owner / name / description / enabled，用于列表
查询与唯一性约束，不再存 ``instructions`` 与 ``tool_names_json``。

写顺序（先文件后 DB）与删顺序（先 DB 后文件）都按"失败时留下可修复状态"
选择：
- 保存：文件写成功但 DB 失败 → 磁盘上多一个目录，扫描能读到但列表查不到，
  下次同 slug 保存会覆盖，无数据丢失；
- 删除：DB 删成功但文件失败 → 孤儿目录，由 ``prune_orphan_directories``
  在启动时清理。

反过来（先 DB 后文件保存）会在文件写失败时留下一条指向空目录的索引行，
``registry`` 扫不到内容，用户看到一个点开是空的 Skill。
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.core.logger import get_logger
from app.skills.loader import (
    MAX_BODY_LENGTH,
    SKILL_FILENAME,
    SLUG_PATTERN,
    SkillDefinition,
    SkillLoadError,
    load_skill_directory,
    render_skill_markdown,
)
from app.skills.registry import invalidate_cache, personal_dir, personal_root
from app.tools.registry import get_tool_registry
from backend.storage.postgres.models_skill_config import CustomSkillConfig

logger = get_logger(__name__)

MAX_TOOL_DEPENDENCIES = 8


class SkillConfigValidationError(ValueError):
    pass


def validate_tool_names(tool_names: Iterable[str]) -> tuple[str, ...]:
    """校验工具名：去重、限量、必须已注册。

    ``available_only=False`` 是有意的 —— 未配置 API Key 的工具（check_fn 失败）
    仍可写进 Skill，配置补齐后自动可用。
    """
    normalized = tuple(
        dict.fromkeys(str(name).strip() for name in tool_names if str(name).strip())
    )
    if len(normalized) > MAX_TOOL_DEPENDENCIES:
        raise SkillConfigValidationError(
            f"一个 Skill 最多配置 {MAX_TOOL_DEPENDENCIES} 个工具"
        )
    known = set(get_tool_registry().list_names(available_only=False))
    unknown = [name for name in normalized if name not in known]
    if unknown:
        raise SkillConfigValidationError("包含未注册工具：" + "、".join(unknown))
    return normalized


def validate_slug(slug: str) -> str:
    """校验用户提供的 slug（目录名，必须是 URL/路径安全的）。"""
    raw = (slug or "").strip().lower()
    if not raw:
        raise SkillConfigValidationError("slug 不能为空")
    if not SLUG_PATTERN.match(raw):
        raise SkillConfigValidationError(
            "slug 只能包含小写字母、数字和单个短横线（如 my-skill）"
        )
    if len(raw) > 128:
        raise SkillConfigValidationError("slug 超过 128 字符")
    return raw


def slugify(name: str, *, fallback: str = "") -> str:
    """从名称生成 slug；中文名等无法转换时用 fallback（通常是 UUID 前缀）。

    只保留 ASCII 字母数字，其余折成短横线 —— 中文名会得到空串，因此
    必须提供 fallback。
    """
    import re

    ascii_only = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    candidate = ascii_only.strip("-")[:128].strip("-")
    if candidate and SLUG_PATTERN.match(candidate):
        return candidate
    if fallback:
        return fallback
    return f"skill-{uuid.uuid4().hex[:8]}"


def validate_body(body: str) -> str:
    """校验 SKILL.md 正文（工作指令）。"""
    text = (body or "").strip()
    if not text:
        raise SkillConfigValidationError("工作指令不能为空")
    if len(text) > MAX_BODY_LENGTH:
        raise SkillConfigValidationError(f"工作指令超过 {MAX_BODY_LENGTH} 字符")
    return text


def write_personal_skill(
    *,
    owner_id: uuid.UUID | str,
    slug: str,
    name: str,
    description: str,
    body: str,
    tool_names: Sequence[str] = (),
    skill_dependencies: Sequence[str] = (),
    category: str = "自定义",
    icon: str = "sparkles",
) -> SkillDefinition:
    """把个人 Skill 渲染并落盘，返回回读解析的定义。

    回读是刻意的往返校验：确认 ``render_skill_markdown`` 的输出能被
    ``load_skill_directory`` 解析回等价定义，避免写出一个自己都读不了的文件。
    """
    owner = str(owner_id)
    safe_slug = validate_slug(slug)
    tools = validate_tool_names(tool_names)
    text = render_skill_markdown(
        name=name.strip(),
        description=description.strip(),
        body=validate_body(body),
        slug=safe_slug,
        tool_dependencies=tools,
        skill_dependencies=skill_dependencies,
        category=category.strip() or "自定义",
        icon=icon.strip() or "sparkles",
    )

    try:
        directory = personal_dir(owner, safe_slug)
    except SkillLoadError as exc:
        raise SkillConfigValidationError(str(exc)) from exc

    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SKILL_FILENAME).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise SkillConfigValidationError(f"写入 Skill 文件失败：{exc}") from exc

    invalidate_cache(owner)
    try:
        return load_skill_directory(directory, source="personal", owner_id=owner)
    except SkillLoadError as exc:
        raise SkillConfigValidationError(
            f"Skill 文件写入后无法解析（已保留文件待排查）：{exc}"
        ) from exc


def delete_personal_skill_files(owner_id: uuid.UUID | str, slug: str) -> bool:
    """删除个人 Skill 目录。返回是否真的删掉了（不存在返回 False，不报错）。"""
    owner = str(owner_id)
    try:
        directory = personal_dir(owner, slug)
    except SkillLoadError as exc:
        logger.warning("[skills] delete skipped, invalid path: %s", exc)
        return False
    if not directory.is_dir():
        return False
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        # 孤儿目录：DB 行已删，文件残留 —— 由 prune_orphan_directories 兜底
        logger.warning("[skills] failed to remove %s: %s", directory, exc)
        return False
    invalidate_cache(owner)
    return True


def read_personal_skill_body(owner_id: uuid.UUID | str, slug: str) -> Optional[str]:
    """读个人 Skill 的正文（配置弹窗编辑用）。找不到返回 None。"""
    owner = str(owner_id)
    try:
        directory = personal_dir(owner, slug)
        definition = load_skill_directory(
            directory, source="personal", owner_id=owner
        )
    except SkillLoadError:
        return None
    return definition.body


def definition_from_record(
    record: CustomSkillConfig, *, owner_id: uuid.UUID | str
) -> Optional[SkillDefinition]:
    """按 DB 索引行定位并加载磁盘上的 Skill 定义。

    文件缺失时返回 None（索引与文件不一致 —— 磁盘被外部删除，或写入中途失败）。
    调用方应据此跳过该行并告警，而不是给用户一个空 Skill。
    """
    owner = str(owner_id)
    try:
        directory = personal_dir(owner, record.slug)
        return load_skill_directory(directory, source="personal", owner_id=owner)
    except SkillLoadError as exc:
        logger.warning(
            "[skills] index row %s (slug=%s) has no readable file: %s",
            record.id, record.slug, exc,
        )
        return None


def prune_orphan_directories(owner_id: str, known_slugs: Iterable[str]) -> list[str]:
    """清理"磁盘有目录、DB 无索引行"的孤儿（删除失败留下的残留）。

    只在明确知道该用户全部有效 slug 时调用（列表查询后），否则会误删。
    返回被清理的 slug 列表。
    """
    root: Path
    try:
        root = personal_root() / str(owner_id)
    except Exception:
        return []
    if not root.is_dir():
        return []
    valid = set(known_slugs)
    removed: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name in valid:
            continue
        try:
            shutil.rmtree(entry)
            removed.append(entry.name)
        except OSError as exc:
            logger.warning("[skills] prune failed for %s: %s", entry, exc)
    if removed:
        invalidate_cache(str(owner_id))
        logger.info("[skills] pruned orphan directories: %s", removed)
    return removed


__all__ = [
    "MAX_TOOL_DEPENDENCIES",
    "SkillConfigValidationError",
    "definition_from_record",
    "delete_personal_skill_files",
    "prune_orphan_directories",
    "read_personal_skill_body",
    "slugify",
    "validate_body",
    "validate_slug",
    "validate_tool_names",
    "write_personal_skill",
]
