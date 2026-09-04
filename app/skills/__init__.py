"""Skill 系统 — SKILL.md 文件目录 + 渐进式披露（2026-09-04 重构，参照 Yuxi）。

模块分工：

- ``loader``     — SKILL.md 解析与 frontmatter 校验 → ``SkillDefinition``
- ``registry``   — 两来源磁盘索引（builtin / personal），文件为真相
- ``runtime``    — 三层集合（available / effective / activated）+ 工具门控
- ``read_tool``  — ``read_skill`` 工具，渐进式披露的激活入口
- ``middleware`` — ``SkillsMiddleware``，挂在 ``create_agent`` 上

设计要点见 ``docs/plans/2026-09-04-skill-management-refactor-yuxi.md``。
"""

from app.skills.loader import (
    SkillDefinition,
    SkillLoadError,
    load_skill_directory,
    parse_skill_markdown,
    render_skill_markdown,
)
from app.skills.middleware import build_skills_middleware
from app.skills.registry import (
    get_skill,
    invalidate_cache,
    merge_available_skills,
    list_builtin_skills,
    list_personal_skills,
    personal_dir,
    validate_builtin_dependencies,
)
from app.skills.runtime import (
    READ_SKILL_TOOL_NAME,
    SkillRuntimeContext,
    get_active_skill_context,
    resolve_dependency_closure,
    use_skill_context,
)

__all__ = [
    "READ_SKILL_TOOL_NAME",
    "SkillDefinition",
    "SkillLoadError",
    "SkillRuntimeContext",
    "build_skills_middleware",
    "get_active_skill_context",
    "get_skill",
    "invalidate_cache",
    "list_builtin_skills",
    "list_personal_skills",
    "load_skill_directory",
    "merge_available_skills",
    "parse_skill_markdown",
    "personal_dir",
    "render_skill_markdown",
    "resolve_dependency_closure",
    "use_skill_context",
    "validate_builtin_dependencies",
]
