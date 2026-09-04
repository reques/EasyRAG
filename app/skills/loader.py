"""SKILL.md 解析与 frontmatter 校验（参照 Yuxi skills 机制，2026-09-04 阶段 1）。

Skill 从"代码常量 + DB 行"改为"文件系统上的 SKILL.md 目录"，一个 Skill =
一个目录，根级 ``SKILL.md`` 必需，可选 ``prompts/``（参考资料）与 ``tools/``
（脚本；本期只允许存放，无执行通道——见规划文档 §6.5）。

``SKILL.md`` = YAML frontmatter + Markdown 正文：

    ---
    name: 联网研究
    slug: web-research
    description: 使用联网搜索获取时效信息，并对来源进行交叉核验。
    category: 研究
    icon: globe
    tool_dependencies: [web_search]
    skill_dependencies: []
    ---

    ## 何时使用
    ...

必填只有 ``name`` 与 ``description``（对齐 Yuxi）。``slug`` 省略时用 ``name``
当标识，此时 ``name`` 必须是 ``^[a-z0-9]+(-[a-z0-9]+)*$``；中文或含空格的
名称必须显式写 slug（内置 Skill 全部显式写，name 保留中文展示名）。

**渐进式披露的落点在这里**：``body``（正文）与 ``to_public_dict()`` 分离 ——
列表接口和首轮 prompt 都拿不到 body，只有模型 ``read_skill(slug)`` 之后才
展开正文。因此 body 不进任何面向前端的序列化路径。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from app.core.logger import get_logger

logger = get_logger(__name__)

# slug 约束（对齐 Yuxi）：小写字母数字 + 单个短横线分隔，上限 128
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_SLUG_LENGTH = 128
MAX_NAME_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 300
MAX_BODY_LENGTH = 20000
MAX_TOOL_DEPENDENCIES = 8
MAX_SKILL_DEPENDENCIES = 8

SKILL_FILENAME = "SKILL.md"

# frontmatter 分隔线：文件必须以 --- 开头，正文由第二个 --- 之后开始
_FRONTMATTER_RE = re.compile(
    r"^---[ \t]*\r?\n(?P<meta>.*?)\r?\n---[ \t]*(?:\r?\n(?P<body>.*))?$",
    re.DOTALL,
)


class SkillLoadError(ValueError):
    """SKILL.md 解析或校验失败（缺必填字段、slug 非法、依赖超限等）。"""


@dataclass(frozen=True)
class SkillDefinition:
    """一个 Skill 的完整定义（替代旧 ``SkillProfile``）。

    ``body`` 是渐进式披露的受控内容：未激活时不进 prompt、不进 API 响应。
    ``path`` 指向 SKILL.md 绝对路径，``read_skill`` 工具据此读全文。
    """

    slug: str
    name: str
    description: str
    body: str = ""
    tool_dependencies: tuple[str, ...] = ()
    skill_dependencies: tuple[str, ...] = ()
    category: str = "通用"
    icon: str = "sparkles"
    source: str = "builtin"  # builtin | personal
    owner_id: Optional[str] = None  # personal 才有
    path: str = ""

    @property
    def can_edit(self) -> bool:
        """内置 Skill 不可编辑（对齐 Yuxi：管理员也只能启停）。"""
        return self.source == "personal"

    @property
    def directory(self) -> str:
        """Skill 目录（prompts/ 与 tools/ 的父目录）。"""
        return str(Path(self.path).parent) if self.path else ""

    def summary_line(
        self,
        index: Optional[int] = None,
        *,
        gated_tools: Optional[Sequence[str]] = None,
    ) -> str:
        """首轮 prompt 用的一行摘要（渐进式披露第一步：只给 name + description）。

        含 slug 是为了让模型知道 ``read_skill`` 该传什么参数。

        ``gated_tools``：真正需要激活才能用的工具（调用方从
        ``SkillRuntimeContext.gated_tools_of`` 取，已剔除公共工具）。省略时
        用全部 ``tool_dependencies`` —— 本类不依赖工具注册表，无法自行判断
        哪些是公共工具。
        """
        prefix = f"{index}. " if index is not None else "- "
        names = self.tool_dependencies if gated_tools is None else tuple(gated_tools)
        tools = f"（激活后可用工具：{'、'.join(names)}）" if names else ""
        return f"{prefix}{self.name} [slug: {self.slug}] — {self.description}{tools}"

    def to_public_dict(self) -> Dict[str, Any]:
        """前端契约。**不含 body** —— 正文只能经 read_skill / content 端点获取。"""
        return {
            "slug": self.slug,
            # 兼容既有前端：id 字段仍在（值 = slug），前端 selectedSkillIds 无需改
            "id": self.slug,
            "name": self.name,
            "description": self.description,
            "tool_names": list(self.tool_dependencies),
            "skill_dependencies": list(self.skill_dependencies),
            "category": self.category,
            "icon": self.icon,
            "source": self.source,
            "can_edit": self.can_edit,
            "body_available": bool(self.body),
        }


def _coerce_str_list(value: Any, field: str, limit: int) -> tuple[str, ...]:
    """frontmatter 列表字段归一化：接受 list 或逗号分隔字符串，去重保序。"""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        items = [str(part).strip() for part in value]
    else:
        raise SkillLoadError(f"{field} 必须是列表或逗号分隔字符串")
    normalized = tuple(dict.fromkeys(item for item in items if item))
    if len(normalized) > limit:
        raise SkillLoadError(f"{field} 最多 {limit} 项，实际 {len(normalized)} 项")
    return normalized


def _resolve_slug(meta: Dict[str, Any], name: str) -> str:
    """解析 slug：显式优先；省略时用 name，此时 name 必须符合 slug 格式。

    对齐 Yuxi 的语义 —— 中文或含空格的 name 在省略 slug 时校验不通过。
    """
    raw = str(meta.get("slug") or "").strip()
    if not raw:
        raw = name
        if not SLUG_PATTERN.match(raw):
            raise SkillLoadError(
                f"省略 slug 时 name 必须是小写字母、数字和单个短横线（当前 {name!r}）；"
                "中文或含空格的名称请显式指定 slug"
            )
    if len(raw) > MAX_SLUG_LENGTH:
        raise SkillLoadError(f"slug 超过 {MAX_SLUG_LENGTH} 字符")
    if not SLUG_PATTERN.match(raw):
        raise SkillLoadError(
            f"slug {raw!r} 非法：只允许小写字母、数字和单个短横线分隔"
        )
    return raw


def parse_skill_markdown(
    text: str,
    *,
    source: str = "builtin",
    owner_id: Optional[str] = None,
    path: str = "",
) -> SkillDefinition:
    """解析 SKILL.md 文本为 SkillDefinition（不触碰磁盘，便于单测）。

    Raises:
        SkillLoadError: frontmatter 缺失/非法、缺必填字段、字段超限。
    """
    import yaml

    match = _FRONTMATTER_RE.match(text.lstrip("﻿").strip() + "\n")
    if match is None:
        raise SkillLoadError("缺少 YAML frontmatter（文件需以 --- 开头并以 --- 闭合）")

    try:
        meta = yaml.safe_load(match.group("meta")) or {}
    except yaml.YAMLError as exc:
        raise SkillLoadError(f"frontmatter YAML 解析失败：{exc}") from exc
    if not isinstance(meta, dict):
        raise SkillLoadError("frontmatter 必须是键值映射")

    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not name:
        raise SkillLoadError("frontmatter 缺少必填字段 name")
    if not description:
        raise SkillLoadError("frontmatter 缺少必填字段 description")
    if len(name) > MAX_NAME_LENGTH:
        raise SkillLoadError(f"name 超过 {MAX_NAME_LENGTH} 字符")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise SkillLoadError(f"description 超过 {MAX_DESCRIPTION_LENGTH} 字符")

    slug = _resolve_slug(meta, name)
    body = (match.group("body") or "").strip()
    if len(body) > MAX_BODY_LENGTH:
        raise SkillLoadError(f"正文超过 {MAX_BODY_LENGTH} 字符")

    tool_dependencies = _coerce_str_list(
        meta.get("tool_dependencies"), "tool_dependencies", MAX_TOOL_DEPENDENCIES
    )
    skill_dependencies = _coerce_str_list(
        meta.get("skill_dependencies"), "skill_dependencies", MAX_SKILL_DEPENDENCIES
    )
    if slug in skill_dependencies:
        raise SkillLoadError(f"skill_dependencies 不能包含自身（{slug}）")

    if source not in ("builtin", "personal"):
        raise SkillLoadError(f"未知 source：{source}")

    return SkillDefinition(
        slug=slug,
        name=name,
        description=description,
        body=body,
        tool_dependencies=tool_dependencies,
        skill_dependencies=skill_dependencies,
        category=str(meta.get("category") or "通用").strip() or "通用",
        icon=str(meta.get("icon") or "sparkles").strip() or "sparkles",
        source=source,
        owner_id=owner_id,
        path=path,
    )


def load_skill_directory(
    directory: Path,
    *,
    source: str = "builtin",
    owner_id: Optional[str] = None,
) -> SkillDefinition:
    """从 Skill 目录加载定义（根级 SKILL.md 必需）。

    Raises:
        SkillLoadError: 目录不存在、SKILL.md 缺失/不可读、解析失败。
    """
    skill_file = directory / SKILL_FILENAME
    if not skill_file.is_file():
        raise SkillLoadError(f"缺少 {SKILL_FILENAME}：{directory}")
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillLoadError(f"读取 {skill_file} 失败：{exc}") from exc

    definition = parse_skill_markdown(
        text, source=source, owner_id=owner_id, path=str(skill_file.resolve())
    )
    # 目录名与 slug 不一致时以 frontmatter 为准，但要告警——两者不一致会让
    # "按 slug 定位目录"的写路径（保存/删除）产生歧义。
    if directory.name != definition.slug:
        logger.warning(
            "[skills] directory name %r != slug %r (%s)",
            directory.name, definition.slug, skill_file,
        )
    return definition


def render_skill_markdown(
    *,
    name: str,
    description: str,
    body: str,
    slug: str,
    tool_dependencies: Sequence[str] = (),
    skill_dependencies: Sequence[str] = (),
    category: str = "自定义",
    icon: str = "sparkles",
) -> str:
    """把结构化字段渲染为 SKILL.md 文本（个人 Skill 保存 / DB 迁移导出用）。

    与 ``parse_skill_markdown`` 构成往返：**渲染结果必须能被解析回等价定义**。
    这条保证对迁移是硬要求 —— 导出成一个自己都读不了的文件，等于在删掉旧列
    之后永久丢失该 Skill。

    因此对解析器的必填字段做兜底：``description`` 为空时回落到 ``name``
    （旧表 description 列默认 ``''``，存量行确实可能为空），``body`` 为空时
    回落到 description。两处兜底都记在这里而不是调用方，避免每个调用方
    各自遗漏。
    """
    import yaml

    safe_name = (name or "").strip() or slug
    safe_description = (description or "").strip() or safe_name
    safe_body = (body or "").strip() or safe_description

    meta: Dict[str, Any] = {
        "name": safe_name,
        "slug": slug,
        "description": safe_description,
        "category": (category or "").strip() or "自定义",
        "icon": (icon or "").strip() or "sparkles",
        "tool_dependencies": list(tool_dependencies),
        "skill_dependencies": list(skill_dependencies),
    }
    frontmatter = yaml.safe_dump(
        meta, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{safe_body}\n"


__all__ = [
    "MAX_BODY_LENGTH",
    "MAX_SKILL_DEPENDENCIES",
    "MAX_TOOL_DEPENDENCIES",
    "SKILL_FILENAME",
    "SLUG_PATTERN",
    "SkillDefinition",
    "SkillLoadError",
    "load_skill_directory",
    "parse_skill_markdown",
    "render_skill_markdown",
]
