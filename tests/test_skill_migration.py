"""Skill 存量迁移（DB 列 → 磁盘 SKILL.md）的导出逻辑回归测试。

迁移本身需要 Postgres，这里只覆盖**数据丢失风险所在的那一步** ——
``_export_skill_row_to_disk``：把一行存量记录渲染成 SKILL.md 并落盘。
迁移函数的顺序保证（全部导出成功后才 DROP 旧列）用源码断言兜底。
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.skills.loader import load_skill_directory
from backend.storage.postgres.manager import _export_skill_row_to_disk


@pytest.fixture
def personal_root(tmp_path, monkeypatch):
    """把个人 Skill 根目录指向 tmp_path（避免写进真实 volumes/）。"""
    from app.core.config import get_settings
    from app.skills import registry

    settings = get_settings()
    monkeypatch.setattr(settings, "SKILLS_PERSONAL_DIR", str(tmp_path), raising=False)
    registry.invalidate_cache()
    yield tmp_path
    registry.invalidate_cache()


def _row(**overrides):
    base = {
        "id": uuid.uuid4(),
        "owner_id": uuid.uuid4(),
        "slug": "skill-abc12345",
        "name": "竞品研究",
        "description": "梳理竞品动向",
        "instructions": "## 工作方式\n\n先列维度，再逐项对比。",
        "tool_names_json": '["web_search", "text_tool"]',
        "category": "研究",
        "icon": "globe",
    }
    base.update(overrides)
    return base


def test_exported_row_is_parseable_and_preserves_content(personal_root):
    """导出的 SKILL.md 必须能被 loader 读回，且内容一字不丢。"""
    row = _row()
    _export_skill_row_to_disk(row)

    directory = personal_root / str(row["owner_id"]) / row["slug"]
    definition = load_skill_directory(
        directory, source="personal", owner_id=str(row["owner_id"])
    )
    assert definition.slug == "skill-abc12345"
    assert definition.name == "竞品研究"
    assert definition.description == "梳理竞品动向"
    assert "先列维度，再逐项对比。" in definition.body
    assert definition.tool_dependencies == ("web_search", "text_tool")
    assert definition.category == "研究"
    assert definition.icon == "globe"


def test_export_survives_corrupt_tool_names_json(personal_root):
    """坏的 tool_names_json 不能让整次迁移卡住（降级为无工具依赖）。"""
    row = _row(tool_names_json="{not json at all")
    _export_skill_row_to_disk(row)

    directory = personal_root / str(row["owner_id"]) / row["slug"]
    definition = load_skill_directory(directory, source="personal")
    assert definition.tool_dependencies == ()
    assert "先列维度" in definition.body


def test_export_keeps_unregistered_tool_names(personal_root):
    """存量 Skill 可能引用已下线的 MCP 工具 —— 导出阶段不做注册表校验，
    否则一条坏数据会卡住整次迁移。运行时门控自然忽略未注册工具。"""
    row = _row(tool_names_json='["mcp_gone_tool"]')
    _export_skill_row_to_disk(row)

    directory = personal_root / str(row["owner_id"]) / row["slug"]
    assert load_skill_directory(directory).tool_dependencies == ("mcp_gone_tool",)


def test_export_handles_empty_optional_fields(personal_root):
    """空 description 的存量行也必须导出成可解析的文件。

    旧表 ``description`` 列默认 ``''`` —— 若渲染层不兜底，这类行会导出成
    loader 拒绝解析的文件，而迁移随后就 DROP 掉了旧列：Skill 永久丢失。
    """
    row = _row(description="", category="", icon="", tool_names_json="")
    _export_skill_row_to_disk(row)

    directory = personal_root / str(row["owner_id"]) / row["slug"]
    definition = load_skill_directory(directory)
    assert definition.description == "竞品研究", "空 description 回落到 name"
    assert definition.category == "自定义"
    assert definition.icon == "sparkles"
    assert definition.tool_dependencies == ()


def test_export_handles_empty_instructions(personal_root):
    """正文为空的存量行同样不能让文件变成不可解析。"""
    row = _row(instructions="")
    _export_skill_row_to_disk(row)

    directory = personal_root / str(row["owner_id"]) / row["slug"]
    definition = load_skill_directory(directory)
    assert definition.body == "梳理竞品动向", "空正文回落到 description"


def test_migration_drops_legacy_columns_only_after_full_export():
    """顺序保证：先导出全部行，全部成功才 DROP —— 先删后写会在中途失败时丢数据。"""
    source = (
        Path(__file__).parents[1] / "backend/storage/postgres/manager.py"
    ).read_text(encoding="utf-8")
    body = source.split("async def _migrate_skill_config_to_files")[1].split(
        "\ndef _export_skill_row_to_disk"
    )[0]

    drop_at = body.index("DROP COLUMN instructions")
    guard_at = body.index("keeping legacy columns")
    assert guard_at < drop_at, "失败守卫（return）必须出现在 DROP 之前"
    assert "if failed:" in body[:drop_at]
    # 幂等：slug 列已存在直接返回
    assert "column_name = 'slug'" in body[: body.index("ALTER TABLE")]
