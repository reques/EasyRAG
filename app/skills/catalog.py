"""Built-in skill catalog and the common runtime profile contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SkillProfile:
    id: str
    name: str
    description: str
    instructions: str
    tool_names: tuple[str, ...] = ()
    category: str = "通用"
    icon: str = "sparkles"
    source: str = "builtin"

    @property
    def can_edit(self) -> bool:
        return self.source == "custom"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "tool_names": list(self.tool_names),
            "category": self.category,
            "icon": self.icon,
            "source": self.source,
            "can_edit": self.can_edit,
        }


BUILTIN_SKILLS: tuple[SkillProfile, ...] = (
    SkillProfile(
        id="builtin:knowledge-research",
        name="知识库研究",
        description="优先查阅已授权知识库，整合原文证据并标明信息缺口。",
        instructions=(
            "优先基于当前用户已授权的知识库完成任务。回答时区分原文事实、合理推断和"
            "未知信息；尽可能保留文件来源与条款、章节等定位信息，不得虚构知识库内容。"
        ),
        category="研究",
        icon="book-open",
    ),
    SkillProfile(
        id="builtin:web-research",
        name="联网研究",
        description="使用联网搜索获取时效信息，并对来源进行交叉核验。",
        instructions=(
            "当任务涉及最新动态、公开资料或外部事实时，使用联网搜索。优先选择权威的一手"
            "来源，区分发布日期与事件发生日期；关键结论应附来源，不确定时明确说明。"
        ),
        tool_names=("web_search",),
        category="研究",
        icon="globe",
    ),
    SkillProfile(
        id="builtin:data-analysis",
        name="数据分析",
        description="进行可靠计算、文本统计和结构化结论提炼。",
        instructions=(
            "先明确输入、口径和假设，再进行计算或文本统计。展示必要的计算过程，检查数量级"
            "与异常值，最终以结构化结论和可复核结果交付。"
        ),
        tool_names=("calculator", "text_tool"),
        category="分析",
        icon="chart-no-axes-column",
    ),
    SkillProfile(
        id="builtin:professional-writing",
        name="专业写作",
        description="把材料整理成结构清楚、可直接使用的专业文稿。",
        instructions=(
            "先识别受众、用途和交付格式，再组织内容。使用清晰标题与紧凑段落，避免空泛套话；"
            "事实与引用保持准确，未提供的信息不要擅自补造。"
        ),
        tool_names=("text_tool",),
        category="创作",
        icon="file-pen-line",
    ),
    SkillProfile(
        id="builtin:legal-analysis",
        name="法律分析",
        description="按法条、要件和事实适用关系组织法律问题分析。",
        instructions=(
            "采用法律问题—规则—事实适用—结论的结构。优先引用知识库中的现行法条；涉及"
            "时效性或知识库之外的法律信息时可联网核验。明确区分一般信息与正式法律意见。"
        ),
        tool_names=("web_search",),
        category="专业",
        icon="scale",
    ),
)

_BUILTIN_BY_ID = {skill.id: skill for skill in BUILTIN_SKILLS}


def list_builtin_skills() -> list[SkillProfile]:
    return list(BUILTIN_SKILLS)


def get_builtin_skill(skill_id: str) -> SkillProfile | None:
    return _BUILTIN_BY_ID.get(skill_id)


def merge_tool_names(skills: Iterable[SkillProfile]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        tool_name
        for skill in skills
        for tool_name in skill.tool_names
    ))
