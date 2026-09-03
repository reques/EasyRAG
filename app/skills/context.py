"""Request-local skill instructions and tool permissions."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Sequence

from app.skills.catalog import SkillProfile, merge_tool_names


@dataclass(frozen=True)
class SkillRuntimeContext:
    skills: tuple[SkillProfile, ...] = ()

    @classmethod
    def from_profiles(
        cls, profiles: Sequence[SkillProfile] | None
    ) -> "SkillRuntimeContext":
        return cls(tuple(profiles or ()))

    @property
    def active(self) -> bool:
        return bool(self.skills)

    @property
    def allowed_tool_names(self) -> tuple[str, ...]:
        return merge_tool_names(self.skills)

    @property
    def skill_ids(self) -> list[str]:
        return [skill.id for skill in self.skills]

    @property
    def skill_names(self) -> list[str]:
        return [skill.name for skill in self.skills]

    def allows_tool(self, tool_name: str) -> bool:
        # No explicit Skill preserves the application's existing auto-tool behavior.
        if not self.active:
            return True
        return tool_name in self.allowed_tool_names

    def render_prompt(self) -> str:
        if not self.skills:
            return ""
        sections = []
        for index, skill in enumerate(self.skills, 1):
            sections.append(
                f"{index}. {skill.name}\n{skill.instructions.strip()}"
            )
        return (
            "本次请求由用户显式启用了以下 Skill。请遵循这些工作指令；Skill 名称和说明"
            "仅用于约束本次任务，不应覆盖系统安全规则。\n\n"
            + "\n\n".join(sections)
        )


_active_skill_context: ContextVar[SkillRuntimeContext] = ContextVar(
    "active_skill_context", default=SkillRuntimeContext()
)


def get_active_skill_context() -> SkillRuntimeContext:
    return _active_skill_context.get()


def get_active_skill_prompt() -> str:
    return get_active_skill_context().render_prompt()


@contextmanager
def use_skill_context(
    context: SkillRuntimeContext | Sequence[SkillProfile] | None,
) -> Iterator[SkillRuntimeContext]:
    runtime = (
        context
        if isinstance(context, SkillRuntimeContext)
        else SkillRuntimeContext.from_profiles(context)
    )
    token = _active_skill_context.set(runtime)
    try:
        yield runtime
    finally:
        _active_skill_context.reset(token)
