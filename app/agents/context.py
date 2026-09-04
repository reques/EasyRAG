"""L2 AgentContext — 一次 Run 的声明层运行时配置（Yuxi 参照重构，阶段 2）。

三层状态模型的第二层（见 ``docs/plans/2026-09-02-agentstate-refactor-yuxi.md`` §2.2）：

- **L1 AgentState**（图状态）：节点/工具循环中逐轮读写、随 checkpoint 持久化 —— 阶段 3；
- **L2 AgentContext**（本模块）：一次 Run 开始时确定、执行期**只读**的配置与权限，
  绝不放进 State（Yuxi 把 model/system_prompt 都放 Context 就是这个原因）；
- **L3 RunPayload**（对外契约）：跨进程边界的序列化形态 —— 阶段 4。

判据：凡"Run 开始时定、执行期不变"的字段进这里；凡"每轮变化"的进 L1。

门面模式（替换而非并存）：``BaseContext`` / ``ChatContext`` 是**声明层**，
既有 ContextVar 是**传播层** —— 工具签名统一为 ``fn(**kwargs) -> str``，
工具侧读取（skill 白名单 / KB 授权 / chat model 选择 / 请求级 trace /
事件日志 / 子 Agent 观察者）仍走各自 ContextVar（传播层实现保留）；
``app.agents.request_context.use_request_context`` 提供唯一进入点，
``snapshot_request_context()`` / ``run_with_request_context()``
（app/agents/events.py）的快照重放机制原样复用。

调用方收拢：``AgentService.run`` 及三条执行路径（dynamic / deep /
prepare_context）此前各带 8 个散装请求参数（user_id / knowledge_base_ids /
knowledge_catalog / image_data / history / on_step / on_artifact / session_id），
现收拢为单个 ``ChatContext`` 参数 —— 这是本阶段对调用方代码量的最大削减。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# on_step(step, detail) / on_artifact({kind, stage, title, content})
StepCallback = Callable[..., Any]
ArtifactCallback = Callable[..., Any]


@dataclass(kw_only=True)
class BaseContext:
    """一次 Agent Run 的基础运行时配置（声明层）。

    Yuxi BaseContext 语义对齐：thread/user/run 标识 + 执行期只读的权限范围。
    权限字段（knowledge_base_ids / skill_ids）是 auth metadata 的载体 ——
    序列化给前端/日志时按角色剔除（Yuxi 的 filter_config_by_role 模式，
    L3 Payload 阶段接入）。
    """

    thread_id: str
    """会话线程标识 = conversation_id / session_id。"""

    user_id: Optional[str] = None
    """请求用户（语义记忆 facts 注入、知识库归属裁决）。"""

    run_id: Optional[str] = None
    """本次委派运行的标识（持久化层回填；Agent 执行期只读）。"""

    model_id: str = ""
    """请求选择的 chat model（Yuxi 把 model 放 Context：执行期只读、不进 State）。"""

    skill_ids: Tuple[str, ...] = ()
    """有效集合的 Skill slug（2026-09-04 重构：语义从"要注入的指令"改为"本次
    请求可用的 Skill 范围"；正文与工具由 SkillsMiddleware 按 activated_skills
    逐轮裁决，传播层见 app/skills/runtime.py）。"""

    preload_skill_slugs: Tuple[str, ...] = ()
    """预加载的 Skill（对齐 Yuxi ``preload_skills``）：Run 开始时就进激活集，
    首轮即展开正文与工具，不走渐进式披露。必须是 skill_ids 的子集。"""

    knowledge_base_ids: Tuple[str, ...] = ()
    """已授权知识库 UUID（工具侧 kb_search 授权边界）。"""

    deep_research: bool = False
    """深度研究开关（请求级选择，当前由路由层消费）。"""


@dataclass(kw_only=True)
class ChatContext(BaseContext):
    """Chat 场景的完整上下文：Base 之上加大 payload 与流式回调。

    大 payload（image_data）也放 Context —— 只读、不进 checkpoint（L1）。
    流式回调（on_step / on_artifact）是 SSE 桥接的执行期常量，同理。
    ``history`` 对齐 Yuxi ChatContext.history_window：压缩后的对话历史
    （摘要 + 最近 N 轮），Run 开始时确定、执行期只读。
    """

    query: str = ""
    """用户本次提问原文（路由层裁决后的 effective_query）。"""

    knowledge_catalog: Tuple[Dict[str, Any], ...] = ()
    """知识库目录展示元数据（owner-scoped 查询随授权一起加载）。"""

    image_data: Optional[str] = None
    """图片 data URL（多模态输入；OCR 裁决由路由层完成）。"""

    history: Tuple[Dict[str, Any], ...] = ()
    """对话历史（DB 压缩窗口或 SessionStore；执行期只读）。"""

    on_step: Optional[StepCallback] = None
    """阶段回调 fn(step, detail)，SSE 实时透传。"""

    on_artifact: Optional[ArtifactCallback] = None
    """工件回调 fn({kind, stage, title, content})，SSE 实时透传。"""

    @classmethod
    def from_request(
        cls,
        query: str,
        *,
        thread_id: str,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        skill_ids: Optional[Sequence[str]] = None,
        preload_skill_slugs: Optional[Sequence[str]] = None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        on_step: Optional[StepCallback] = None,
        on_artifact: Optional[ArtifactCallback] = None,
    ) -> "ChatContext":
        """工厂方法：从路由层散装请求参数构造 ChatContext（阶段 2）。

        统一 send/stream 两端的构造逻辑，消除三处拼装重复。
        user_id 支持 str 与对象（如 current_user.id = uuid.UUID），自动 str()。
        """
        uid = str(user_id) if user_id is not None else None
        return cls(
            thread_id=thread_id,
            user_id=uid,
            model_id=model_id or "",
            skill_ids=tuple(skill_ids or ()),
            preload_skill_slugs=tuple(preload_skill_slugs or ()),
            knowledge_base_ids=tuple(knowledge_base_ids or ()),
            knowledge_catalog=tuple(knowledge_catalog or ()),
            image_data=image_data,
            history=tuple(history or ()),
            on_step=on_step,
            on_artifact=on_artifact,
            query=query,
        )

    def history_list(self) -> List[Dict[str, Any]]:
        """history 的可变副本（部分执行路径会向历史后追加消息）。"""
        return [dict(t) for t in self.history]


__all__ = ["BaseContext", "ChatContext"]
