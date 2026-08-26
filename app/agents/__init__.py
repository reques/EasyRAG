"""多智能体层 — DeepAgents（LangGraph 原生）统一实现（2026-08-26 阶段 5）。

历史上的 Orchestrator-Worker（orchestrator.py / workers/ / blackboard.py）
已退役，统一收敛到 ``app/agents/deep``：

- ``deep.agent``        主 Agent（create_react_agent）+ SubAgent 委派
- ``deep.task_tool``    task 工具（单任务委派 + 熔断）
- ``deep.planner``      spawn_tasks（DAG 并行委派）与 revise_plan（动态重规划）
- ``deep.blackboard``   结构化黑板（任务结果共享 + 订阅）
- ``deep.subagents``    子智能体名册与动态工具绑定

统一事件流见 ``app/agents/events.py``；``AGENT_MODE=multi`` 作为
``deepagents`` 的兼容别名保留（见 app/services/agent_service.py）。
"""

__all__: list[str] = []
