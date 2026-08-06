"""阶段 3: Orchestrator-Worker 多智能体层。

提供任务拆解、Worker 派发与汇总能力，架在现有单 Agent 路径之上。
"""

from app.agents.orchestrator import Orchestrator, get_orchestrator

__all__ = ["Orchestrator", "get_orchestrator"]
