"""Orchestrator — 任务拆解、Worker 派发与结果汇总。

借鉴 subagent-driven-development 的「任务简报 brief」思想：
LLM 拆解用户查询为结构化 TaskBrief，按 worker_hint 路由到专家 Worker，
汇总各 WorkerReport 生成最终回答。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from app.agents.workers.base import BaseWorker, TaskBrief, WorkerReport
from app.agents.workers.rag_worker import RagWorker
from app.agents.workers.legal_worker import LegalWorker
from app.agents.workers.code_worker import CodeWorker
from app.core.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """多智能体编排器：拆解 → 派发 → 汇总。"""

    def __init__(self):
        self._workers: Dict[str, BaseWorker] = {}
        self._default_worker: str = "rag"
        self._llm = None
        self.blackboard = None  # M2: 每次 run() 时创建
        self._build_default_registry()

    # ── LLM client（lazy，可注入 mock）─────────────────────────────────────
    @property
    def llm(self):
        if self._llm is not None:
            return self._llm
        from app.llm.client import get_llm_client

        # Resolve on every access so a process-level Orchestrator singleton
        # cannot pin the model chosen by the first chat request.
        return get_llm_client()

    @llm.setter
    def llm(self, value):
        self._llm = value

    def _build_default_registry(self):
        """注册默认 Worker。"""
        for worker_cls in (RagWorker, LegalWorker, CodeWorker):
            w = worker_cls()
            self._workers[w.name] = w
        logger.info("[orchestrator] registered workers: %s", list(self._workers.keys()))

    # ── 拆解 prompt ─────────────────────────────────────────────────────────
    _DECOMPOSE_PROMPT = """你是一个任务拆解专家。分析用户查询，判断是否需要拆分为多个子任务，并分配给合适的专家 Worker。

可用 Worker 名册：
- rag: 知识库问答专家，擅长检索和回答知识库相关问题
- legal: 法律专家，擅长法律法规查询、条文解读、合规分析
- code: 代码专家，擅长编写、解释、调试代码

输出严格 JSON 格式（不要 markdown fence）：
{{
  "needs_decomposition": true/false,
  "sub_tasks": [
    {{
      "task_id": "task-1",
      "goal": "子任务目标描述",
      "worker_hint": "rag|legal|code",
      "context": "背景信息（可为空字符串）",
      "constraints": ["约束1", "约束2"]
    }}
  ],
  "execution_mode": "parallel|sequential",
  "final_instruction": "汇总要求（如何整合各子任务结果）"
}}

规则：
1. 单一意图查询（如纯问答、纯代码）返回 needs_decomposition=false，sub_tasks 为空
2. 拆解为 2-4 个子任务，避免过度拆分
3. task_id 格式为 task-N（N 从 1 开始）
4. execution_mode: 子任务间无依赖用 parallel，有依赖用 sequential
5. final_instruction 说明如何整合结果（如"综合法律条文和计算脚本给出完整方案"）

用户查询：{query}"""

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        status_callback=None,
        worker_done_callback=None,
        return_synthesize_payload: bool = False,
        tasks_callback=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """执行多智能体编排，返回与单 Agent 兼容的响应格式。

        status_callback: 可选回调 fn(step, detail, task_id="")，在关键步骤时调用，
                         供 SSE 流式端点透传状态事件到前端。
        worker_done_callback: 可选回调 fn(report: WorkerReport)，每个 Worker
                         完成时调用，供 SSE 实时推送子任务产出（边执行边输出）。
        tasks_callback: 可选回调 fn(tasks: list[dict])，拆解完成后一次性传入
                         子任务清单 [{task_id, goal, worker_hint}]，供前端侧边
                         任务进度面板渲染待办列表。
        return_synthesize_payload: True 时跳过 LLM 汇总，在 result 中返回
                         synthesize_payload（含 query/reports/final_inst），
                         由调用方（chat_router）在主事件循环中流式整合——
                         避免 executor 线程里 asyncio.run 调异步流式与主 loop 冲突。
        """
        start = time.perf_counter()
        steps = [f"orchestrator 接收查询: {query[:80]}"]

        def _status(step: str, detail: str = "", task_id: str = ""):
            steps.append(f"[status] {step}: {detail}")
            self._emit_task_status(status_callback, step, detail, task_id)

        def _worker_done(report):
            if worker_done_callback:
                try:
                    worker_done_callback(report)
                except Exception:
                    pass

        # M2: 每次请求创建独立黑板
        from app.agents.blackboard import Blackboard

        self.blackboard = Blackboard()
        steps.append("[board] 黑板已创建")

        try:
            # 1. 拆解
            _status("decompose", "正在拆解任务...")
            briefs, exec_mode, final_inst = self._decompose(query, history)
            steps.append(f"拆解为 {len(briefs)} 个子任务，模式: {exec_mode}")
            _status("decompose_done", f"拆解为 {len(briefs)} 个子任务")

            if not briefs:
                # 拆解器判定为单一意图 → 多智能体大材小用。
                # 返回退化信号，由调用方回退单 Agent 快速路径（意图识别+工具分支），
                # 而不是塞给默认 rag worker 去检索知识库（会导致查天气捞民法典这类错配）。
                steps.append("单一意图，退出多智能体，回退快速路径")
                _status("degenerate", "单一意图，走快速路径")
                return {
                    "query": query,
                    "final_answer": "",
                    "intent": "degenerate",
                    "degenerate_to_single": True,
                    "is_fallback": False,
                    "steps": steps,
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                }

            authorised_ids = list(knowledge_base_ids or [])
            authorised_catalog = list(knowledge_catalog or [])
            for brief in briefs:
                brief.knowledge_base_ids = authorised_ids
                brief.knowledge_catalog = authorised_catalog

            # 拆解完成 → 通知前端渲染侧边任务进度面板的待办清单
            if tasks_callback:
                try:
                    tasks_callback([
                        {"task_id": b.task_id, "goal": b.goal, "worker_hint": b.worker_hint}
                        for b in briefs
                    ])
                except Exception:
                    pass

            # 2. 派发
            _status("dispatch", f"正在派发 {len(briefs)} 个子任务...")
            reports = self._dispatch(briefs, exec_mode, steps, _worker_done, _status)
            _status("dispatch_done", f"派发完成，{sum(1 for r in reports if r.ok())} 成功")

            # 3. 汇总
            _status("synthesize", "正在汇总结果...")
            if return_synthesize_payload:
                # 调用方（chat_router）负责流式汇总，这里只打包 payload；
                # 收尾状态 synthesize_done 由调用方在流式完成后补发，
                # 避免"汇总完成"先于汇总内容到达前端。
                synthesize_payload = {
                    "query": query,
                    "reports": reports,
                    "final_inst": final_inst,
                }
                final_answer = ""  # 由调用方流式生成
                steps.append("跳过内部汇总，交由调用方流式生成")
            else:
                final_answer = self._synthesize(query, reports, final_inst, steps)
                synthesize_payload = None
                _status("synthesize_done", "汇总完成")

            elapsed = time.perf_counter() - start
            steps.append(f"orchestrator 完成，耗时 {elapsed:.2f}s")

            # 合并黑板日志
            if self.blackboard:
                steps.extend(self.blackboard.render_log())

            return {
                "query": query,
                "final_answer": final_answer,
                "synthesize_payload": synthesize_payload,
                "intent": "multi_agent",
                "intent_confidence": 1.0,
                "retrieval_triggered": any(r.worker_name == "rag" for r in reports),
                "retrieved_docs_count": sum(
                    r.artifacts.get("retrieved_count", 0) for r in reports
                ),
                "tool_triggered": False,
                "sub_tasks": [b.goal for b in briefs],
                "task_details": [
                    {
                        "task_id": b.task_id,
                        "goal": b.goal,
                        "worker_hint": b.worker_hint,
                    }
                    for b in briefs
                ],
                "worker_reports": [
                    {
                        "task_id": report.task_id,
                        "worker_name": report.worker_name,
                        "status": report.status,
                        "summary": report.detail or report.summary,
                        "error": report.error or "",
                    }
                    for report in reports
                ],
                "steps": steps,
                "validation_passed": True,
                "validation_feedback": "",
                "is_fallback": False,
                "sources": self._collect_sources(reports),
                "elapsed_seconds": round(elapsed, 3),
                "execution_mode": exec_mode,
                "blackboard": self.blackboard.all_artifacts() if self.blackboard else [],
            }

        except Exception as exc:
            logger.error("[orchestrator] fatal error: %s", exc)
            steps.append(f"FATAL: {exc}")
            return {
                "query": query,
                "final_answer": f"多智能体编排失败: {exc}",
                "intent": "multi_agent",
                "is_fallback": True,
                "error_message": str(exc),
                "steps": steps,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
        finally:
            self.blackboard = None

    # ── 拆解 ─────────────────────────────────────────────────────────────────
    def _decompose(
        self, query: str, history: Optional[List[Dict[str, str]]]
    ) -> tuple[List[TaskBrief], str, str]:
        """LLM 拆解查询为 TaskBrief 列表。"""
        from app.skills.context import get_active_skill_prompt

        skill_prompt = get_active_skill_prompt()
        system_prompt = "你是一个任务拆解专家，输出严格 JSON。"
        if skill_prompt:
            system_prompt += "\n\n" + skill_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._DECOMPOSE_PROMPT.format(query=query)},
        ]

        try:
            # 用 chat_json_sync（自动剥 fence）
            # DeepSeek 等 reasoning 模型 max_tokens 过低会把额度耗在思考上导致空响应
            result = self.llm.chat_json_sync(messages, temperature=0.1, max_tokens=2048)
        except Exception as exc:
            logger.warning("[orchestrator] decompose failed, fallback single: %s", exc)
            return [], "sequential", ""

        # 拆解器判定为单一意图 → 不拆解，返回空 briefs，由调用方回退快速路径
        if not result.get("needs_decomposition", False):
            return [], "sequential", result.get("final_instruction", "")

        # 防御：拆解器说 needs_decomposition=true 但一个子任务都没给 → 视为未拆解
        raw_tasks = [t for t in result.get("sub_tasks", []) if (t.get("goal") or "").strip()]
        if not raw_tasks:
            logger.info("[orchestrator] decompose returned no usable sub-tasks, treat as single intent")
            return [], "sequential", result.get("final_instruction", "")

        briefs = []
        for t in raw_tasks[:4]:  # 限制最多 4 个
            briefs.append(
                TaskBrief(
                    task_id=t.get("task_id", f"task-{len(briefs)+1}"),
                    goal=t.get("goal", ""),
                    context=t.get("context", ""),
                    constraints=t.get("constraints", []),
                    worker_hint=t.get("worker_hint", self._default_worker),
                )
            )

        exec_mode = result.get("execution_mode", "sequential")
        final_inst = result.get("final_instruction", "")
        return briefs, exec_mode, final_inst

    # ── 派发 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _emit_task_status(status_cb, step: str, detail: str, task_id: str) -> None:
        if not status_cb:
            return
        try:
            status_cb(step, detail, task_id)
        except TypeError:
            try:
                status_cb(step, detail)
            except Exception:
                pass
        except Exception:
            pass

    def _attach_tool_callback(self, worker, status_cb, task_id: str) -> None:
        """给 worker 注入工具调用钩子 → 桥接到 status_callback（前端侧边面板展示）。"""

        def _on_tool(tool_name: str, args: dict):
            self._emit_task_status(
                status_cb,
                "tool_call",
                f"{tool_name}({args or {}})",
                task_id,
            )

        worker.tool_callback = _on_tool

    def _dispatch(
        self, briefs: List[TaskBrief], exec_mode: str, steps: List[str], on_worker_done=None, status_cb=None
    ) -> List[WorkerReport]:
        """按模式派发任务到 Worker。on_worker_done: 每个 Worker 完成时回调。"""
        if exec_mode == "parallel":
            return self._dispatch_parallel(briefs, steps, on_worker_done, status_cb)
        return self._dispatch_sequential(briefs, steps, on_worker_done, status_cb)

    def _dispatch_sequential(
        self, briefs: List[TaskBrief], steps: List[str], on_worker_done=None, status_cb=None
    ) -> List[WorkerReport]:
        """顺序派发，前序产出注入后续 brief.context。"""
        reports: List[WorkerReport] = []
        artifact_store: Dict[str, WorkerReport] = {}

        for brief in briefs:
            # M2: 解析 task-N 引用
            resolved_ctx = self._resolve_refs(brief.context, artifact_store)
            if resolved_ctx != brief.context:
                brief.context = resolved_ctx
                steps.append(f"{brief.task_id} 引用已解析")

            worker = self._get_worker(brief.worker_hint)
            steps.append(f"{brief.task_id} -> {worker.name}")
            if status_cb:
                self._emit_task_status(
                    status_cb,
                    "task_started",
                    f"{worker.name} 开始执行",
                    brief.task_id,
                )
                self._attach_tool_callback(worker, status_cb, brief.task_id)

            # 注入黑板
            worker.blackboard = self.blackboard
            try:
                report = worker.run_with_board(brief)
            finally:
                worker.blackboard = None
                worker.tool_callback = None

            reports.append(report)
            artifact_store[brief.task_id] = report
            steps.append(f"{brief.task_id} <- {report.status} ({len(report.summary)} chars)")
            if on_worker_done:
                try:
                    on_worker_done(report)
                except Exception:
                    pass

        return reports

    def _dispatch_parallel(
        self, briefs: List[TaskBrief], steps: List[str], on_worker_done=None, status_cb=None
    ) -> List[WorkerReport]:
        """并行派发（ThreadPoolExecutor 真并发）。"""
        reports: List[WorkerReport] = []

        from app.llm.client import get_active_chat_model_profile, use_chat_model
        from app.skills.context import get_active_skill_context, use_skill_context

        selected_model = get_active_chat_model_profile()
        selected_skills = get_active_skill_context()

        def _run_one(brief: TaskBrief) -> WorkerReport:
            # contextvars do not flow into ThreadPoolExecutor workers. Re-enter
            # the request's selection so every sub-agent uses the same model.
            with use_chat_model(selected_model), use_skill_context(selected_skills):
                worker = self._get_worker(brief.worker_hint)
                worker.blackboard = self.blackboard
                if status_cb:
                    self._emit_task_status(
                        status_cb,
                        "task_started",
                        f"{worker.name} 开始执行",
                        brief.task_id,
                    )
                    self._attach_tool_callback(worker, status_cb, brief.task_id)
                try:
                    return worker.run_with_board(brief)
                finally:
                    worker.blackboard = None
                    worker.tool_callback = None

        with ThreadPoolExecutor(max_workers=len(briefs)) as pool:
            futures = {pool.submit(_run_one, b): b for b in briefs}
            for future in as_completed(futures):
                brief = futures[future]
                try:
                    report = future.result()
                    reports.append(report)
                    steps.append(
                        f"{brief.task_id} <- {report.status} ({len(report.summary)} chars)"
                    )
                    if on_worker_done:
                        try:
                            on_worker_done(report)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.error("[orchestrator] parallel task %s failed: %s", brief.task_id, exc)
                    err_report = WorkerReport(
                        task_id=brief.task_id,
                        worker_name=brief.worker_hint,
                        status="error",
                        error=str(exc),
                        steps=[f"parallel ERROR: {exc}"],
                    )
                    reports.append(err_report)
                    steps.append(f"{brief.task_id} <- error ({exc})")
                    if on_worker_done:
                        try:
                            on_worker_done(err_report)
                        except Exception:
                            pass

        # 按原任务序恢复
        order = {b.task_id: i for i, b in enumerate(briefs)}
        reports.sort(key=lambda r: order.get(r.task_id, 999))
        return reports

    def _get_worker(self, hint: str) -> BaseWorker:
        """按 hint 路由 Worker，未知回退默认。"""
        return self._workers.get(hint, self._workers[self._default_worker])

    def _resolve_refs(
        self, context: str, artifact_store: Dict[str, WorkerReport]
    ) -> str:
        """解析 context 中的 task-N 引用，替换为前序产出摘要。"""
        import re

        def _replace(match):
            task_id = match.group(0)
            report = artifact_store.get(task_id)
            if report:
                return f"[引用 {task_id} 的产出：{report.summary[:300]}]"
            return match.group(0)

        return re.sub(r"task-\d+", _replace, context)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    def _synthesize(
        self,
        query: str,
        reports: List[WorkerReport],
        final_inst: str,
        steps: List[str],
    ) -> str:
        """汇总各 Worker 产出为最终回答。"""
        # 单任务成功直接返回，不二次加工
        if len(reports) == 1 and reports[0].ok():
            steps.append("单任务成功，直接返回")
            return reports[0].detail or reports[0].summary

        # 多任务：LLM 整合
        combined = "\n\n".join(
            f"## {r.task_id} ({r.worker_name})\n{r.detail or r.summary}"
            for r in reports
            if r.ok()
        )

        if not combined:
            return "所有子任务执行失败，无法生成回答。"

        prompt = (
            f"用户原始查询：{query}\n\n"
            f"各子任务产出：\n{combined}\n\n"
            f"汇总要求：{final_inst or '综合各子任务结果，给出完整、连贯的回答。'}"
        )
        from app.skills.context import get_active_skill_prompt

        skill_prompt = get_active_skill_prompt()
        if skill_prompt:
            prompt = skill_prompt + "\n\n" + prompt

        try:
            answer = self.llm.chat_sync(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=16384,
            )
            steps.append(f"LLM 汇总完成 ({len(answer)} chars)")
            return answer
        except Exception as exc:
            logger.warning("[orchestrator] synthesize failed, fallback concat: %s", exc)
            steps.append("LLM 汇总失败，回退原始拼接")
            return combined

    def _collect_sources(self, reports: List[WorkerReport]) -> List[Dict[str, str]]:
        """从各 WorkerReport 收集引用来源。"""
        sources: List[Dict[str, str]] = []
        for r in reports:
            for s in r.artifacts.get("sources", []):
                if s and s not in sources:
                    sources.append({"title": s, "url": ""})
        return sources


# ── 单例 ─────────────────────────────────────────────────────────────────────
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
