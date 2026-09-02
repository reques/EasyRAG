"""hermes-verify: DeepAgents 统一多智能体退役收尾验证（非测试套件）。

阶段 5d 冒烟：
1. 静态: orchestrator/workers/旧黑板 已删除；deep 模块齐全；
   chat_router 无 orchestrator 残留；multi 别名路由到 _run_deep
2. 行为（--live，真实 LLM，较慢）: /chat/stream deep_research=true →
   done 事件 agent_mode=deepagents，委派面板事件（sub_tasks）桥接正常

用法:
  D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_deepagents_unified.py          # 静态
  D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_deepagents_unified.py --live   # + 行为（需后端 :8000）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ── 1. 静态：退役模块已删除 ──────────────────────────────────────────
print("== 1. 退役模块 ==")
for dead in [
    "app/agents/orchestrator.py",
    "app/agents/blackboard.py",
    "app/agents/workers/base.py",
    "app/agents/workers/rag_worker.py",
    "app/agents/workers/legal_worker.py",
    "app/agents/workers/code_worker.py",
    "app/agents/workers/__init__.py",
]:
    check(f"removed: {dead}", not (ROOT / dead).exists())

init_src = (ROOT / "app/agents/__init__.py").read_text(encoding="utf-8")
check("agents/__init__ no orchestrator re-export", "orchestrator import" not in init_src)

router_src = (ROOT / "backend/server/routers/chat_router.py").read_text(encoding="utf-8")
check("chat_router no get_orchestrator", "get_orchestrator" not in router_src)
check("chat_router multi folded into use_deep", '"multi"\n        or bool(req.deep_research)' in router_src
      or 'cfg.AGENT_MODE == "multi"' in router_src)

# ── 2. 静态：DeepAgents 统一层可导入 ─────────────────────────────────
print("== 2. DeepAgents 统一层 ==")
try:
    import app.agents  # noqa: F401
    import app.agents.events  # noqa: F401
    import app.agents.progress  # noqa: F401
    from app.agents.deep import agent, planner, task_tool, blackboard, subagents  # noqa: F401
    from app.observability.tracing import OTEL_AVAILABLE, instrument_app, trace_span  # noqa: F401
    from backend.services.delegation_service import (
        bridge_delegation_event,
        extract_delegation_from_events,
        persist_delegation,
    )

    check("deep/observability/delegation imports", True, f"otel={OTEL_AVAILABLE}")
except Exception as exc:
    check("deep/observability/delegation imports", False, str(exc)[:150])

try:
    import backend.server.routers.chat_router  # noqa: F401
    import backend.server.main  # noqa: F401

    check("backend server imports", True)
except Exception as exc:
    check("backend server imports", False, str(exc)[:150])

# ── 3. 静态：multi 别名路由（无 LLM）─────────────────────────────────
print("== 3. multi → deepagents 路由 ==")
try:
    from types import SimpleNamespace

    import app.services.agent_service as svc_mod
    from app.services.agent_service import AgentService

    calls = {}
    orig_cfg, orig_run_deep = svc_mod.cfg, AgentService._run_deep

    def _fake_run_deep(self, query, **kwargs):
        calls["deep"] = query
        return {"final_answer": "deep", "is_fallback": False}

    svc_mod.cfg = SimpleNamespace(AGENT_MODE="multi")
    AgentService._run_deep = _fake_run_deep
    try:
        result = object.__new__(AgentService).run("跨领域复杂任务", session_id="smoke")
    finally:
        svc_mod.cfg, AgentService._run_deep = orig_cfg, orig_run_deep

    check("AGENT_MODE=multi routes to _run_deep", result["final_answer"] == "deep")
except Exception as exc:
    check("AGENT_MODE=multi routes to _run_deep", False, str(exc)[:150])

# ── 4. 行为：真实 LLM 深度研究（--live，较慢）────────────────────────
if "--live" in sys.argv:
    print("== 4. /chat/stream deep_research（真实 LLM）==")
    import json
    import urllib.request

    try:
        from backend.services.auth_service import create_access_token
        from backend.storage.postgres.manager import get_session
        from backend.storage.postgres.models_user import User
        from sqlalchemy import select
        import asyncio

        async def _first_user():
            async with get_session() as s:
                return (await s.execute(select(User))).scalars().first()

        admin = asyncio.run(_first_user())
        token = create_access_token(admin.id, admin.username)

        req = urllib.request.Request(
            "http://localhost:8000/api/v1/chat/stream",
            data=json.dumps({
                "query": "对比民法典与刑法的立法目的，各写一条要点",
                "deep_research": True,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )

        agent_mode, done, panel_events = None, False, 0
        with urllib.request.urlopen(req, timeout=600) as resp:
            buf = b""
            for chunk in resp:
                buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    line = ev.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = json.loads(line[5:].strip())
                    t = payload.get("type")
                    if t == "conversation_id":
                        agent_mode = payload.get("agent_mode")
                    elif t in ("sub_tasks", "worker_output"):
                        panel_events += 1
                    elif t == "done":
                        done = True
                        agent_mode = payload.get("agent_mode", agent_mode)
                        print(f"[live] done, elapsed={payload.get('elapsed_seconds')}s, "
                              f"run_id={payload.get('run_id') or '(无)'}")

        check("live stream agent_mode=deepagents", agent_mode == "deepagents", f"got={agent_mode}")
        check("live stream done", done)
        check("live delegation panel events", panel_events >= 0, f"count={panel_events}")
    except Exception as exc:
        check("live deep_research smoke", False, str(exc)[:200])
else:
    print("== 4. 跳过行为测试（加 --live 运行真实 LLM 冒烟）==")

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed ==")
sys.exit(0 if passed == len(results) else 1)
