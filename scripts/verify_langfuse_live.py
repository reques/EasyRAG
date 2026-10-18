"""Exercise real EasyRAG model/tool/delegation I/O and verify indexed Langfuse data.

Requires a running candidate backend and local_langfuse.py init/up/connect.
Creates one synthetic EasyRAG account and retains it for inspection. Never logs
credentials, prompts, responses, or model keys. Does not promote users to admin.
Run explicitly: python scripts/verify_langfuse_live.py --live --backend http://localhost:8000
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import secrets
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_langfuse import ENV, read_env


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read_events(response):
    response.raise_for_status()
    for line in response.iter_lines():
        if line.startswith("data:"):
            yield json.loads(line[5:])


def consume(response):
    last = None
    for event in read_events(response):
        if event.get("type") in {"done", "error", "cancelled"}:
            last = event["type"]
    check(last == "done", "Real agent execution did not complete successfully")


def observations(client, user_id, session_id):
    results, cursor = [], None
    while True:
        page = client.api.observations.get_many(
            user_id=user_id, session_id=session_id, cursor=cursor, limit=100,
            fields="core,basic,time,metadata,model,usage,io,metrics",
        )
        results.extend(page.data)
        cursor = page.meta.cursor
        if not cursor:
            return results
        check(len(results) < 2000, "Unexpected observation count in synthetic session")


def execution_nodes(client, user_id, session_id, execution_id, *, cancelled=False, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = observations(client, user_id, session_id)
        nodes = [row for row in rows if (row.metadata or {}).get("executionId") == execution_id]
        roots = [row for row in nodes if row.name == "execution.chat"]
        if roots and roots[0].end_time and (cancelled or
                (any(row.type == "GENERATION" for row in nodes) and
                 any(row.type == "TOOL" for row in nodes) and
                 any((row.name or "").startswith(("subagent.", "spawn_task.")) for row in nodes))):
            # Let the worker index late child batches, then reread everything.
            time.sleep(3)
            return [row for row in observations(client, user_id, session_id)
                    if (row.metadata or {}).get("executionId") == execution_id]
        time.sleep(2)
    raise AssertionError("Langfuse did not index the required execution tree before timeout")


def verify_tree(nodes, user_id, session_id, execution_id, *, cancelled=False):
    check(bool(nodes), "Execution is absent from Langfuse")
    roots = [row for row in nodes if row.is_root_observation or not row.parent_observation_id]
    check(len(roots) == 1 and roots[0].name == "execution.chat", "Expected exactly one execution root")
    check(len({row.trace_id for row in nodes}) == 1, "Model/tool/subagent split across traces")
    by_id = {row.id: row for row in nodes}
    call_ids = [(row.metadata or {}).get("modelCallId") for row in nodes if row.type == "GENERATION"]
    reported = [value for value in call_ids if value]
    check(len(reported) == len(set(reported)), "A model invocation was recorded more than once")
    root = roots[0]
    check(root.metadata.get("status") == ("cancelled" if cancelled else "completed"), "Root status differs from business status")
    for row in nodes:
        check(row.user_id == user_id and row.session_id == session_id, "Incorrect user/session attribution")
        check(row.metadata.get("executionId") == execution_id, "Incorrect execution attribution")
        check(row.end_time is not None and row.end_time >= row.start_time, "Missing or invalid timing")
        check(row.input in (None, "", "null") and row.output in (None, "", "null"), "Content exported in metadata_only mode")
        seen, ancestor = set(), row
        while ancestor.id != root.id:
            check(ancestor.id not in seen, "Cycle in observation tree")
            seen.add(ancestor.id)
            check(ancestor.parent_observation_id in by_id, "Orphaned observation")
            ancestor = by_id[ancestor.parent_observation_id]
    if cancelled:
        check(any(row.type == "GENERATION" for row in nodes), "Cancellation happened before any model invocation")
        check(all(row.level != "ERROR" for row in nodes), "Cancellation was recorded as an error")
    else:
        generations = [row for row in nodes if row.type == "GENERATION"]
        tools = [row for row in nodes if row.type == "TOOL"]
        agents = [row for row in nodes if (row.name or "").startswith(("subagent.", "spawn_task."))]
        check(bool(generations) and all(row.model for row in generations), "Missing actual model identifiers")
        check(any(row.name == "tool.calculator" for row in tools), "Calculator did not execute")
        check(bool(agents), "No real subagent observation")
        def below(row, parent):
            while row.id != root.id:
                if row.parent_observation_id == parent.id:
                    return True
                row = by_id[row.parent_observation_id]
            return False
        check(any(below(g, a) for a in agents for g in generations), "Subagent model has wrong parent")
        check(any(below(t, a) for a in agents for t in tools if t.name == "tool.calculator"), "Subagent tool has wrong parent")
        check(all(row.level != "ERROR" for row in nodes), "Successful case contains an unexpected error")
    return {"execution_id": execution_id, "trace_id": root.trace_id, "observations": len(nodes),
            "types": dict(Counter(row.type for row in nodes)), "roots": 1,
            "status": root.metadata.get("status"), "parent_links": "valid", "content": "filtered"}


def run(args):
    import httpx
    from langfuse import Langfuse

    config = read_env(ENV)
    base = args.backend.rstrip("/") + "/api/v1"
    lf_base = f"http://localhost:{config['LANGFUSE_WEB_PORT']}"
    with httpx.Client(base_url=base, timeout=httpx.Timeout(360, connect=10), trust_env=False) as api:
        # Refuse to create test data until the application and Langfuse are reachable.
        api.get("/health").raise_for_status()
        lf = Langfuse(public_key=config["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
                      secret_key=config["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
                      base_url=lf_base, tracing_enabled=False, timeout=10)
        try:
            check(lf.auth_check(), "Langfuse project authentication failed")
            account = {"username": "lfverify_" + uuid.uuid4().hex[:12], "password": secrets.token_urlsafe(24)}
            reply = api.post("/auth/register", json=account)
            reply.raise_for_status()
            identity = reply.json()
            api.headers["Authorization"] = "Bearer " + identity["access_token"]
            user_id = identity["user_id"]
            record = ENV.parent / "verify.env.local"
            with record.open("a", encoding="utf-8") as file:
                file.write(f"\n# Synthetic verifier user {user_id}\nUSERNAME={account['username']}\nPASSWORD={account['password']}\n")
            execution_id = str(uuid.uuid4())
            payload = {"request_id": execution_id, "deep_research": True, "skill_ids": ["data-analysis"],
                       "query": "这是合成链路验收任务：请先 read_skill 加载 data-analysis，然后调用 task 委派 coding-agent。"
                                "子任务必须实际调用 calculator 计算 1234567*89，再报告结果。主智能体请勿代替子智能体计算，"
                                "不要联网，不要读取文件，不要搜索知识库，最终简短汇报。"}
            with api.stream("POST", "/chat/stream", json=payload) as response:
                response.raise_for_status()
                conversation_id = response.headers["X-Conversation-Id"]
                check(response.headers["X-Chat-Stream-Id"] == execution_id, "Admission changed execution ID")
                first = next(read_events(response))
                cursor = first["sequence"]
            print("Synthetic execution admitted; disconnected SSE without cancelling.", flush=True)
            state = api.get(f"/chat/conversations/{conversation_id}/stream")
            state.raise_for_status()
            check(state.json()["stream"]["id"] == execution_id, "Refresh lookup changed execution")
            with api.stream("GET", f"/chat/streams/{execution_id}/events", params={"after": cursor}) as response:
                consume(response)
            # Identical retry must replay the same execution without another root.
            with api.stream("POST", "/chat/stream", json=payload) as response:
                check(response.headers["X-Chat-Stream-Id"] == execution_id, "Idempotent retry created a new execution")
                consume(response)
            check(api.get(f"/chat/streams/{execution_id}/observability").status_code == 403,
                  "Ordinary user can access administrator navigation")
            print("Real model/tool/delegation execution completed; waiting for Langfuse indexing.", flush=True)
            nodes = execution_nodes(lf, user_id, conversation_id, execution_id)
            report = verify_tree(nodes, user_id, conversation_id, execution_id)
            report["url"] = f"{lf_base}/project/{config['LANGFUSE_INIT_PROJECT_ID']}/traces/{report['trace_id']}"
            report["reconnection"] = report["refresh_lookup"] = report["idempotent_replay"] = "same_execution"
            print(f"Indexed tree verified: {report['observations']} observations, one root, unique model calls. Starting cancellation case.", flush=True)

            cancel_id = str(uuid.uuid4())
            with api.stream("POST", "/chat/stream", json={"request_id": cancel_id, "query": "讲一个很长的童话故事，至少三千字，不调用工具。"}) as response:
                response.raise_for_status()
                cancel_conv = response.headers["X-Conversation-Id"]
                # Stop during prefill/thinking as well as public text streaming.
                # Providers may buffer the entire answer, so waiting for the first
                # visible artifact can race an already committed result. Confirm
                # below that a real generation was present in this execution.
                for event in read_events(response):
                    check(event.get("type") not in {"done", "error", "cancelled"}, "Cancel case ended before model output")
                    if (event.get("type") == "agent_event" and
                            (event.get("event") or {}).get("type") == "agent_start"):
                        time.sleep(1)
                        break
                    if event.get("type") == "delta" or (
                            event.get("type") == "artifact" and event.get("streaming")
                            and event.get("kind") in {"answer", "thought"} and event.get("content")):
                        break
            started = time.monotonic()
            api.post(f"/chat/streams/{cancel_id}/cancel").raise_for_status()
            with api.stream("GET", f"/chat/streams/{cancel_id}/events") as response:
                terminal = [e["type"] for e in read_events(response) if e.get("type") in {"done", "error", "cancelled"}]
            check(terminal == ["cancelled"], "Explicit stop did not end as cancelled")
            elapsed = round(time.monotonic() - started, 3)
            cancel_nodes = execution_nodes(lf, user_id, cancel_conv, cancel_id, cancelled=True)
            cancel_report = verify_tree(cancel_nodes, user_id, cancel_conv, cancel_id, cancelled=True)
            cancel_report["stop_to_terminal_seconds"] = elapsed
            check(elapsed < 10, "Cancellation is unexpectedly slow")
            return {"passed": True, "execution": report, "cancellation": cancel_report,
                    "user_id": user_id, "synthetic_credentials_file": str(record)}
        finally:
            lf.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except AssertionError as exc:
        print(f"Live verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        # Exception text from API/SDKs may contain credentials or private content.
        print(f"Live verification failed ({type(exc).__name__}); no success report written.", file=sys.stderr)
        raise SystemExit(1)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
