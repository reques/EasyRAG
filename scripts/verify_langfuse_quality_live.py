"""Real local question/retrieval, feedback and evaluation acceptance checks.

Opt in with --live. Uses a retained synthetic account/KB and a real configured
model (--model-id), or creates an Ollama Qwen profile when no model is selected.
Credentials live only in the ignored work directory; reports contain no keys.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import secrets
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_langfuse import ENV, read_env


QUESTION = "根据 Orion 验收知识库，标准服务首次响应时限是多少小时？请先检索知识库再回答，只答时限。"
CANARY = "sk-synthetic-observability-canary"
DOCUMENT = (
    "Orion 服务规则（仅用于本地 Langfuse 验收）\n\n"
    "Orion 标准服务首次响应时限为 4 小时。升级服务首次响应时限为 1 小时。\n"
    "Orion 标准服务保留支持记录 30 天。升级服务保留支持记录 90 天。\n"
    "这些都是合成规则，不对应任何真实客户。\n"
    f"以下是假密钥，用于验证脱敏：{CANARY}\n"
)


def check(value, message):
    if not value:
        raise AssertionError(message)


class Verification:
    def __init__(self, args):
        import httpx
        from langfuse import Langfuse
        self.args = args
        self.folder = args.work_dir
        self.folder.mkdir(parents=True, exist_ok=True)
        self.state_path = self.folder / "quality-private.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.report = self.state.get("report", {})
        self.api = httpx.Client(base_url=args.backend.rstrip("/") + "/api/v1", timeout=360, trust_env=False)
        cfg = read_env(ENV)
        self.langfuse_url = f"http://localhost:{cfg['LANGFUSE_WEB_PORT']}"
        self.project_id = cfg["LANGFUSE_INIT_PROJECT_ID"]
        self.lf = Langfuse(public_key=cfg["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
                           secret_key=cfg["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
                           base_url=self.langfuse_url, tracing_enabled=False, timeout=10)

    def save(self):
        self.state["report"] = self.report
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.folder / "quality-report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")

    def request(self, method, path, **kwargs):
        reply = self.api.request(method, path, **kwargs)
        if reply.is_error:
            (self.folder / "last-api-error.txt").write_text(reply.text, encoding="utf-8")
            raise RuntimeError(f"Application {method} {path} returned {reply.status_code}")
        return reply.json()

    def login(self):
        self.api.get("/health").raise_for_status()
        check(self.lf.auth_check(), "Langfuse project credentials did not authenticate")
        if "username" not in self.state:
            self.state.update(username="lfquality_" + uuid.uuid4().hex[:10], password=secrets.token_urlsafe(24))
            self.save()
            identity = self.request("POST", "/auth/register", json={
                "username": self.state["username"], "password": self.state["password"]})
        else:
            identity = self.request("POST", "/auth/login", json={
                "username": self.state["username"], "password": self.state["password"]})
        self.api.headers["Authorization"] = "Bearer " + identity["access_token"]
        self.state["user_id"] = identity["user_id"]
        self.report["user_id"] = identity["user_id"]
        self.save()

    def wait(self, title, predicate, timeout=240):
        deadline = time.monotonic() + timeout
        last_notice = 0
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                print(title + " verified", flush=True)
                return result
            if time.monotonic() - last_notice > 25:
                print("Waiting for " + title, flush=True)
                last_notice = time.monotonic()
            time.sleep(3)
        raise AssertionError(title + " was not visible before timeout")

    def prepare(self):
        if self.args.model_id:
            check(not self.state.get("model_id") or self.state["model_id"] == self.args.model_id,
                  "Use a separate --work-dir to verify a different model")
            self.state["model_id"] = self.args.model_id
            self.save()
        if "model_id" not in self.state:
            model = self.request("POST", "/chat/models", json={
                "name": "Langfuse 验收 Qwen3.5", "provider_name": "Ollama",
                "provider_type": "local", "base_url": "http://host.docker.internal:11434/v1",
                "model_name": "qwen3.5:latest", "requires_api_key": False, "temperature": 0,
            })
            self.state["model_id"] = model["id"]
            self.save()
        if "kb_id" not in self.state:
            kb = self.request("POST", "/knowledge/bases", json={"name": "Langfuse Orion 验收 20261101"})
            self.state["kb_id"] = kb["id"]
            self.save()
        if "file_id" not in self.state:
            uploaded = self.request("POST", f"/knowledge/bases/{self.state['kb_id']}/upload",
                files={"file": ("orion-verification.txt", DOCUMENT.encode(), "text/plain")},
                data={"parser": "local", "strategy": "recursive"})
            self.state["file_id"] = uploaded["file_id"]
            self.save()
        def indexed():
            rows = self.request("GET", f"/knowledge/bases/{self.state['kb_id']}/files")
            row = next(row for row in rows if row["id"] == self.state["file_id"])
            if row["status"] == "failed":
                raise RuntimeError("Synthetic file indexing failed")
            return row if row["status"] in {"completed", "indexed"} else None
        row = self.wait("synthetic knowledge file indexing", indexed)
        retrieved = self.request("POST", f"/knowledge/bases/{self.state['kb_id']}/retrieval/test",
                                json={"query": QUESTION, "top_k": 3, "score_threshold": 0})
        check(retrieved["results"], "Real vector retrieval returned no chunks")
        # The legacy Milvus schema retains KB/source/content but no file UUID.
        # This query is already scoped to the synthetic user's single KB/file.
        self.state["chunk_ids"] = [item["chunk_id"] for item in retrieved["results"]
            if item["file_id"] == self.state["file_id"] or
               (item["file_id"] is None and item["source"] == "orion-verification.txt")]
        check(self.state["chunk_ids"], "Retrieval did not return the synthetic file")
        self.report["knowledge"] = {"kb_id": self.state["kb_id"], "file_id": self.state["file_id"],
                                    "chunks": row["chunk_count"], "retrieval_hits": len(self.state["chunk_ids"])}
        self.save()

    def chat(self):
        if "message_id" not in self.state:
            self.state.setdefault("request_id", str(uuid.uuid4()))
            self.save()
            with self.api.stream("POST", "/chat/stream", json={"query": QUESTION,
                    "model_id": self.state["model_id"], "request_id": self.state["request_id"]}) as response:
                response.raise_for_status()
                terminal = None
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    if event.get("conversation_id"):
                        self.state["conversation_id"] = event["conversation_id"]
                    if event.get("stream_id"):
                        self.state["stream_id"] = event["stream_id"]
                    if event.get("type") in {"done", "error", "cancelled"}:
                        terminal = event
                        self.state["terminal"] = event
                        self.save()
            check(terminal and terminal["type"] == "done", "Real question/answer did not complete")
            check(terminal.get("message_id"), "Durable done event did not contain saved message ID")
            self.state["message_id"] = terminal["message_id"]
            self.report["question_answer"] = {"execution_id": self.state["stream_id"],
                "conversation_id": self.state["conversation_id"], "message_id": terminal["message_id"],
                "model_id": self.state["model_id"], "answer": terminal.get("content", "")}
            self.save()
        check("4" in self.state["terminal"].get("content", "") or "四" in self.state["terminal"].get("content", ""),
              "Answer did not include the expected four-hour policy")

    def observations(self, **filters):
        rows, cursor = [], None
        while True:
            page = self.lf.api.observations.get_many(limit=100, cursor=cursor,
                        fields="core,basic,time,metadata,model,usage,io,metrics", **filters)
            rows.extend(page.data)
            cursor = page.meta.cursor
            if not cursor:
                return rows

    def trace(self):
        def visible():
            rows = self.observations(user_id=self.state["user_id"], session_id=self.state["conversation_id"])
            rows = [row for row in rows if (row.metadata or {}).get("executionId") == self.state["stream_id"]]
            roots = [row for row in rows if row.name == "execution.chat" and row.end_time]
            retrievals = [row for row in rows if row.type == "RETRIEVER"]
            if not roots or not retrievals or not any(row.type == "GENERATION" for row in rows):
                return None
            check(len(roots) == 1, "Expected exactly one question/answer root")
            root = roots[0]
            check(root.input and root.output, "Root question/answer content was not indexed in redacted mode")
            check((root.input if isinstance(root.input, dict) else json.loads(root.input)).get("contexts"),
                  "Root has no selected retrieval context")
            serialized = json.dumps([row.model_dump(mode="json") for row in rows], ensure_ascii=False)
            check(CANARY not in serialized, "Synthetic secret canary escaped redaction")
            check("[REDACTED]" in serialized, "No redacted marker was present in synthetic evidence")
            check(len({row.trace_id for row in rows}) == 1, "Question/retrieval/model split into multiple traces")
            self.state["trace_id"] = root.trace_id
            self.report["trace"] = {"trace_id": root.trace_id, "observations": len(rows),
                "retrieval_nodes": len(retrievals), "generation_nodes": sum(row.type == "GENERATION" for row in rows),
                "question_answer_content": "visible", "selected_context": "visible", "canary_redacted": True,
                "url": f"{self.langfuse_url}/project/{self.project_id}/traces/{root.trace_id}"}
            self.save()
            return root
        return self.wait("indexed question/answer Trace", visible)

    def feedback(self):
        reply = self.request("PUT", f"/chat/messages/{self.state['message_id']}/feedback",
                             json={"rating": "dislike", "reason": "合成验收：请补充条款来源，" + CANARY})
        check(reply["rating"] == "dislike", "Local user feedback did not persist")
        def indexed():
            rows = self.lf.api.scores.get_many(trace_id=self.state["trace_id"], name="user-feedback", fields="details,subject").data
            if not rows:
                return None
            check(len(rows) == 1, "User feedback duplicated the score")
            score = rows[0]
            if score.value != 0:
                return None
            if score.environment != "local":
                return None
            comment = getattr(score, "comment", "") or ""
            check(CANARY not in comment and "[REDACTED]" in comment, "Feedback reason was not correctly redacted")
            check(score.subject.kind == "trace" and score.subject.id == self.state["trace_id"], "Feedback is linked to the wrong subject")
            self.report["feedback"] = {"score_id": score.id, "trace_id": score.subject.id,
                                        "name": score.name, "value": score.value, "environment": score.environment,
                                        "reason_redacted": True}
            self.save()
            return score
        self.wait("indexed user-feedback Score", indexed)

    def evaluation(self):
        if "dataset_id" not in self.state:
            cases = [{"question": QUESTION, "expected_file_id": self.state["file_id"],
                      "expected_chunk_ids": self.state["chunk_ids"], "reference_answer": "4 小时。"},
                     {"question": "Orion 标准服务的支持记录保留多少天？", "expected_file_id": self.state["file_id"],
                      "expected_chunk_ids": self.state["chunk_ids"], "reference_answer": "30 天。"}]
            dataset = self.request("POST", "/evaluation/datasets", json={"name": "Orion Langfuse acceptance",
                    "kb_id": self.state["kb_id"], "cases": cases})
            self.state["dataset_id"] = dataset["id"]
            self.save()
        if "benchmark_id" not in self.state:
            benchmark = self.request("POST", "/evaluation/benchmarks", json={
                "name": "Orion retrieval acceptance", "knowledge_base_id": self.state["kb_id"],
                "dataset_id": self.state["dataset_id"], "retrieval_config": {"top_k": 3, "score_threshold": 0, "mode": "basic"}})
            self.state["benchmark_id"] = benchmark["id"]
            self.save()
        if "evaluation" not in self.state:
            result = self.request("POST", f"/evaluation/benchmarks/{self.state['benchmark_id']}/runs",
                                  json={"name": "Orion local acceptance 20261101"})
            self.state["evaluation"] = result
            self.save()
        result = self.state["evaluation"]
        linked = result["metrics"]["observability"]
        check(linked["linked_items"] == 2, "Evaluation did not link both actual cases")
        check(result["recall_at_k"] == 1, "Synthetic retrieval evaluation did not match its golden chunks")
        start = datetime.now(timezone.utc) - timedelta(days=1)
        def visible():
            experiments = self.lf.api.experiments.list(from_start_time=start, id=linked["experiment_id"], fields="core,metadata,scores").data
            items = self.lf.api.experiments.list_items(from_start_time=start, experiment_id=linked["experiment_id"], fields="core,io,scores").data
            if not experiments or len(items) != 2:
                return None
            expected = {row["observability"]["observation_id"] for row in result["details"]}
            # V3's experiment_id score filter addresses run-level scores.
            # Item-level scores are attached to their observation subjects and
            # are returned by the experiment-item API itself.
            scores = [score for item in items for score in (item.scores or [])]
            if len(scores) < 10:
                return None
            check(all(score.environment == "experiment" for score in scores), "Evaluation scores use a different environment from their roots")
            check({score.subject.id for score in scores if score.subject.kind == "observation"} == expected,
                  "Evaluation scores were not attached to the actual item roots")
            check(len({row["observability"]["trace_id"] for row in result["details"]}) == 2, "Dataset cases shared a trace")
            self.report["experiment"] = {"experiment_id": linked["experiment_id"], "items": len(items),
                "scores": len(scores), "recall_at_k": result["recall_at_k"], "mrr_at_k": result["mrr_at_k"],
                "url": linked["url"], "local_run_id": result["id"], "root_score_links": "valid"}
            self.save()
            return experiments[0]
        self.wait("indexed Experiment and item Scores", visible)

    def run(self):
        self.login()
        phases = {"prepare": self.prepare, "chat": self.chat, "trace": self.trace,
                  "feedback": self.feedback, "evaluation": self.evaluation}
        for name, operation in phases.items():
            if self.args.phase in {"all", name}:
                print("Running " + name, flush=True)
                operation()
        self.report["checked_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        print("Report saved: " + str(self.folder / "quality-report.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--model-id", help="Existing EasyRAG model profile ID; default creates an Ollama Qwen profile")
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/langfuse-quality-live"))
    parser.add_argument("--phase", choices=["all", "prepare", "chat", "trace", "feedback", "evaluation"], default="all")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required to create synthetic test data and run a real model")
    verification = Verification(args)
    try:
        verification.run()
    finally:
        verification.api.close()
        verification.lf.shutdown()


if __name__ == "__main__":
    main()
