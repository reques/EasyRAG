"""Run INSIDE the EasyRAG backend container: execute a Golden Set for real.

This is the end-to-end proof that offline chunk-id labels line up with the live
index: it calls the same ``run_evaluation`` the HTTP API uses, with real
retrieval against the configured vector store.

Usage (from the host):
  docker cp eval/cloudway24/cloudway24_golden_set.json easyrag-backend:/tmp/gs.json
  docker cp <kb_files.json>             easyrag-backend:/tmp/kb_files.json
  docker cp eval/run_golden_baseline.py easyrag-backend:/tmp/run.py
  docker exec -w /app easyrag-backend python /tmp/run.py \
      --golden /tmp/gs.json --kb-files /tmp/kb_files.json --kb-id <uuid> --top-k 6
"""
import argparse
import json
import sys

from backend.services.evaluation_service import EvaluationCase, run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--kb-files", default="", help="file_id -> filename map (for expected_source)")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out", default="/tmp/golden_baseline_run.json")
    args = parser.parse_args()

    golden = json.load(open(args.golden, encoding="utf-8"))
    files = {}
    if args.kb_files:
        files = {
            f["file_id"]: f["filename"]
            for f in json.load(open(args.kb_files, encoding="utf-8-sig"))
        }

    cases = [
        EvaluationCase(
            question=c["question"],
            expected_file_id=c["expected_file_id"],
            expected_chunk_ids=tuple(c.get("expected_chunk_ids") or []),
            reference_answer=c.get("reference_answer", ""),
            # Milvus does not persist file_id, so scoring recovers it from the
            # source filename; this mirrors what the HTTP import path builds.
            expected_source=files.get(c["expected_file_id"], ""),
            expect_miss=c.get("expect_miss", False),
        )
        for c in golden["cases"]
    ]
    print("loaded %d cases (%d positive, %d negative)" % (
        len(cases),
        sum(1 for c in cases if not c.expect_miss),
        sum(1 for c in cases if c.expect_miss),
    ))

    result = run_evaluation(cases, top_k=args.top_k, knowledge_base_id=args.kb_id, run_ragas=False)

    print("\n=== aggregate metrics (k=%d) ===" % args.top_k)
    for key in ("hit_rate_at_k", "mrr_at_k", "recall_at_k", "precision_at_k", "ndcg_at_k",
                "file_hit_rate_at_k", "file_recall_at_k", "avg_score"):
        print("  %-20s %s" % (key, result.get(key)))

    print("\n=== run metadata (reproducibility snapshot) ===")
    print(json.dumps(result.get("run_metadata"), ensure_ascii=False, indent=2))

    analysis = result.get("analysis") or {}
    print("\n=== failure analysis ===")
    for key in ("missed_count", "low_recall_count", "false_positive_count"):
        print("  %-22s %s" % (key, analysis.get(key)))

    modes = {}
    for d in result["details"]:
        modes[d["reference_mode"]] = modes.get(d["reference_mode"], 0) + 1
    print("\nreference modes:", modes)

    nonzero = sum(1 for d in result["details"]
                  if not d["expect_miss"] and d["chunk_metrics"]["hit_rate_at_k"] > 0)
    total_pos = sum(1 for d in result["details"] if not d["expect_miss"])
    fp = sum(1 for d in result["details"] if d["expect_miss"] and d["false_positive"])
    neg = sum(1 for d in result["details"] if d["expect_miss"])
    print("positives with a chunk-level hit: %d/%d" % (nonzero, total_pos))
    print("negatives that falsely hit their target file: %d/%d" % (fp, neg))

    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, default=str)
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
