"""Host-side structural validation of a generated Golden Set.

Checks the file against the EasyRAG evaluation data contract *without* needing a
running backend, then optionally verifies that every referenced chunk ID really
exists in an exported index dump (``golden_chunks.json`` from golden_prepare.py).

Usage:
  python validate_golden_set.py --golden eval/cloudway24/cloudway24_golden_set.json \
      --review eval/cloudway24/cloudway24_qa_review.csv \
      --chunks eval/_work/golden_chunks.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

CONTRACT_KEYS = {
    "question", "expected_file_id", "expected_chunk_ids", "reference_answer",
    "expect_miss", "expected_filename",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--review", default="")
    parser.add_argument("--chunks", default="", help="index dump for chunk-id integrity check")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    cases = golden["cases"]
    pos = [c for c in cases if not c.get("expect_miss")]
    neg = [c for c in cases if c.get("expect_miss")]

    print("file      :", args.golden)
    print("name      :", golden.get("name"))
    print("desc len  :", len(golden.get("description", "")), "(limit 512)")
    print("cases     :", len(cases), "(limit 1000) ->", len(pos), "positive /", len(neg), "negative")

    checks = {
        "top-level keys are name/description/cases":
            set(golden) == {"name", "description", "cases"},
        "description within 512 chars": len(golden.get("description", "")) <= 512,
        "case count within 1..1000": 1 <= len(cases) <= 1000,
        "every case carries only contract keys":
            all(set(c) <= CONTRACT_KEYS for c in cases),
        "every question non-empty and <=4096":
            all(c["question"].strip() and len(c["question"]) <= 4096 for c in cases),
        "every case has expected_file_id or expected_filename":
            all(c.get("expected_file_id") or c.get("expected_filename") for c in cases),
        "every case has expected_chunk_ids <=32":
            all(len(c.get("expected_chunk_ids") or []) <= 32 for c in cases),
        "no duplicate chunk ids inside a case":
            all(len(c["expected_chunk_ids"]) == len(set(c["expected_chunk_ids"])) for c in cases),
        "negatives carry no chunk ids / no reference_answer":
            all(not c.get("expected_chunk_ids") and not c.get("reference_answer") for c in neg),
        "positives carry a reference_answer":
            all(c.get("reference_answer", "").strip() for c in pos),
        "reference_answer <=100000":
            all(len(c.get("reference_answer", "")) <= 100_000 for c in cases),
    }
    print("\n=== contract checks ===")
    for label, ok in checks.items():
        print("  [%s] %s" % ("OK" if ok else "FAIL", label))

    if args.chunks and Path(args.chunks).exists():
        known = {c["chunk_id"] for c in json.loads(Path(args.chunks).read_text(encoding="utf-8"))}
        ids = [cid for c in pos for cid in c.get("expected_chunk_ids") or []]
        unknown = [cid for cid in ids if cid not in known]
        hexish = all(len(cid) == 64 and all(ch in "0123456789abcdef" for ch in cid) for cid in ids)
        print("\n=== chunk-id integrity ===")
        print("  referenced ids:", len(ids), "distinct:", len(set(ids)))
        print("  [%s] all ids are 64-hex sha256" % ("OK" if hexish else "FAIL"))
        print("  [%s] all ids exist in the exported index (%d unknown)" % ("OK" if not unknown else "FAIL", len(unknown)))
        checks["chunk ids resolvable"] = not unknown and hexish

    if args.review and Path(args.review).exists():
        rows = list(csv.DictReader(open(args.review, encoding="utf-8-sig", newline="")))
        flagged = [r for r in rows if r.get("needs_review") == "yes"]
        print("\n=== review sheet ===")
        print("  rows:", len(rows), "->", dict(Counter(r.get("case_type") for r in rows)))
        print("  flagged for human review:", len(flagged))
        print("  query types:", dict(Counter(
            r["query_type"] for r in rows if r.get("case_type") == "positive")))

    failed = [k for k, v in checks.items() if not v]
    print("\nRESULT:", "PASS" if not failed else "FAIL -> " + "; ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
