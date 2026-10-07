"""Stage 2 of the standardized Golden Set pipeline (runs on the host, no services needed).

Turns RAG-Multi-Corpus query files into an importable EasyRAG Golden Set:
``{name, description, cases:[{question, expected_file_id, expected_filename,
expected_chunk_ids, reference_answer, expect_miss}]}``

Why evidence->chunk matching is semantic, not textual
-----------------------------------------------------
The corpus "Supporting Facts" are LLM-written *paraphrases* of the source
documents, not verbatim spans (measured: only ~29% appear verbatim in any
chunk). So plain substring matching cannot recover chunk-level labels. We
instead embed each evidence fact with the same embedder that built the index
and accept the chunks of the target file whose cosine similarity clears a
threshold. That is the offline equivalent of the retrieval-then-confirm
workflow the EasyRAG docs prescribe (`POST /evaluation/chunk-candidates`).

Usage
-----
  python build_golden_set.py \
     --corpus   "E:\\Project\\RAG-Multi-Corpus" \
     --enterprise "Cloudway 24" \
     --kb-files  eval/_work/kb_files.json \
     --chunks    eval/_work/golden_chunks.json \
     --embeddings eval/_work/golden_embeddings.json \
     --out-dir   eval/cloudway24
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

US = "\x1f"

# ── text normalisation ───────────────────────────────────────────────────────

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U0000FE0F\U00002190-\U000021FF\U00002B00-\U00002BFF]+"
)


def _fold(text: str) -> str:
    """Aggressive, markdown-aware folding used for filename + chunk matching."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*", " ", s, flags=re.M)
    s = re.sub(r"^\s{0,3}[-*+]\s+", " ", s, flags=re.M)
    s = re.sub(r"^\s{0,3}\d+[.)]\s+", " ", s, flags=re.M)
    s = re.sub(r"^\s{0,3}>\s*", " ", s, flags=re.M)
    for ch in ("**", "__"):
        s = s.replace(ch, " ")
    for ch in ("*", "_", "`", "\u00a0", "\u2022"):
        s = s.replace(ch, " ")
    s = _EMOJI.sub(" ", s)
    for src, dst in (
        ("\u2013", "-"), ("\u2014", "-"), ("\u2019", "'"), ("\u2018", "'"),
        ("\u201c", '"'), ("\u201d", '"'),
    ):
        s = s.replace(src, dst)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().casefold()


def compute_chunk_id(kb_id: str, source: str, content: str, chunk_index=None) -> str:
    """Mirror of app.rag.retriever.get_document_chunk_id (Milvus carries no chunk_index)."""
    payload = US.join((
        str(kb_id),
        str(source or ""),
        str(chunk_index if chunk_index is not None else ""),
        str(content or ""),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine(a, b, norm_a=None, norm_b=None) -> float:
    na = norm_a if norm_a is not None else math.sqrt(sum(x * x for x in a))
    nb = norm_b if norm_b is not None else math.sqrt(sum(x * x for x in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    return dot / (na * nb)


# ── input loading ────────────────────────────────────────────────────────────


def normalize_enterprise(name: str) -> str:
    """'Cloudway 24' / 'CloudWay-24' / 'cloudway_24' -> 'cloudway24'."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").casefold())


# The corpus ships 7 canonical query types plus a handful of one-off labels
# ("Supporting", "Eligibility Inquiry", "Eligibility", "Safety"). Aggregating
# metrics by an open-ended label set produces singleton buckets, so unknown
# labels are folded into the canonical taxonomy and the original is retained
# for traceability.
CANONICAL_TYPES = (
    "Descriptive", "Analytical", "Comparative", "Boolean",
    "Temporal", "Procedural", "Open-Ended",
)
_CANONICAL_BY_FOLD = {t.casefold(): t for t in CANONICAL_TYPES}


def canonical_type(raw: str) -> str:
    key = str(raw or "").strip().casefold()
    if key in _CANONICAL_BY_FOLD:
        return _CANONICAL_BY_FOLD[key]
    if "eligib" in key or "safety" in key or "support" in key:
        return "Descriptive"
    return "Descriptive"


def load_queries(corpus: Path, enterprise: str):
    path = corpus / "datasets" / "Dataset categories - queries_01122025.csv"
    if not path.exists():
        raise SystemExit(f"master query CSV not found: {path}")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = normalize_enterprise(enterprise)
    kept, other = [], []
    for index, row in enumerate(rows, start=2):  # 2 = first data row (1-based incl. header)
        try:
            facts = json.loads(row.get("Supporting Facts") or "[]")
        except json.JSONDecodeError:
            facts = []
        entry = {
            "row": index,
            "enterprise": (row.get("Enterprise Name") or "").strip(),
            "query_type": (row.get("Query Type") or "").strip(),
            "question": (row.get("Query") or "").strip(),
            "facts": [f for f in facts if (f.get("text") or "").strip()],
        }
        (kept if normalize_enterprise(entry["enterprise"]) == target else other).append(entry)
    return kept, other, path


def load_kb_files(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [{"file_id": str(f["file_id"]), "filename": str(f["filename"])} for f in raw]


def load_chunks(path: Path, kb_id: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = []
    mismatch = 0
    for item in raw:
        computed = compute_chunk_id(kb_id, item["source"], item["content"])
        declared = item.get("chunk_id")
        if declared and declared != computed:
            mismatch += 1
        chunks.append({
            "source": item["source"],
            "content": item["content"],
            "chunk_id": declared or computed,
            "vector": item.get("vector") or [],
        })
    if mismatch:
        print(f"  WARNING: {mismatch} chunk ids did not match the local formula")
    else:
        print(f"  chunk-id formula verified against {len(chunks)} index chunks")
    return chunks


def load_embeddings(path: Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t: v for t, v in zip(raw["texts"], raw["vectors"])}


# ── core build ───────────────────────────────────────────────────────────────


def build(args) -> dict:
    corpus = Path(args.corpus)
    kb_files = load_kb_files(args.kb_files)
    queries, other_queries, csv_path = load_queries(corpus, args.enterprise)

    kb_id = args.kb_id
    if not kb_id:
        kb_id = str(json.loads(Path(args.kb_files).read_text(encoding="utf-8-sig"))[0].get("kb_id") or "")
    chunks = load_chunks(args.chunks, kb_id)
    embeddings = load_embeddings(args.embeddings)

    print(f"[build] enterprise={args.enterprise!r} queries={len(queries)} kb_files={len(kb_files)} chunks={len(chunks)}")

    # filename resolution: exact -> folded
    by_exact = {f["filename"]: f for f in kb_files}
    by_folded: dict[str, list[dict]] = defaultdict(list)
    for f in kb_files:
        by_folded[_fold(f["filename"])].append(f)

    chunks_by_source: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_source[c["source"]].append(c)
    # pre-normalise chunk vectors
    for c in chunks:
        if c["vector"]:
            n = math.sqrt(sum(x * x for x in c["vector"]))
            c["_norm"] = n if n > 1e-12 else 1e-12
    source_by_folded = {_fold(s): s for s in chunks_by_source}

    def resolve(raw_name: str):
        if raw_name in by_exact:
            return by_exact[raw_name], "exact"
        cands = by_folded.get(_fold(raw_name)) or []
        if len(cands) == 1:
            return cands[0], "folded"
        if len(cands) > 1:
            return None, "ambiguous"
        return None, "missing"

    cases, review, excluded, warnings = [], [], [], []
    seen_questions: dict[str, int] = {}
    stats = Counter()

    for entry in queries:
        question = entry["question"]
        if not question:
            excluded.append({**entry, "missing_file": "", "reason": "空问题"})
            stats["empty_question"] += 1
            continue

        file_names = []
        for fact in entry["facts"]:
            name = (fact.get("filename") or "").strip()
            if name and name not in file_names:
                file_names.append(name)

        resolved, reasons = [], []
        for name in file_names:
            kbfile, how = resolve(name)
            if kbfile is None:
                reasons.append(f"{name} ({how})")
            else:
                resolved.append((name, kbfile, how))
                if how != "exact":
                    warnings.append(f"row {entry['row']}: filename {name!r} matched by {how}")

        if not resolved:
            excluded.append({
                **entry,
                "missing_file": "; ".join(file_names),
                "reason": "评测集引用的证据文件在当前知识库中不存在: " + "; ".join(reasons),
            })
            stats["unresolved_file"] += 1
            continue

        # primary expected file = the one carrying the most evidence
        weight = Counter(kb["file_id"] for _, kb, _ in resolved)
        primary_id = weight.most_common(1)[0][0]
        primary = next(kb for _, kb, _ in resolved if kb["file_id"] == primary_id)

        matched: dict[str, dict] = {}
        fact_sims = []
        best_chunk = None  # (sim, {chunk_id, content}) — global best across all facts
        for fact in entry["facts"]:
            vector = embeddings.get(fact["text"])
            if not vector:
                continue
            qn = math.sqrt(sum(x * x for x in vector)) or 1e-12
            for name, kbfile, _ in resolved:
                source = kbfile["filename"]
                if source not in chunks_by_source:
                    source = source_by_folded.get(_fold(source), source)
                pool = chunks_by_source.get(source) or []
                scored = []
                for c in pool:
                    if not c["vector"]:
                        continue
                    sim = cosine(vector, c["vector"], qn, c["_norm"])
                    scored.append((sim, c))
                scored.sort(key=lambda t: -t[0])
                if scored:
                    fact_sims.append(scored[0][0])
                    if best_chunk is None or scored[0][0] > best_chunk[0]:
                        best_chunk = (scored[0][0], scored[0][1])
                for sim, c in scored:
                    if sim < args.threshold:
                        break
                    prev = matched.get(c["chunk_id"])
                    if prev is None or sim > prev["sim"]:
                        matched[c["chunk_id"]] = {
                            "sim": sim,
                            "content": c["content"],
                            "source": c["source"],
                        }

        ranked = sorted(matched.items(), key=lambda kv: -kv[1]["sim"])
        # Guarantee a chunk-level reference for every case that has any located
        # evidence: keep the single best chunk even below `threshold` when it
        # still clears `floor`. This keeps reference_mode="chunk_ids" (the whole
        # point of a chunk-level golden set) while the confidence flag routes
        # the weak ones to human review.
        if not ranked and best_chunk is not None and best_chunk[0] >= args.floor:
            sim, chunk = best_chunk
            ranked = [(chunk["chunk_id"], {
                "sim": sim,
                "content": chunk["content"],
                "source": chunk["source"],
            })]
        chunk_ids = [cid for cid, _ in ranked[: args.max_chunks]]
        sim_max = ranked[0][1]["sim"] if ranked else 0.0
        sim_min_selected = ranked[min(len(ranked), args.max_chunks) - 1][1]["sim"] if ranked else 0.0
        needs_review = (not chunk_ids) or sim_max < args.review_threshold

        seen_questions.setdefault(question, 0)
        seen_questions[question] += 1

        # de-duplicate evidence text, preserving order
        evidence, seen_text = [], set()
        for fact in entry["facts"]:
            text = fact["text"].strip()
            if text and text not in seen_text:
                seen_text.add(text)
                evidence.append(text)

        case = {
            "question": question,
            "expected_file_id": primary["file_id"],
            "expected_chunk_ids": chunk_ids,
            "reference_answer": "\n\n".join(evidence),
            "expect_miss": False,
        }
        cases.append(case)
        review.append({
            "id": f"{args.prefix}-{len(cases):03d}",
            "case_type": "positive",
            "row": entry["row"],
            "query_type": canonical_type(entry["query_type"]),
            "query_type_raw": entry["query_type"],
            "question": question,
            "expected_filename": primary["filename"],
            "expected_file_id": primary["file_id"],
            "num_evidence": len(entry["facts"]),
            "num_chunks_labeled": len(chunk_ids),
            "sim_max": round(sim_max, 4),
            "sim_min_selected": round(sim_min_selected, 4),
            "needs_review": "yes" if needs_review else "",
            "expected_chunk_ids": "|".join(chunk_ids),
            "reference_answer": case["reference_answer"],
            "top_snippet": (ranked[0][1]["content"][:200].replace("\n", " ") if ranked else ""),
        })
        stats["cases"] += 1
        if needs_review:
            stats["needs_review"] += 1
        if not chunk_ids:
            stats["no_chunk_label"] += 1

    negatives = build_negatives(args, other_queries, kb_files, chunks, embeddings, len(cases))
    cases.extend(negatives)

    return {
        "cases": cases,
        "review": review,
        "excluded": excluded,
        "warnings": warnings,
        "stats": stats,
        "negatives": negatives,
        "seen_questions": {q: n for q, n in seen_questions.items() if n > 1},
        "csv_path": csv_path,
        "kb_files": kb_files,
        "chunks": chunks,
    }


def build_negatives(args, other_queries, kb_files, chunks, embeddings, offset) -> list[dict]:
    """Negative samples: a cross-domain question plus a provably unrelated file.

    A case is accepted only if the question's best cosine similarity against
    *every* chunk of the chosen file stays below ``--negative-max-sim``, so the
    document genuinely is not evidence for it.

    Candidates are drawn round-robin across source enterprises and then across
    query types, so the negative set is not dominated by whichever enterprise
    happens to come first in the CSV.
    """
    if args.negatives <= 0:
        return []
    chunks_by_source: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["vector"]:
            n = math.sqrt(sum(x * x for x in c["vector"])) or 1e-12
            c["_norm"] = n
            chunks_by_source[c["source"]].append(c)
    usable = [f for f in kb_files if chunks_by_source.get(f["filename"])]
    if not usable:
        return []

    # level 1: round-robin across source enterprises
    by_enterprise: dict[str, list] = defaultdict(list)
    for entry in other_queries:
        if entry["question"] and entry["question"] in embeddings:
            by_enterprise[entry["enterprise"] or "Unknown"].append(entry)
    interleaved: list = []
    cursors = {e: 0 for e in by_enterprise}
    while any(cursors[e] < len(by_enterprise[e]) for e in cursors):
        for ent in sorted(by_enterprise):
            if cursors[ent] < len(by_enterprise[ent]):
                interleaved.append(by_enterprise[ent][cursors[ent]])
                cursors[ent] += 1

    # level 2: round-robin across query types, preserving the order above
    by_type: dict[str, list] = defaultdict(list)
    for entry in interleaved:
        by_type[canonical_type(entry["query_type"])].append(entry)

    out: list[dict] = []
    type_cursors = {t: 0 for t in by_type}
    while len(out) < args.negatives and any(
        type_cursors[t] < len(by_type[t]) for t in by_type
    ):
        for qtype in sorted(by_type):
            if len(out) >= args.negatives:
                break
            pool = by_type[qtype]
            while type_cursors[qtype] < len(pool):
                entry = pool[type_cursors[qtype]]
                type_cursors[qtype] += 1
                vector = embeddings[entry["question"]]
                qn = math.sqrt(sum(x * x for x in vector)) or 1e-12
                best = None
                for f in usable:
                    peak = max(
                        (cosine(vector, c["vector"], qn, c["_norm"])
                         for c in chunks_by_source[f["filename"]]),
                        default=1.0,
                    )
                    if best is None or peak < best[0]:
                        best = (peak, f)
                if best is None or best[0] > args.negative_max_sim:
                    continue
                peak, f = best
                out.append({
                    "question": entry["question"],
                    "expected_file_id": f["file_id"],
                    "expected_chunk_ids": [],
                    "reference_answer": "",
                    "expect_miss": True,
                    "_meta": {
                        "source_enterprise": entry["enterprise"],
                        "query_type": qtype,
                        "query_type_raw": entry["query_type"],
                        "target_filename": f["filename"],
                        "max_similarity": round(peak, 4),
                    },
                })
                break
    return out


# ── output ───────────────────────────────────────────────────────────────────


def write_outputs(args, result) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.slug

    name = args.name or f"{slug}-golden-v1"
    description = (
        f"{args.enterprise} 评测集 v1（规范化构建）。"
        f"来源：RAG-Multi-Corpus master query CSV；"
        f"expected_file_id 已绑定当前知识库真实文件 UUID；"
        f"expected_chunk_ids 由 BGE-M3 证据语义匹配得到"
        f"（接受阈值 {args.threshold}，兜底 {args.floor}，每例上限 {args.max_chunks}）；"
        f"reference_answer 为 golden evidence 原文；负样本 {args.negatives} 条。"
    )[:512]

    # The golden set carries ONLY the five contract keys (+ expected_filename for
    # readability, which the importer also understands). Build metadata such as
    # the negative-sample provenance lives in the review CSV instead.
    clean_cases = []
    for case in result["cases"]:
        clean_cases.append({k: v for k, v in case.items() if k != "_meta"})

    golden = {
        "name": name,
        "description": description,
        "cases": clean_cases,
    }
    golden_path = out_dir / f"{slug}_golden_set.json"
    golden_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")

    review_path = out_dir / f"{slug}_qa_review.csv"
    fields = [
        "id", "case_type", "row", "query_type", "query_type_raw", "question",
        "expected_filename", "expected_file_id", "num_evidence", "num_chunks_labeled",
        "sim_max", "sim_min_selected", "needs_review", "expected_chunk_ids",
        "reference_answer", "top_snippet",
    ]
    negative_rows = []
    for index, case in enumerate(result["negatives"], start=1):
        meta = case.get("_meta") or {}
        negative_rows.append({
            "id": f"{args.prefix}-neg{index:03d}",
            "case_type": "negative",
            "row": "",
            "query_type": meta.get("query_type", ""),
            "query_type_raw": meta.get("query_type_raw", ""),
            "question": case["question"],
            "expected_filename": meta.get("target_filename", ""),
            "expected_file_id": case["expected_file_id"],
            "num_evidence": 0,
            "num_chunks_labeled": 0,
            "sim_max": meta.get("max_similarity", ""),
            "sim_min_selected": "",
            "needs_review": "",
            "expected_chunk_ids": "",
            "reference_answer": "",
            "top_snippet": "source_enterprise=%s" % meta.get("source_enterprise", ""),
        })
    with open(review_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["review"] + negative_rows:
            writer.writerow(row)

    excluded_path = out_dir / f"{slug}_excluded.csv"
    with open(excluded_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["row", "query_type", "question", "missing_file", "reason"]
        )
        writer.writeheader()
        for row in result["excluded"]:
            writer.writerow({
                "row": row["row"],
                "query_type": row["query_type"],
                "question": row["question"],
                "missing_file": row["missing_file"],
                "reason": row["reason"],
            })

    review = result["review"]
    sims = sorted(float(r["sim_max"]) for r in review if r["sim_max"] != "")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "enterprise": args.enterprise,
        "source_csv": str(result["csv_path"]),
        "knowledge_base_id": args.kb_id,
        "params": {
            "threshold": args.threshold,
            "floor": args.floor,
            "review_threshold": args.review_threshold,
            "max_chunks_per_case": args.max_chunks,
            "negatives": args.negatives,
            "negative_max_sim": args.negative_max_sim,
        },
        "counts": {
            "kb_files": len(result["kb_files"]),
            "index_chunks": len(result["chunks"]),
            "queries_in_corpus": len(result["review"]) + len(result["excluded"]),
            "positive_cases": len(review),
            "negative_cases": len(result["negatives"]),
            "total_cases": len(result["cases"]),
            "excluded_cases": len(result["excluded"]),
            "cases_needing_review": result["stats"]["needs_review"],
            "cases_without_chunk_label": result["stats"]["no_chunk_label"],
        },
        "label_quality": {
            "cases_with_chunk_ids": sum(1 for r in review if r["num_chunks_labeled"]),
            "chunk_label_coverage": (
                round(sum(1 for r in review if r["num_chunks_labeled"]) / len(review), 4)
                if review else None
            ),
            "sim_max_min": round(sims[0], 4) if sims else None,
            "sim_max_median": round(sims[len(sims) // 2], 4) if sims else None,
            "cases_below_review_threshold": sum(1 for s in sims if s < args.review_threshold),
            "avg_labeled_chunks": round(
                sum(r["num_chunks_labeled"] for r in review) / len(review), 2
            ) if review else None,
        },
        "query_type_distribution": dict(Counter(r["query_type"] for r in review)),
        "negative_query_types": dict(
            Counter(n["_meta"]["query_type"] for n in result["negatives"])
        ),
        "negative_max_similarity_range": (
            [round(min(n["_meta"]["max_similarity"] for n in result["negatives"]), 4),
             round(max(n["_meta"]["max_similarity"] for n in result["negatives"]), 4)]
            if result["negatives"] else None
        ),
        "duplicate_questions": len(result["seen_questions"]),
        "warnings": result["warnings"],
        "outputs": {
            "golden_set": golden_path.name,
            "qa_review": review_path.name,
            "excluded": excluded_path.name,
        },
    }
    summary_path = out_dir / f"{slug}_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[build] {golden_path}")
    print(f"[build] {review_path}")
    print(f"[build] {excluded_path}")
    print(f"[build] {summary_path}")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["label_quality"], ensure_ascii=False, indent=2))
    if result["warnings"]:
        print("[build] warnings:")
        for w in result["warnings"][:20]:
            print("   ", w)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a standardized EasyRAG Golden Set")
    parser.add_argument("--corpus", default=r"E:\Project\RAG-Multi-Corpus")
    parser.add_argument("--enterprise", required=True)
    parser.add_argument("--kb-id", default="")
    parser.add_argument("--kb-files", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--slug", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="cosine similarity required to ACCEPT an evidence chunk")
    parser.add_argument("--floor", type=float, default=0.60,
                        help="if nothing clears --threshold, still label the single best chunk "
                             "when it clears this floor (keeps chunk-level reference mode)")
    parser.add_argument("--review-threshold", type=float, default=0.80,
                        help="flag a case for human review when its best match is below this")
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=0)
    parser.add_argument("--negative-max-sim", type=float, default=0.55)
    args = parser.parse_args()

    if not args.slug:
        args.slug = normalize_enterprise(args.enterprise)
    if not args.prefix:
        args.prefix = args.slug[:3]

    result = build(args)
    write_outputs(args, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
