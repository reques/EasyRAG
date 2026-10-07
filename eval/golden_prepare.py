"""Stage 1 of the standardized Golden Set pipeline (runs INSIDE the EasyRAG backend container).

Why it must run in the container
--------------------------------
Chunk IDs are not stored anywhere: ``get_document_chunk_id`` derives them as
``sha256(kb_id US source US chunk_index US content)`` (US = U+001F). Reproducing
that offline is possible, but importing the *real* function removes any risk of
drift, so this stage calls the actual ``app.rag.retriever`` helper.

It also embeds the corpus questions and evidence facts with the *same* embedder
that built the index, which is what makes offline evidence->chunk matching
meaningful (BGE-M3 here).

Outputs (JSON, copied back to the host by the caller):
  /tmp/golden_chunks.json   [{source, content, chunk_id, vector}]
  /tmp/golden_embeddings.json {"mode": str, "model": str, "dim": int,
                               "texts": [...], "vectors": [[...]]}

Usage (inside the container):
  python golden_prepare.py --kb-id <uuid> --queries /tmp/master_queries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from app.rag.embeddings import get_embedder
from app.rag.retriever import get_document_chunk_id
from app.core.config import get_settings


def load_questions(queries_csv: str) -> list[str]:
    with open(queries_csv, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: list[str] = []
    for row in rows:
        question = (row.get("Query") or "").strip()
        if question and question not in seen:
            seen.append(question)
    return seen


def load_facts(queries_csv: str) -> list[str]:
    with open(queries_csv, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: list[str] = []
    for row in rows:
        try:
            facts = json.loads(row.get("Supporting Facts") or "[]")
        except json.JSONDecodeError:
            continue
        for fact in facts:
            text = (fact.get("text") or "").strip()
            if text and text not in seen:
                seen.append(text)
    return seen


def export_chunks(kb_id: str) -> list[dict]:
    from pymilvus import Collection, connections

    cfg = get_settings()
    connections.connect(
        alias="default",
        host=cfg.MILVUS_HOST,
        port=str(cfg.MILVUS_PORT),
    )
    collection = Collection(cfg.MILVUS_COLLECTION)
    collection.load()
    rows = collection.query(
        expr=f'knowledge_base_id == "{kb_id}"',
        output_fields=["content", "source", "vector"],
        limit=16384,
    )
    chunks = []
    for row in rows:
        content = row["content"]
        source = row["source"]
        chunks.append({
            "source": source,
            "content": content,
            # The exact derivation used at retrieval/scoring time.
            "chunk_id": get_document_chunk_id(kb_id, content, {"source": source}),
            "vector": [float(x) for x in row["vector"]],
        })
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--queries", required=True, help="master query CSV (with header)")
    parser.add_argument("--chunks-out", default="/tmp/golden_chunks.json")
    parser.add_argument("--embeddings-out", default="/tmp/golden_embeddings.json")
    args = parser.parse_args()

    chunks = export_chunks(args.kb_id)
    print(f"[prepare] exported {len(chunks)} chunks for kb {args.kb_id}")
    with open(args.chunks_out, "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False)

    questions = load_questions(args.queries)
    facts = load_facts(args.queries)
    print(f"[prepare] {len(questions)} unique questions, {len(facts)} unique evidence facts")

    # Facts and questions share one embedding pass; they are disjoint in practice.
    texts = facts + [q for q in questions if q not in set(facts)]
    embedder = get_embedder()
    print(f"[prepare] embedding {len(texts)} texts with {type(embedder).__name__}")
    vectors = embedder.embed_texts(texts)

    payload = {
        "mode": str(getattr(get_settings(), "EMBEDDING_TYPE", "")),
        "model": str(getattr(get_settings(), "EMBEDDING_MODEL_NAME", "")),
        "dim": len(vectors[0]) if vectors else 0,
        "texts": texts,
        "vectors": [[float(x) for x in v] for v in vectors],
    }
    with open(args.embeddings_out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(f"[prepare] wrote {args.chunks_out} and {args.embeddings_out} (dim={payload['dim']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
