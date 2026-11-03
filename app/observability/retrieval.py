"""Selected retrieval facts and content, shared by all retriever backends."""
from __future__ import annotations

from functools import wraps
from contextvars import ContextVar

from app.observability.context import execution_context, observation_scope
from app.observability.redaction import selected_content

_retrieval_depth = ContextVar("observability_retrieval_depth", default=0)


def documents_for_observation(result):
    from app.core.config import get_settings
    cfg = get_settings()
    documents = result if isinstance(result, list) else getattr(result, "raw_docs", [])
    rows = []
    for doc in documents[:20]:
        if isinstance(doc, dict):
            meta = doc.get("metadata") or {}
            text = doc.get("content")
        else:
            meta = getattr(doc, "metadata", {}) or {}
            text = getattr(doc, "content", "")
            meta = {**meta, "score": getattr(doc, "score", meta.get("score"))}
        if not meta.get("chunk_id") and meta.get("knowledge_base_id") and isinstance(text, str):
            # Legacy Milvus stores only KB/source/content; use the same stable
            # identity as the retrieval/evaluation APIs without changing its schema.
            from app.rag.retriever import get_document_chunk_id
            meta = {**meta, "chunk_id": get_document_chunk_id(meta["knowledge_base_id"], text, meta)}
        rows.append({
            **{key: meta.get(key) for key in ("chunk_id", "file_id", "knowledge_base_id", "score", "rerank_score")},
            "content": text[:cfg.EASYRAG_OBSERVABILITY_CONTENT_MAX_CHARS]
                       if isinstance(text, str) and cfg.EASYRAG_OBSERVABILITY_CONTENT_MODE == "redacted" else "",
        })
    return rows


def record_retrieval(observation, query, result, *, include_root=True):
    rows = documents_for_observation(result)
    documents = result if isinstance(result, list) else getattr(result, "raw_docs", [])
    observation.update(output=selected_content(contexts=rows), metadata={
        "resultCount": len(documents), "retrievalResults": rows,
    })
    context = execution_context()
    if include_root and context is not None and context.root is not None:
        with context.lock:
            # Bound retained context even for long-running agents and retries.
            for row in rows:
                if row not in context.retrieval_documents:
                    context.retrieval_documents.append(row)
            context.retrieval_documents = context.retrieval_documents[-20:]
            context.root.update(input=selected_content(context.question, context.retrieval_documents),
                                metadata={"retrievalResults": context.retrieval_documents})


def observe_retrieval(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, query, *args, **kwargs):
            if execution_context() is None:
                return fn(self, query, *args, **kwargs)
            depth = _retrieval_depth.get()
            token = _retrieval_depth.set(depth + 1)
            try:
                with observation_scope(name, kind="retriever", input=query, metadata={
                    "knowledgeBaseIds": kwargs.get("knowledge_base_ids"),
                    "topK": kwargs.get("top_k"), "scoreThreshold": kwargs.get("score_threshold"),
                    "retrievalMode": "enhanced" if name.endswith("enhanced") else "basic",
                }) as observation:
                    result = fn(self, query, *args, **kwargs)
                    # Intermediate candidates remain on their retrieval nodes;
                    # only results returned to the caller become root context.
                    record_retrieval(observation, query, result, include_root=depth == 0)
                    return result
            finally:
                _retrieval_depth.reset(token)
        return wrapped
    return decorate


def observe_stage(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if execution_context() is None:
                return fn(*args, **kwargs)
            candidates = next((value for value in args[1:] if isinstance(value, list)
                               and value and hasattr(value[0], "content")), None)
            with observation_scope(name, metadata={"candidateCount": len(candidates) if candidates is not None else None}) as observation:
                result = fn(*args, **kwargs)
                if isinstance(result, list):
                    observation.update(metadata={"resultCount": len(result),
                        "retrievalResults": documents_for_observation(result)})
                return result
        return wrapped
    return decorate
