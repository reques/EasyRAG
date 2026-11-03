# Langfuse Quality Loop Implementation Plan

> Execute the authorized tasks in this chat; retain the current checkout and existing runtime boundaries.

**Goal:** Observe question/answer and retrieval data, collect durable user feedback, and link existing evaluation runs to Langfuse Experiments.

**Architecture:** PostgreSQL remains authoritative. Explicitly selected, bounded content follows metadata_only/redacted settings. A database-backed score queue delivers feedback and evaluation scores outside request transactions. Existing evaluation cases become native OpenTelemetry experiment items, preserving their actual retrieval/model traces without rerunning them.

**Tech Stack:** FastAPI, SQLAlchemy/PostgreSQL, Langfuse Python 5.0.2, OpenTelemetry, Vue 3.

## 1. Question/answer and retrieval observations

- Extend app/observability/redaction.py with explicit structured-content types and bounded field selection; arbitrary dictionaries remain rejected.
- Add app/observability/retrieval.py for basic/enhanced retrieval decorators and root context summaries. Include result IDs, ranking scores, authorized KB IDs and safe text only in redacted mode.
- Instrument app/rag/retriever.py and enhanced_retriever.py, including decomposition, fusion and reranking; propagate context into retrieval executor threads.
- Capture the persisted user question in backend/services/chat_stream_service.py. Record the final answer only after successful result commit; add the saved assistant message ID to durable SSE events.
- Verify privacy, failures, cancellation, parent relationships and retrieval output with an actual in-memory SDK exporter.

## 2. Durable feedback and score delivery

- Add MessageFeedback and ObservabilityScore models with unique owner/message identity, stable score IDs, delivery status and expiring claim leases.
- Add owner-scoped GET/PUT feedback routes under /chat/messages/{id}/feedback; only assistant messages can be rated. Resolve exact sampled trace from persisted stream identity.
- Deliver scores through the public SDK scores API in a background worker; mark accepted only after an accepted API response. Retry after restart and avoid stale delivery acknowledgements after edits.
- Add a compact MessageFeedback.vue component following the existing chat styling, with accessible like/dislike controls, optional reason, loading and failure states.
- Register models/migration and worker lifecycle. Verify ownership, idempotency, revisions, retry, disabled tracing, and UI build.

## 3. Existing evaluations as Experiments

- Add app/observability/experiments.py using documented OTel experiment attributes through a SpanProcessor; no SDK private methods or duplicate experiment runner.
- Give each evaluation run/item stable correlation IDs. Snapshot dataset contents and retrieval configuration; carry one trace/root per actual case.
- Reuse deterministic ranking and per-case Ragas values as scores; missing/failed values remain missing. Queue scores only after the local run commits.
- Add result/history experiment metadata and a safe configured-origin link in the evaluation UI. Support existing ad-hoc and benchmark entry points.
- Verify real SDK exported attributes, per-item trace isolation, score/root matching, unchanged local evaluation metrics and remote failure isolation.

## 4. Documentation and validation

- Update docs/langfuse-observability.md with content modes, feedback delivery semantics, experiment comparison and local dataset identity.
- Run targeted backend regressions, scripts/verify_langfuse.py, frontend tests and production build; inspect UI when available.
- Do not claim remote indexing from local acceptance or exporter tests. No runtime deployment or secret changes are required for code completion.

## Implemented design choices

- The existing checkout was reused. No deployment or credentials were changed.
- Experiments use the documented local-data OTel attribute contract with a dedicated SpanProcessor, keeping one trace per real retrieval case. They do not upload duplicate managed Datasets or replay application calls.
- Feedback and evaluation scores share an acknowledged-API delivery queue. Score edits and delivery acknowledgements serialize on the score row, while outbound HTTP runs outside transactions.
- Root Python regressions follow the repository's existing local-only tests convention. The committed verification harness is frontend/tests/observability-quality.html; it uses synthetic API responses.
- Docker and the dedicated PostgreSQL regression instance were unavailable during this turn. Actual SDK/OTel export, the score HTTP protocol and SQLite-backed application behavior were verified locally; remote indexing and PostgreSQL-specific lock behavior remain deployment checks.
