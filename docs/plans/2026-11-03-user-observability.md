# User Observability Implementation Plan

**Goal:** Let authenticated EasyRAG users inspect their own executions, observations, feedback and evaluation experiments in the application, and explicitly judge a completed answer with an existing model.

**Architecture:** PostgreSQL remains authoritative for execution lifecycle, saved messages, feedback, evaluation runs and judge results. The backend resolves each Langfuse trace from an owned local execution or owned evaluation item before reading the exact remote trace. The Vue interface uses local records even if Langfuse is disabled or unavailable. No project-wide Langfuse query or browser credential is exposed.

**Tech Stack:** FastAPI, async SQLAlchemy/PostgreSQL, existing OpenAI-compatible model resolver, httpx, Vue 3, existing EvaluationView and Langfuse score outbox.

## Decisions

- Native `/observability` page with overview, execution records, evaluation results and evaluation workspace tabs. White/teal styling follows the current workspace. Observation waterfall lives inside execution/item detail, with explicit source labels.
- Bounded, paginated owner-scoped SQL for traces; aggregate counters, model/tool usage and feedback use the same filters. Costs remain null when no reliable price is configured.
- Langfuse read calls have a short timeout, no redirects, bounded response size and no retries; only allowlisted attributes and recursively masked content are returned. Local events remain available separately and are never presented as remote spans.
- Single-turn judging uses a persisted row per trace, a lease to exclude concurrent calls, explicit model selection, a bounded input snapshot, strict numeric result validation and stable Score IDs. No automatic model call on page load. A result stores model, rubric version and explanation; stale/failed runs can retry. Independent judge token usage/cost is currently unknown and is not added to conversation totals. No reasoning or system prompts are used as judge evidence.
- Reuse existing dataset/benchmark editor and execution endpoints; extend summaries with reproducibility and experiment references. Comparing historical runs warns when their datasets/snapshots differ. Every experiment item can open its owned observations in the native page.

## Tasks and validation

1. Add user observability query/read services and authenticated router; integrate `backend/server/main.py`. Test cross-user list/detail/remote access, filter totals, error states, bounded/redacted payloads and score attribution.
2. Add `TurnJudgement` model, additive SQL migration and judge service/router. Test concurrency, duplicate request/cooldown, stale leases, model ownership, malformed model output, score outbox and restart persistence.
3. Add `ObservabilityView.vue`, shared observation waterfall/detail component and pure presentation helpers/tests. Wire sidebar and chat/evaluation item links. Reuse EvaluationView for dataset creation/import/export and benchmark runs; add native history/detail/comparison.
4. Run targeted Python regression tests and frontend Node tests/build. Rebuild application using existing dependencies, apply additive migration, verify health and actual owner-scoped traces/observations/feedback/evaluation/judge with the local stack. Inspect desktop and mobile UI, save screenshots and write operator/code documentation.

## Acceptance

- Ordinary user opens `/observability` without a Langfuse account and sees only their data.
- Existing chat opens exact native trace; child observations retain hierarchy/timing and source, failed/unindexed traces have clear states.
- Existing evaluation run links to each item's observations; two saved runs can be compared, and the existing editor remains functional.
- Explicit judge action calls a configured model, persists validated scores and explanations, and queues the same scores to Langfuse when linked. Pending delivery is distinct from confirmed remote indexing.
- API keys, other users' records and unselected model prompts are absent from responses.
