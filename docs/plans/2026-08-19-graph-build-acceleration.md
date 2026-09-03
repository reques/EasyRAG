# Knowledge Graph Build Acceleration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce first-time knowledge-graph build latency by packing small chunks into fewer LLM requests, increasing safe concurrency, and reusing cached extraction results on subsequent builds.

**Architecture:** Keep the existing Milvus retrieval chunks unchanged for search quality and provenance. Add an LLM-specific packed extraction path that groups several original chunks into one JSON request while returning one result per chunk, then persist each original chunk result in PostgreSQL using a content/model/prompt-version cache key. The existing Neo4j, PostgreSQL, and Milvus graph writers continue to receive a result list aligned one-to-one with the original chunks.

**Tech Stack:** Python 3.11, asyncio, SQLAlchemy async/PostgreSQL, OpenAI-compatible chat JSON API, pytest, Vue 3.

---

### Task 1: Lock down packed extraction behavior

**Files:**
- Create: `tests/test_graph_build_acceleration.py`
- Modify: `app/rag/extractors/llm_extractor.py`
- Modify: `app/rag/extractors/base.py`

**Step 1: Write the failing tests**

Test that chunks are grouped up to the configured character/count limits, one LLM call returns per-chunk results keyed by stable item IDs, malformed or missing items degrade to empty results, and output ordering still matches input ordering.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_build_acceleration.py -q`

Expected: FAIL because the packed extraction API is not implemented.

**Step 3: Implement the minimal packed extractor**

Add a versioned packed prompt and override `LLMExtractor.extract_batch`. Build packs without splitting original chunks, call the LLM once per pack under the existing semaphore, parse a top-level `items` array, and map results back to original positions. Retain the single-chunk `extract` method for compatibility.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_build_acceleration.py -q`

Expected: PASS.

### Task 2: Add persistent extraction cache and incremental reuse

**Files:**
- Modify: `backend/storage/postgres/models_knowledge.py`
- Modify: `backend/services/graph_build_service.py`
- Test: `tests/test_graph_build_acceleration.py`

**Step 1: Write failing cache tests**

Test deterministic cache keys, model/prompt-version invalidation, cache-hit/miss reconstruction in original chunk order, and serialization round trips for `ExtractionResult`.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_build_acceleration.py -q`

Expected: FAIL because cache helpers and model do not exist.

**Step 3: Implement cache model and helpers**

Add `GraphExtractionCache` with a SHA-256 primary key and JSON result payload. Before extraction, bulk-load matching keys; extract only misses; upsert successful miss results; then return the complete aligned result list. Include extractor name, concrete model name, and prompt version in the key.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_build_acceleration.py -q`

Expected: PASS.

### Task 3: Tune defaults and correct long-running UI behavior

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.template`
- Modify: `frontend/src/views/KnowledgeView.vue`
- Test: `tests/test_graph_build_acceleration.py`

**Step 1: Add configuration assertions**

Assert defaults of eight concurrent packed requests, 1,800 characters and four original chunks per request, graph-specific output-token limit, cache enabled, and embedding batch size 32.

**Step 2: Implement configuration and UI changes**

Document the new settings in `.env.template`. Replace the fixed ten-minute polling stop with continuous polling plus a non-fatal long-running notice so the UI no longer reports an active backend build as interrupted.

**Step 3: Verify**

Run: `pytest tests/test_graph_build_acceleration.py -q`

Run: `npm --prefix frontend run build`

Expected: both commands pass.

### Task 4: Regression verification

**Files:**
- Verify: `verify/verify_graphrag.py`
- Verify: existing knowledge ingestion and configuration tests

**Step 1: Run focused regression tests**

Run: `pytest tests/test_graph_build_acceleration.py tests/test_ingestion_progress.py tests/test_custom_model_config.py -q`

Expected: PASS.

**Step 2: Run offline GraphRAG verification**

Run: `python verify/verify_graphrag.py`

Expected: all available checks pass.

**Step 3: Inspect runtime state**

Confirm the development server restarted cleanly, the interrupted build is no longer left in `running`, and the new cache table exists.
