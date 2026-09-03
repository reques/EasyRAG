# Deep Research Progress Journal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Stream concise, user-facing progress summaries throughout deep research without exposing internal reasoning, and preserve them in conversation history.

**Architecture:** Convert DeepAgents execution steps into a separate `progress_summary` event stream using a deterministic backend projector. The chat router filters private reasoning events, emits and persists summaries, while Vue renders them in a dedicated journal instead of the raw activity stream for deep research messages.

**Tech Stack:** Python, FastAPI SSE, pytest, Vue 3, Lucide icons, Vite.

---

### Task 1: Progress summary projector

**Files:**
- Create: `app/agents/deep/progress.py`
- Create: `tests/test_deep_research_progress.py`

1. Write tests for planning, knowledge retrieval, web search, tool completion, warning, synthesis, and duplicate suppression.
2. Run `pytest tests/test_deep_research_progress.py -q`; expect failure because the projector does not exist.
3. Implement `DeepResearchProgressProjector.feed(step, detail)` returning a JSON-compatible event or `None`.
4. Ensure generated text contains only high-level actions/results/next steps and never copies reasoning text or raw tool arguments.
5. Run the focused tests; expect all to pass.

### Task 2: DeepAgents SSE and persistence alignment

**Files:**
- Modify: `backend/server/routers/chat_router.py`
- Test: `tests/test_deep_research_progress.py`

1. Add a test covering the public/private step filter.
2. Instantiate the projector per deep-research request.
3. Emit `progress_summary` events through the existing queue as stages advance.
4. Keep the full raw step/artifact stream server-side; send only projected progress summaries and the final answer to deep-research clients.
5. Persist `progress_summaries` in assistant message metadata and include it in the final `done` event.
6. Run the focused tests and the existing DeepAgents observer tests.

### Task 3: Vue progress journal

**Files:**
- Create: `frontend/src/components/ProgressJournal.vue`
- Modify: `frontend/src/views/ChatView.vue`
- Modify: `frontend/src/style.css`

1. Add a compact journal with phase icon, timestamp, continuous connector, live state, and accessible status semantics.
2. Restore `progress_summaries` from message history.
3. Append incoming `progress_summary` SSE events to the active assistant message.
4. Prefer the journal over `AgentActivity` whenever progress summaries exist, leaving ordinary conversations unchanged.
5. Preserve live summaries on `done`, error, and interrupted requests.

### Task 4: Verification

**Files:**
- Verify all files above.

1. Run `pytest tests/test_deep_research_progress.py tests/test_deepagents_observe.py -q`.
2. Run `npm run build` in `frontend`.
3. Run `git diff --check -- app backend frontend tests docs/plans`.
4. Confirm the event contract: progress appears before `done`, final content remains separate, private reasoning is absent, and history replay restores the journal.
