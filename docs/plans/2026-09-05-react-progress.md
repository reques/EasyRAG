# Contextual ReAct Progress Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the dynamic agent's canned startup log with brief, question-specific action updates generated during its actual tool loop.

**Architecture:** Keep create_agent and its tools/skills. A middleware adds a required public action summary to the model-facing tool schema, then strips that metadata before tool execution. Optional <progress> blocks and <answer> blocks stream through an incremental parser. Tool-call IDs and event ordering connect actual actions, observations, and persisted history. Legacy untagged replies remain supported.

**Tech Stack:** Python, LangChain/LangGraph, FastAPI SSE, Vue 3, Vite, pytest, Node test runner.

---

### Task 1: Model output and stream classification
- Modify `app/agents/dynamic.py`; create `app/agents/response_stream.py` and `app/agents/action_progress.py`.
- Remove the artificial understand step. Request short public action descriptions grounded in the question and preceding observations; simple questions go straight to the answer.
- Parse split delimiters, text content blocks, and untagged responses. Never send tool preambles or provider reasoning fields to the answer bubble.
- Verify with parser tests and dynamic-agent fake streams, including multiple rounds, batched tools, direct answers, and recursion fallback.

### Task 2: Ordered live and persisted events
- Modify `backend/server/routers/chat_router.py` and create a small event collector beside the response parser.
- Assign sequence numbers, coalesce progress deltas for history, preserve tool-call IDs, and avoid resending an already streamed answer as a full delta.
- Verify stream/history equivalence and answer exclusion from persisted artifacts.

### Task 3: Process panel and history
- Modify `frontend/src/components/WorkProgress.vue` and `frontend/src/views/ChatView.vue`; extract timeline ingestion helpers into `frontend/src/utils/work-progress.js`.
- Retain existing collapsible summaries; remove redundant placeholder text while a summary exists. Match observations to their tool calls, retain repeated rounds, and replay sequence numbers.
- Add behavior tests for interleaved summaries, tool matching, and historical order; run Node tests and Vite build; inspect the component in a browser with representative events.

### Validation commands
- `python -m pytest tests/test_response_stream.py tests/test_dynamic_agent.py tests/test_context_injection.py tests/test_skill_progressive_disclosure.py -q`
- `node --test frontend/tests/*.test.mjs`
- `npm --prefix frontend run build`

### Scope and tradeoff
- Use the same model call to produce user-facing progress; no extra planning request or fixed phrase rotation.
- Untagged model text waits for message classification so tool-call preambles cannot leak into answers.
- Keep the pre-existing uncommitted fixes for answer artifacts and collapsed thoughts. No changes to model providers or the deep-research loop.

### Implementation verification
- Real create_agent graph with deterministic model output: tool summaries remain in the process, presentation arguments are removed before tool validation, and the final answer is emitted exactly once.
- Configured provider smoke test: calculator request generated the public summary “计算 17 乘以 23 的结果”, executed the calculator, and answered “17 × 23 = **391**。” without fallback.
- Browser fixture: `frontend/tests/work-progress.html` covers incremental summary updates, keyboard expansion, concurrent tool result matching, completion, and historical replay.
- Final validation: 41 backend tests and 5 frontend tests passed; production frontend build passed. Browser replay retained three separate searches and two action summaries with correctly paired observations; no console warnings or errors.
