# Agent Management Implementation Plan

**Goal:** Add usable main/subagent cards and a provider tab, connected to existing execution and model configuration.

**Architecture:** Preserve built-in/global definitions and per-user model storage. Persist personal agent definitions in PostgreSQL, resolve owner-authorized models and subagents before chat admission, and carry an immutable request configuration through ContextVars and worker threads. No shared mutable agent cache may capture personal prompts or tool scopes.

**Tech Stack:** FastAPI, SQLAlchemy/PostgreSQL, Vue 3, LangChain's existing agent harness.

### 1. Persistence and management API
- Add `models_agent_config.py`, an additive SQL migration, `agent_config_service.py` and `/agents` routes.
- List actual built-in main agents and file/built-in subagents. Support personal create/edit/enable/delete and copying presets. Validate names, model ownership, child kinds and tool names; reject dangling deletes.
- Add owner-only model configuration detail/update routes to existing `/chat/models`, preserving encrypted keys when blank. Keep built-in environment configuration read-only.

### 2. Runtime integration
- Add a request-scoped runtime module, extend ChatContext and agent builders, and select engine from the chosen configuration.
- Limit model-bound and registry-invoked tools; use configured child catalog and optional child model in task/spawn_tasks without bypassing Skills, KB or sandbox authorization.
- Snapshot selection in message metadata for history/resume, preserve old request fingerprints when agent_id is absent, and resolve selections before writing messages.
- Verify request isolation, owner boundaries, cache isolation, disabled/deleted references, real graph prompt/tools and child routing.

### 3. Frontend
- Add `/agents` and navigation. Main and child cards share the screenshot's white/teal layout, counts, search, empty/error/loading states and edit/copy/delete controls.
- Provider tab groups the existing catalog. Add/edit/delete personal model endpoints and show global presets; never return keys.
- Choose an agent in chat, restore its identity from history, and open a fresh draft via the card's chat/test action. Refresh configuration after management changes.

### 4. Verify and deploy
- Run focused API/runtime regression tests, existing model/stream/skills checks, frontend tests and production build.
- Back up PostgreSQL/configured application images, apply additive migration, rebuild/recreate backend/frontend without changing dependency versions.
- Exercise actual browser CRUD, refresh, provider editing and selecting a profile; call a real configured model through a personal main profile and a configured child, retain safe evidence and screenshots.

Built-in definitions remain global presets. Personal edits affect the owner's next run; a run already admitted retains its own resolved snapshot. Model endpoints continue using the existing encryption and URL validation service.

## Completed acceptance

- PostgreSQL migration applied with a prior database backup; current source rebuilt on verified existing dependency layers. Both application containers healthy.
- 137 backend checks passed across targeted API/runtime/Skill/model/Langfuse and 16 dedicated PostgreSQL recovery/admission cases; 65 frontend tests and Vite build passed.
- Actual model executions proved personal main instructions, coordinator→personal child→calculator (42), and direct child trial. Browser-created/edited agent survived refresh and returned `UI-CONFIG-OK` through SSE. Model edit retained the encrypted key, and copy/disable actions persisted.
- Live verification script and user/operator guide: `scripts/verify_agent_management_live.py`, `docs/agent-management.md`. Local credentials and evidence stay under ignored `tmp/agent-management-live/`.
