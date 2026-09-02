# Claude-Inspired Frontend Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Refactor EasyRAG into a warm, editorial AI workspace inspired by Claude's official public product language without copying proprietary brand assets.

**Architecture:** Preserve the current Vue component and data-flow architecture. Replace the green visual theme with a centralized warm-neutral/clay token system, update generic brand iconography, and align auth, shell, chat, knowledge, evaluation, progress, image, focus, loading, error, and responsive states through component-local and global CSS.

**Tech Stack:** Vue 3, Vite, plain CSS, Lucide Vue icons, Node test runner.

---

### Task 1: Design tokens and application shell

**Files:**
- Modify: `frontend/src/style.css`
- Modify: `frontend/src/views/LayoutView.vue`

1. Replace the green accent scale with clay/orange tokens around `#cc785c`, warm ink around `#141413`, parchment canvas around `#faf9f5`, and warm hairlines.
2. Use a locally available Chinese serif fallback stack for editorial display headings and the existing sans stack for controls.
3. Keep primary actions ink-black; reserve clay for the brand mark, focus, selection, and small status accents.
4. Change the EasyRAG mark to a generic Lucide asterisk/starburst while retaining the EasyRAG name.
5. Verify sidebar collapse behavior at 1060px and 760px.

### Task 2: Auth and chat surfaces

**Files:**
- Modify: `frontend/src/style.css`
- Modify: `frontend/src/views/LoginView.vue`
- Modify: `frontend/src/views/RegisterView.vue`
- Modify: `frontend/src/views/ChatView.vue`

1. Restyle auth as a parchment editorial panel plus white form card, using serif headlines and restrained decoration.
2. Replace the chat greeting mark with the generic asterisk/starburst.
3. Flatten starter cards and composer elevation, using warm border contrast before shadows.
4. Align user bubbles, active deep-research state, attachment controls, focus states, and mobile layout.

### Task 3: Knowledge, evaluation, and progress surfaces

**Files:**
- Modify: `frontend/src/styles/knowledge-workspace.css`
- Modify: `frontend/src/views/EvaluationView.vue`
- Modify: `frontend/src/components/ProgressJournal.vue`

1. Apply the parchment/white/ink/clay system to catalog cards, metrics, tabs, tables, graph controls, and modals.
2. Keep primary knowledge/evaluation actions ink-black, with clay used only for selection and focus.
3. Restyle progress entries as a paper journal with clay markers and warm warning/error surfaces.
4. Preserve loading, empty, error, disabled, keyboard focus, and reduced-motion states.

### Task 4: Verification

**Files:**
- Verify all files above.

1. Run `npm run test:stream` in `frontend`.
2. Run `npm run build` in `frontend`.
3. Run `git diff --check -- frontend docs/plans`.
4. Visually inspect desktop auth, desktop chat, desktop knowledge, and 390px mobile chat.
5. Confirm no Claude/Anthropic proprietary logo or font asset was copied into the repository.
