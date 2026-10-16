# Access control and reconnectable chat implementation plan

**Goal:** Protect MCP/global legacy APIs and make network loss independent of chat execution, with explicit owner-authorized cancellation and replay.

**Architecture:** MCP configuration and the global legacy data APIs require an active administrator. The public health endpoint remains available for Compose. Normal user/knowledge APIs retain existing owner checks. A detached producer owns each chat execution; PostgreSQL stores stream identity, ownership, status and ordered events, while HTTP connections only subscribe. Duplicate request IDs cannot start duplicate executions; one active stream per conversation is enforced in the database. Cancellation is a durable explicit request, observed at model/tool boundaries. It never masquerades as a transport disconnect.

**Tech Stack:** FastAPI, SQLAlchemy async/PostgreSQL, asyncio, thread-safe cancellation context, fetch SSE/Vue, local pytest/HTTP and Node regression coverage.

## Design and boundaries

- Keep the current single API/executor deployment. This change does not introduce a distributed task queue. Durable status/heartbeats identify interrupted execution after process loss; checkpoints allow an explicit new resume, without claiming exactly-once replay of external side effects.
- Ordered replay uses authenticated GET with an event cursor. Completed runs remain readable. Refresh discovers the conversation's latest stream. A stopped browser subscription does not cancel the producer.
- A stable client request ID handles a lost initial response. In-flight duplicate requests and another active turn receive the existing stream or a conflict, never a second execution.
- Explicit cancellation marks `cancel_requested`; work already inside third-party I/O cannot be forcibly rolled back. Stop scheduling further tool/model work and acknowledge cancellation only when the owning executor drains. Preserve user/history records and prevent a late completed answer after cancellation.
- Final answer persistence must succeed before publishing `done`. Completion and cancellation contend on the same stream row lock.
- Root `tests/` remains ignored and local as requested. Existing frontend regression tests remain tracked.

## Steps

1. Reproduce missing API authorization and disconnect-induced message/checkpoint deletion with local tests.
2. Add reusable administrator dependency and apply it to MCP/global legacy routes; retain anonymous health.
3. Add stream run/event models, store and detached producer with ordered replay, duplicate admission, ownership, cancellation and stale-run handling.
4. Integrate both dynamic and deep streaming paths; add cooperative cancellation to the model/tool execution boundary and guard completion.
5. Update fetch streaming to reconnect without repeating the POST, de-duplicate cursor events, and make Stop call the cancellation endpoint. Discover/replay active or interrupted streams after reload.
6. Verify missing/invalid JWT, non-admin access, cross-user replay/cancel, network disconnect/reconnect, lost handshake retry, real explicit cancellation, duplicate requests, persistence failure, completed replay and interruption behavior.
7. Run relevant Python and frontend regressions, build frontend, review the scoped diff, then fast-forward `home` and `main` together. Preserve unrelated `.gitignore` and temporary files.

## Verification

Use a real localhost HTTP server with a deterministic stub Agent and isolated SQL database to exercise actual streaming/disconnection. Test PostgreSQL locking/index semantics against local PostgreSQL when available. The test Agent replaces external LLM/tool services; real provider behavior and irreversible side effects are outside this deterministic verification.
