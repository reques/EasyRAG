-- Additive and idempotent. Apply before deploying the quality-loop backend.
CREATE TABLE IF NOT EXISTS message_feedback (
    id UUID PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating VARCHAR(8) NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_message_feedback_owner UNIQUE (message_id, user_id)
);
CREATE INDEX IF NOT EXISTS ix_message_feedback_message_id ON message_feedback(message_id);
CREATE INDEX IF NOT EXISTS ix_message_feedback_user_id ON message_feedback(user_id);
CREATE TABLE IF NOT EXISTS observability_scores (
    id VARCHAR(128) PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    state VARCHAR(16) NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_until TIMESTAMPTZ,
    lease_token VARCHAR(32),
    error_type VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS ix_observability_scores_user_id ON observability_scores(user_id);
CREATE INDEX IF NOT EXISTS ix_observability_scores_state ON observability_scores(state);
CREATE INDEX IF NOT EXISTS ix_observability_scores_next_attempt_at ON observability_scores(next_attempt_at);
