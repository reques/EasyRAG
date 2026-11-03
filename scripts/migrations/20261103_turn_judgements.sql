-- Additive and idempotent. Judge admission/result survives process restarts.
CREATE TABLE IF NOT EXISTS turn_judgements (
    trace_id VARCHAR(32) PRIMARY KEY REFERENCES execution_traces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    state VARCHAR(16) NOT NULL,
    model_id VARCHAR(160) NOT NULL,
    request_id VARCHAR(36) NOT NULL,
    lease_token VARCHAR(32),
    lease_until TIMESTAMPTZ,
    result_json TEXT,
    error_type VARCHAR(100),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_turn_judgements_user_id ON turn_judgements(user_id);
