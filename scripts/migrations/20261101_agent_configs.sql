-- Additive, idempotent. Existing agent/model definitions remain untouched.
CREATE TABLE IF NOT EXISTS agent_configs (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    slug VARCHAR(64) NOT NULL,
    name VARCHAR(80) NOT NULL,
    kind VARCHAR(16) NOT NULL,
    description VARCHAR(500) NOT NULL DEFAULT '',
    config_json TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_config_owner_slug UNIQUE(owner_id, slug)
);
CREATE INDEX IF NOT EXISTS ix_agent_configs_owner_id ON agent_configs(owner_id);
