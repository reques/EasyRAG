-- Additive and idempotent. Existing trace IDs and historical rows stay intact.
ALTER TABLE chat_streams ADD COLUMN IF NOT EXISTS local_trace_id VARCHAR(32);
ALTER TABLE chat_streams ADD COLUMN IF NOT EXISTS observability_trace_id VARCHAR(32);
ALTER TABLE chat_streams ADD COLUMN IF NOT EXISTS observability_sampled BOOLEAN;
ALTER TABLE chat_streams ADD COLUMN IF NOT EXISTS resumed_from_stream_id UUID;
ALTER TABLE execution_traces ADD COLUMN IF NOT EXISTS stream_id UUID;
ALTER TABLE execution_traces ADD COLUMN IF NOT EXISTS observability_trace_id VARCHAR(32);
CREATE UNIQUE INDEX IF NOT EXISTS uq_execution_trace_stream ON execution_traces(stream_id);
