/** Fold canonical snapshots and older per-delta trace records into logical events. */
export function normalizeTraceEvents(events) {
  const unique = new Map()
  for (const event of events || []) {
    if (event?.id) unique.set(event.id, event)
  }
  const ordered = [...unique.values()].sort((a, b) =>
    (a.sequence || 0) - (b.sequence || 0)
    || String(a.timestamp || '').localeCompare(String(b.timestamp || ''))
  )
  const result = []
  const streams = new Map()
  for (const event of ordered) {
    const meta = event.metadata || {}
    const key = event.type === 'reasoning_summary' && meta.source_event_id
      ? JSON.stringify([event.trace_id || '', meta.span || '', event.type, meta.source_event_id])
      : null
    const index = key ? streams.get(key) : undefined
    if (index != null) {
      const previous = result[index]
      result[index] = {
        ...event,
        id: previous.id,
        timestamp: previous.timestamp,
        sequence: previous.sequence,
        parent_id: previous.parent_id,
        output: meta.stream_mode === 'snapshot'
          ? event.output
          : String(previous.output || '') + String(event.output || ''),
      }
    } else {
      if (key) streams.set(key, result.length)
      result.push({ ...event })
    }
  }
  return result
}

const TOOL_TYPES = new Set(['tool_call', 'tool_result', 'file_operation', 'code_execution'])
const HIDDEN_STAGES = new Set(['understand', 'generate', 'generate_done', 'tool', 'tool_done', 'reason', 'model_usage'])
const valueText = value => typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value, null, 2)
const isResult = event => event.type === 'tool_result'
  || ['tool_end', 'tool_error', 'tool_done'].includes(event.metadata?.stage)
const scope = event => JSON.stringify([event.trace_id || '', event.metadata?.span || ''])
const callKey = event => event.metadata?.tool_call_id
  ? `${scope(event)}:${event.metadata.tool_call_id}` : null

/** A chronological reading view: public updates, expandable operations, then the answer. */
export function buildTraceRows(events) {
  const normalized = normalizeTraceEvents(events)
  const rows = []
  const callsById = new Map()
  const callsByKey = new Map()
  // Resolve by identity, not adjacency: parallel tool results may arrive out of order.
  for (const event of normalized) {
    if (TOOL_TYPES.has(event.type) && !isResult(event)) {
      const row = { id: event.id, event, kind: 'operation', results: [] }
      callsById.set(event.id, row)
      if (callKey(event)) callsByKey.set(callKey(event), row)
    }
  }
  for (const event of normalized) {
    if (event.type === 'agent_start' || event.type === 'final_response') continue
    if (isResult(event)) {
      const call = callsById.get(event.parent_id) || callsByKey.get(callKey(event))
      if (call) { call.results.push(event); continue }
    }
    if (event.type === 'reasoning_summary') {
      const text = valueText(event.output)
      if (text) rows.push({ id: event.id, event, kind: 'commentary', text, results: [] })
    } else if (event.type === 'planning' && HIDDEN_STAGES.has(event.metadata?.stage)) {
      continue
    } else {
      rows.push(callsById.get(event.id) || { id: event.id, event, kind: 'operation', results: [] })
    }
  }
  return rows
}

export function traceRowStatus(row, { running = false, error = '', stopped = false } = {}) {
  const events = [row.event, ...row.results]
  if (events.some(e => e.status === 'error' || e.type === 'error' || e.metadata?.is_error)) return 'error'
  if (row.results.length || row.event.status === 'completed' || row.event.metadata?.streaming === false) return 'completed'
  if (row.event.status !== 'running') return row.event.status || 'info'
  if (running) return 'running'
  return error || stopped ? 'stopped' : 'completed'
}

export { valueText as formatTraceValue }
