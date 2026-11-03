import test from 'node:test'
import assert from 'node:assert/strict'
import {
  observationRows,
  comparableRuns,
  safeHttpUrl,
  formatMs,
  formatMetric,
  traceFilters,
} from '../src/utils/observability.js'

test('waterfall keeps parents first and handles missing parents and cycles', () => {
  const rows = observationRows([
    {
      id: 'child',
      parent_id: 'root',
      started_at: '2026-11-03T00:00:01Z',
      completed_at: '2026-11-03T00:00:02Z',
    },
    {
      id: 'root',
      started_at: '2026-11-03T00:00:00Z',
      completed_at: '2026-11-03T00:00:03Z',
    },
    { id: 'orphan', parent_id: 'missing' },
    { id: 'a', parent_id: 'b' },
    { id: 'b', parent_id: 'a' },
  ])
  assert.equal(rows.length, 5)
  assert.equal(rows[0].id, 'root')
  assert.equal(rows[1].depth, 1)
  assert.equal(rows[1].duration, 1000)
  assert.ok(rows.every((r) => r.offset == null || Number.isFinite(r.offset)))
  assert.equal(rows.find((r) => r.id === 'orphan').duration, null)
})
test('missing timings and metrics remain unknown, zero stays visible', () => {
  assert.equal(formatMs(null), '—')
  assert.equal(formatMetric(undefined), '—')
  assert.equal(formatMetric(0), '0.000')
  assert.equal(
    observationRows([
      { id: 'one', started_at: 'invalid', metadata: { durationMs: 500 } },
    ])[0].duration,
    500,
  )
})
test('experiment comparison explains snapshot and knowledge base differences', () => {
  const one = {
    knowledge_base_id: 'kb',
    observability: { dataset_id: 'snapshot' },
  }
  assert.match(comparableRuns([one, one]), /一致/)
  assert.match(
    comparableRuns([one, { ...one, observability: { dataset_id: 'other' } }]),
    /不同/,
  )
  assert.match(
    comparableRuns([one, { ...one, observability: null }]),
    /无法确认/,
  )
  assert.match(
    comparableRuns([one, { ...one, knowledge_base_id: 'other' }]),
    /不同知识库/,
  )
})
test('external experiment links reject credentials and non-http schemes', () => {
  assert.equal(safeHttpUrl('javascript:alert(1)'), '')
  assert.equal(safeHttpUrl('http://user:secret@localhost:3000'), '')
  assert.equal(
    safeHttpUrl('http://localhost:3000/project/test'),
    'http://localhost:3000/project/test',
  )
})

test('empty optional UUID filters are omitted from API queries and explicit values survive', () => {
  assert.deepEqual(
    traceFilters({
      window: '7d',
      conversation_id: '',
      q: '',
      status: '',
      page: 1,
    }),
    { window: '7d', page: 1 },
  )
  assert.deepEqual(
    traceFilters({ conversation_id: 'owner-conversation', q: 'budget' }),
    { conversation_id: 'owner-conversation', q: 'budget' },
  )
})
