import assert from 'node:assert/strict'
import test from 'node:test'
import { buildTraceRows, normalizeTraceEvents, traceRowStatus } from '../src/utils/agent-trace.js'

const event = (id, type, output, metadata = {}, extra = {}) => ({
  id, type, output, status: 'running', trace_id: 'run1', metadata: { span: 'main', ...metadata }, ...extra,
})
const progress = (id, output, source = 'p1', metadata = {}) => event(id, 'reasoning_summary', output, { source_event_id: source, streaming: true, ...metadata })

test('snapshots replace in place and the end marker does not add a row', () => {
  const input = [progress('p', '先', 'p1', { stream_mode: 'snapshot' }), event('c', 'tool_call', null),
    progress('p', '先查知识库', 'p1', { stream_mode: 'snapshot', streaming: false })]
  const rows = buildTraceRows(input)
  assert.deepEqual(rows.map(r => r.id), ['p', 'c'])
  assert.equal(rows[0].text, '先查知识库')
  assert.equal(traceRowStatus(rows[0], { running: true }), 'completed')
  assert.equal(input[0].output, '先')
})

test('legacy per-character records merge across intervening events, preserving separate rounds and workers', () => {
  const rows = buildTraceRows([
    progress('a', '先'), event('c', 'tool_call', null), progress('b', '查知识库'),
    progress('worker', '子任务分析', 'p1', { span: 'worker' }),
    progress('end', '', 'p1', { streaming: false }), progress('next', '再核对', 'p2'),
  ])
  assert.deepEqual(rows.map(r => r.id), ['a', 'c', 'worker', 'next'])
  assert.equal(rows[0].text, '先查知识库')
  assert.equal(rows[2].text, '子任务分析')
  assert.equal(rows[3].text, '再核对')
})

test('source IDs cannot merge different runs', () => {
  const rows = normalizeTraceEvents([progress('a', '甲'), { ...progress('b', '乙'), trace_id: 'run2' }])
  assert.equal(rows.length, 2)
})

test('parallel tool results belong to their calls regardless of arrival order', () => {
  const rows = buildTraceRows([
    event('c1', 'tool_call', null, { tool_call_id: 'call1' }),
    event('c2', 'tool_call', null, { tool_call_id: 'call2' }),
    event('r2', 'tool_result', '乙', { tool_call_id: 'call2' }),
    event('r1', 'tool_result', '甲', {}, { parent_id: 'c1' }),
  ])
  assert.equal(rows.length, 2)
  assert.equal(rows[0].results[0].output, '甲')
  assert.equal(rows[1].results[0].output, '乙')
  assert.equal(traceRowStatus(rows[0], { running: true }), 'completed')
})

test('file and command outputs are expandable results of one operation', () => {
  const rows = buildTraceRows([
    event('c', 'code_execution', null, { stage: 'tool_start', tool: 'shell' }, { input: 'echo hi' }),
    event('r', 'code_execution', 'hi', { stage: 'tool_end' }, { parent_id: 'c' }),
  ])
  assert.equal(rows.length, 1)
  assert.equal(rows[0].event.input, 'echo hi')
  assert.equal(rows[0].results[0].output, 'hi')
})

test('errors and interrupted operations never become successful completion', () => {
  const [failed] = buildTraceRows([
    event('c', 'tool_call', null), event('r', 'tool_result', '失败', { is_error: true }, { parent_id: 'c' }),
  ])
  assert.equal(traceRowStatus(failed), 'error')
  const [pending] = buildTraceRows([event('c', 'tool_call', null)])
  assert.equal(traceRowStatus(pending, { stopped: true }), 'stopped')
  assert.equal(traceRowStatus(pending, { error: '连接断开' }), 'stopped')
})

test('public progress stays in order and final response is displayed only in the answer', () => {
  const rows = buildTraceRows([
    event('root', 'agent_start', null), progress('a', '先检索'),
    event('s', 'planning', '调用工具', { stage: 'tool' }),
    event('c', 'tool_call', null), progress('b', '已有结果，整理回答', 'p2'),
    event('g', 'planning', '生成回答', { stage: 'generate' }), event('f', 'final_response', '最终答案'),
  ])
  assert.deepEqual(rows.map(r => r.kind), ['commentary', 'operation', 'commentary'])
  assert.ok(!JSON.stringify(rows).includes('最终答案'))
})
