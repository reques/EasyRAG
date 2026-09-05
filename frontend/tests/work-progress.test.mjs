import assert from 'node:assert/strict'
import test from 'node:test'
import { appendWorkArtifact, buildHistoryWorkItems } from '../src/utils/work-progress.js'

test('progress deltas merge exactly once across intervening tool events', () => {
  let state = { artifacts: [], workItems: [] }
  const feed = (event, wid) => {
    state = appendWorkArtifact(state.artifacts, state.workItems, event, wid)
  }
  feed({ id: 'p1', kind: 'thought', content: '先核对', streaming: true }, 'w1')
  const first = state.artifacts[0]
  feed({ kind: 'tool', content: '{}', tool_call_id: 'c1' }, 'w2')
  feed({ id: 'p1', kind: 'thought', content: '付款期限。', streaming: true }, 'w3')
  feed({ id: 'p1', kind: 'thought', content: '', streaming: false }, 'w4')
  assert.equal(first.content, '先核对', 'do not mutate prior Vue state')
  assert.equal(state.artifacts.length, 2)
  assert.equal(state.workItems.length, 2)
  assert.equal(state.artifacts[0].content, '先核对付款期限。')
  assert.equal(state.workItems[0].content, state.artifacts[0].content)
  assert.equal(state.workItems[0].streaming, false)
  assert.equal(state.workItems[1].tool_call_id, 'c1')
})

test('history preserves action/observation order, repeated rounds and tool IDs', () => {
  const items = buildHistoryWorkItems({ meta: {
    intent: 'dynamic',
    steps: [{ step: 'tool', sequence: 2 }, { step: 'generate_done', sequence: 9 }],
    artifacts: [
      { kind: 'thought', content: '先检索合同', sequence: 1 },
      { kind: 'tool', tool_call_id: 'c1', sequence: 3 },
      { kind: 'tool_result', tool_call_id: 'c1', sequence: 4 },
      { kind: 'thought', content: '资料不足，换关键词补查', sequence: 5 },
      { kind: 'tool', tool_call_id: 'c2', sequence: 6 },
      { kind: 'tool_result', tool_call_id: 'c2', sequence: 7 },
      { kind: 'answer', content: '正文', sequence: 8 },
    ],
  } })
  assert.deepEqual(items.map(item => item.sequence), [1, 2, 3, 4, 5, 6, 7, 9])
  assert.equal(items[3].tool_call_id, 'c1')
  assert.equal(items[6].tool_call_id, 'c2')
  assert.ok(items.every(item => item.streaming === false))
})

test('legacy history omits answer tokens and canned dynamic startup', () => {
  const items = buildHistoryWorkItems({ meta: {
    intent: 'dynamic',
    steps: [{ step: 'understand', detail: '动态 Agent 开始处理…' }, { step: 'tool', detail: '调用 calculator' }],
    artifacts: [{ kind: 'answer', content: '1' }, { kind: 'tool_result', content: '2' }],
  } })
  assert.equal(items.length, 2)
  assert.equal(items[0].step, 'tool')
  assert.equal(items[1].kind, 'tool_result')
})
