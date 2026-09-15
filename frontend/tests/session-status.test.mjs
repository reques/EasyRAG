import assert from 'node:assert/strict'
import test from 'node:test'
import { interruptTasks, latestContextUsage, mergeTasks, sessionUsage, shouldAutoOpen, statusArtifacts } from '../src/utils/session-status.js'

test('context uses the latest single call while conversation usage sums turns', () => {
  const messages = [
    { role: 'assistant', meta: { tokenUsage: { input_tokens: 20000, output_tokens: 200, total_tokens: 20200 }, contextUsage: { input_tokens: 10000 } } },
    { role: 'assistant', meta: { tokenUsage: { input_tokens: 50000, output_tokens: 300, total_tokens: 50300 }, contextUsage: { input_tokens: 12000 } } },
  ]
  assert.equal(sessionUsage(messages).total_tokens, 70500)
  assert.equal(latestContextUsage(messages).input_tokens, 12000)
  assert.equal(latestContextUsage([...messages, { role: 'assistant', meta: { tokenUsage: { total_tokens: 99 } } }]), null)
})

test('incremental task lists preserve completed work and accept late tasks', () => {
  const current = [{ task_id: 'a', goal: 'First', status: 'done', output: 'Result', tools: ['search'] }]
  const merged = mergeTasks(current, [{ task_id: 'a' }, { task_id: 'b', goal: 'Second' }])
  assert.equal(merged.length, 2)
  assert.equal(merged[0].status, 'done')
  assert.equal(merged[0].output, 'Result')
  assert.equal(merged[1].status, 'pending')
  assert.equal(current.length, 1)
})

test('auto-open respects manual dismissal and mobile view', () => {
  assert.equal(shouldAutoOpen({}, true), true)
  assert.equal(shouldAutoOpen({ dismissed: true }, true), false)
  assert.equal(shouldAutoOpen({ narrow: true }, true), false)
  assert.equal(shouldAutoOpen({ narrow: true, pinned: true }, true), true)
  assert.equal(shouldAutoOpen({}, false), false)
})

test('interruption stops unfinished spinners and preserves completed results', () => {
  const tasks = [{ task_id: 'a', status: 'done', output: 'Saved' }, { task_id: 'b', status: 'running' }]
  assert.deepEqual(interruptTasks(tasks, true).map(t => t.status), ['done', 'cancelled'])
  assert.deepEqual(interruptTasks(tasks, false, 'Disconnected').map(t => t.status), ['done', 'error'])
  assert.equal(tasks[1].status, 'running')
})

test('artifacts include actual outputs once, without treating tool logs as files', () => {
  const result = statusArtifacts({ artifacts: [
    { id: 'worker-a', kind: 'worker', content: 'Report', title: 'Report' },
    { kind: 'tool_result', content: 'Found 3 documents' },
  ] }, [{ task_id: 'a', output: 'Report' }, { task_id: 'b', output: 'Second report' }])
  assert.equal(result.length, 2)
  assert.deepEqual(result.map(a => a.content), ['Report', 'Second report'])
})
