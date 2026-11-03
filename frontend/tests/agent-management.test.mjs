import test from 'node:test'
import assert from 'node:assert/strict'
import { filterAgents, groupProviders } from '../src/utils/agent-management.js'

test('provider grouping keeps built-in and personal model identities', () => {
  const models = [{id:'global',provider:'DeepSeek',source:'builtin'}, {id:'custom:one',provider:'deepseek',source:'custom'}, {id:'local',provider:'Ollama'}]
  const grouped = groupProviders(models)
  assert.equal(grouped.length,2)
  assert.deepEqual(grouped[0].models.map(m => m.id),['global','custom:one'])
  assert.equal(models.length,3)
})
test('agent search respects main/child sections and finds names and descriptions', () => {
  const items = [{id:'one',kind:'main',name:'研究助手',slug:'research',description:'资料核验'}, {id:'two',kind:'subagent',name:'Research child',slug:'child',description:'计算'}]
  assert.deepEqual(filterAgents(items,' RESEARCH ','main').map(a => a.id),['one'])
  assert.deepEqual(filterAgents(items,'计算','subagent').map(a => a.id),['two'])
  assert.equal(filterAgents(items,'missing','main').length,0)
})
