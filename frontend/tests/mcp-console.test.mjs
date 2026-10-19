import assert from 'node:assert/strict'
import test from 'node:test'
import { loadMcpConsole } from '../src/api/mcp-console.js'

test('members load safe status without requesting administrator endpoints', async () => {
  const calls = []
  const result = await loadMcpConsole({ get: async path => {
    calls.push(path)
    assert.equal(path, '/mcp/overview')
    return { can_manage: false, servers: [{ server_id: 'shared' }] }
  } })
  assert.deepEqual(calls, ['/mcp/overview'])
  assert.equal(result.canManage, false)
  assert.deepEqual(result.definitions, [])
  assert.equal(result.servers[0].server_id, 'shared')
})

test('administrators load full settings only after server authorization', async () => {
  const calls = []
  const result = await loadMcpConsole({ get: async path => {
    calls.push(path)
    return path === '/mcp/overview' ? { can_manage: true } : { definitions: [{ server_id: 'admin-config' }], servers: [] }
  } })
  assert.deepEqual(calls, ['/mcp/overview', '/mcp/servers'])
  assert.equal(result.canManage, true)
  assert.equal(result.definitions[0].server_id, 'admin-config')
})

test('revoked access falls back to the sanitized overview without stale settings', async () => {
  const result = await loadMcpConsole({ get: async path => {
    if (path === '/mcp/overview') return { can_manage: true, servers: [{ server_id: 'safe' }] }
    throw { response: { status: 403 } }
  } })
  assert.equal(result.canManage, false)
  assert.deepEqual(result.definitions, [])
})

test('network failures surface as failures rather than an empty service list', async () => {
  await assert.rejects(() => loadMcpConsole({ get: async () => { throw new Error('offline') } }), /offline/)
})
