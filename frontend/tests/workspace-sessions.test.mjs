import assert from 'node:assert/strict'
import test from 'node:test'
import { createPinia, setActivePinia } from 'pinia'
import api from '../src/api/index.js'
import { useChatStore } from '../src/stores/chat.js'
import { groupSessions, sectionWorkspaces, selectionKey } from '../src/utils/workspace-sessions.js'

const values = new Map()
globalThis.localStorage = {
  getItem: key => values.get(key) ?? null,
  setItem: (key, value) => values.set(key, value),
  removeItem: key => values.delete(key),
}
function setup() {
  values.clear()
  values.set('user', JSON.stringify({ user_id: 'alice' }))
  setActivePinia(createPinia())
  return useChatStore()
}
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r }); return { promise, resolve } }

test('repeated new conversation clicks reuse one unsaved draft without API writes', async () => {
  const store = setup()
  let writes = 0
  api.post = async () => ({ id: `persisted-${++writes}`, workspace_id: 'project' })
  const first = await store.createConversation('project')
  const version = store.newConversationVersion
  const second = await store.createConversation('project')
  assert.equal(writes, 0)
  assert.equal(first.id, second.id)
  assert.equal(store.newConversationVersion, version)
  assert.equal(store.grouped.groups.length, 0)
  assert.equal(store.conversations.length, 0)
  assert.equal(store.draft.workspace_id, 'project')
})

test('selecting a persisted conversation discards the unsaved draft', async () => {
  const store = setup()
  api.post = async () => ({ id: 'unexpected-write' })
  await store.createConversation('project')
  store.selectConversation('existing')
  assert.equal(store.draft, null)
  assert.equal(store.activeConversationId, 'existing')
  assert.equal(store.conversations.length, 0)
})

test('groups preserve session identities and orphaned histories remain visible', () => {
  const old = { id: 'legacy', workspace_id: null }
  const missing = { id: 'orphan', workspace_id: 'removed' }
  const conv = { id: 'a', session_id: 'a', workspace_id: 'project' }
  const result = groupSessions([{ id: 'project', name: 'Project' }, { id: 'empty', name: 'Empty' }], [conv, old, missing])
  assert.deepEqual(result.groups[0].conversations, [conv])
  assert.equal(result.groups[1].conversations.length, 0)
  assert.deepEqual(result.ungrouped.conversations, [old, missing])
})

test('loads all pages and restores a selected session beyond the first 200', async () => {
  values.clear()
  values.set('user', JSON.stringify({ user_id: 'alice' }))
  values.set(selectionKey('alice'), 'oldest')
  setActivePinia(createPinia())
  const store = useChatStore()
  const first = Array.from({ length: 200 }, (_, i) => ({ id: String(i), workspace_id: 'w' }))
  api.get = async (path, params) => path.endsWith('workspace-sections') ? [] : path.endsWith('workspaces') ? [{ id: 'w' }] : params.offset === 0 ? first : [{ id: 'oldest', workspace_id: 'w' }]
  await store.loadConversations()
  assert.equal(store.conversations.length, 201)
  assert.equal(store.activeConversationId, 'oldest')
  assert.equal(store.activeWorkspaceId, 'w')
})

test('failed loads preserve visible histories and selection for retry', async () => {
  const store = setup()
  store.conversations = [{ id: 'kept' }]
  store.selectConversation('kept')
  api.get = async () => { throw new Error('offline') }
  await store.loadConversations()
  assert.equal(store.conversations[0].id, 'kept')
  assert.equal(store.activeConversationId, 'kept')
  assert.ok(store.loadError)
})

test('a late stream handshake registers a submitted conversation without switching the current view', async () => {
  const store = setup()
  store.createConversation('project')
  store.selectConversation('other')
  store.registerConversation('new', 'project', 'alice')
  assert.equal(store.activeConversationId, 'other')
  assert.equal(store.conversations[0].id, 'new')
  assert.equal(store.conversations[0].has_messages, true)
})

test('new drafts replace rather than accumulate across workspaces and after sending content', () => {
  const store = setup()
  store.workspaces = [{ id: 'a' }, { id: 'b' }]
  const first = store.createConversation('a')
  const second = store.createConversation('b')
  assert.notEqual(first.id, second.id)
  assert.equal(store.grouped.groups[0].conversations.length, 0)
  assert.equal(store.grouped.groups[1].conversations.length, 1)
  const version = store.newConversationVersion
  store.draft.has_content = true
  const third = store.createConversation('b')
  assert.notEqual(second.id, third.id)
  assert.equal(store.newConversationVersion, version + 1)
  assert.equal(store.conversations.length, 0)
})

test('archived and legacy empty sessions are hidden while a draft is shown', () => {
  const result = groupSessions([{ id: 'w' }], [
    { id: 'empty', workspace_id: 'w', has_messages: false },
    { id: 'archived', workspace_id: 'w', has_messages: true, archived_at: '2026-10-18' },
    { id: 'draft', workspace_id: 'w', has_messages: false, is_draft: true },
    { id: 'saved', workspace_id: 'w', has_messages: true },
  ])
  assert.deepEqual(result.groups[0].conversations.map(c => c.id), ['draft', 'saved'])
})

test('legacy empty selected sessions become one local draft instead of a persisted sidebar row', async () => {
  const store = setup()
  store.selectConversation('empty')
  api.get = async path => path.endsWith('workspace-sections') ? [] : path.endsWith('workspaces') ? [{ id: 'w' }] : [{ id: 'empty', workspace_id: 'w', has_messages: false }]
  await store.loadConversations()
  assert.equal(store.activeConversationId, null)
  assert.equal(store.draft.workspace_id, 'w')
  assert.equal(store.grouped.groups[0].conversations.length, 1)
  assert.equal(store.grouped.groups[0].conversations[0].is_draft, true)
  assert.equal(values.get(selectionKey('alice')), undefined)
})

test('pinned projects appear once and return to their section after unpinning', () => {
  const groups = [{ id: 'a', is_pinned: true, section_id: 's' }, { id: 'b', section_id: 's' }, { id: 'c', section_id: 'removed' }]
  const buckets = sectionWorkspaces(groups, [{ id: 's', name: 'Work' }])
  assert.deepEqual(buckets.map(b => b.groups.map(g => g.id)), [['a'], ['c'], ['b']])
  const unpinned = sectionWorkspaces(groups.map(g => ({ ...g, is_pinned: false })), [{ id: 's' }])
  assert.deepEqual(unpinned.map(b => b.groups.map(g => g.id)), [['c'], ['a', 'b']])
})

test('account changes discard stale loads and reset selection by user', async () => {
  const store = setup()
  store.selectConversation('alice-private')
  const response = deferred()
  api.get = () => response.promise
  const load = store.loadConversations()
  values.set('user', JSON.stringify({ user_id: 'bob' }))
  store.resetForUser()
  response.resolve([{ id: 'alice-private' }])
  await load
  assert.deepEqual(store.conversations, [])
  assert.equal(store.activeConversationId, null)
  assert.equal(values.get(selectionKey('alice')), 'alice-private')
  assert.equal(values.get(selectionKey('bob')), undefined)
})

test('late updates and workspace creation never populate another account', async () => {
  const store = setup()
  const response = deferred()
  api.patch = () => response.promise
  api.post = () => response.promise
  const update = store.updateConversation('a', { title: 'private' })
  const create = store.createWorkspace('private')
  values.set('user', JSON.stringify({ user_id: 'bob' }))
  store.resetForUser()
  response.resolve({ id: 'a', name: 'private' })
  assert.equal(await update, null)
  assert.equal(await create, null)
  assert.deepEqual(store.workspaces, [])
})
