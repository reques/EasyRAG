import assert from 'node:assert/strict'
import test from 'node:test'

globalThis.localStorage = {
  getItem() { return null },
  removeItem() {},
}

const { default: api } = await import('../src/api/index.js')

function sseResponse(payload) {
  return new Response(`data: ${JSON.stringify(payload)}\n\n`, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

test('streamChat rejects a non-function event callback instead of silently dropping events', async () => {
  globalThis.fetch = async () => sseResponse({ type: 'progress_summary', text: 'working' })

  await assert.rejects(
    () => api.streamChat('/chat/stream', {}, { signal: undefined }),
    /onEvent must be a function/,
  )
})

test('streamChat dispatches SSE payloads and forwards AbortSignal', async () => {
  const controller = new AbortController()
  const events = []
  let receivedSignal = null
  globalThis.fetch = async (_url, options) => {
    receivedSignal = options.signal
    return sseResponse({ type: 'progress_summary', text: '正在检索资料' })
  }

  await api.streamChat(
    '/chat/stream',
    { query: 'test' },
    event => events.push(event),
    { signal: controller.signal },
  )

  assert.equal(receivedSignal, controller.signal)
  assert.deepEqual(events, [{ type: 'progress_summary', text: '正在检索资料' }])
})
