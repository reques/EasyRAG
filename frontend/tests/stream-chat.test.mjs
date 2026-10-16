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

function journalResponse(events) {
  return new Response(events.map(event => `id: ${event.sequence}\ndata: ${JSON.stringify(event)}\n\n`).join(''), {
    headers: { 'X-Chat-Stream-Id': 'run-1', 'X-Conversation-Id': 'conversation-1' },
  })
}

test('unexpected EOF reconnects after the last sequence and suppresses duplicate output', async () => {
  const requests = []
  const delivered = []
  const states = []
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init })
    return requests.length === 1
      ? journalResponse([{ type: 'delta', content: 'first', sequence: 1 }])
      : journalResponse([
        { type: 'delta', content: 'duplicate', sequence: 1 },
        { type: 'delta', content: 'last', sequence: 2 },
        { type: 'done', content: 'firstlast', sequence: 3 },
      ])
  }
  await api.streamChat('/chat/stream', { query: 'hello' }, event => delivered.push(event), {
    retryDelay: 0, onConnection: state => states.push(state),
  })
  assert.equal(requests.length, 2)
  assert.equal(requests[1].url, '/api/v1/chat/streams/run-1/events?after=1')
  assert.equal(requests[1].init.method, 'GET')
  assert.deepEqual(delivered.map(e => e.sequence), [1, 2, 3])
  assert.deepEqual(states, ['connected', 'reconnecting', 'connected'])
})

test('a lost initial handshake retries the same request UUID and request body', async () => {
  const bodies = []
  globalThis.fetch = async (_url, init) => {
    bodies.push(init.body)
    if (bodies.length === 1) throw new TypeError('Failed to fetch')
    return journalResponse([{ type: 'done', sequence: 1 }])
  }
  await api.streamChat('/chat/stream', { query: 'hello' }, () => {}, { retryDelay: 0 })
  assert.equal(bodies[0], bodies[1])
  assert.match(JSON.parse(bodies[0]).request_id, /^[a-f0-9-]{36}$/)
})

test('refresh attachment uses GET without creating another execution', async () => {
  let request
  let handshake
  globalThis.fetch = async (url, init) => {
    request = { url, init }
    return journalResponse([{ type: 'done', sequence: 3 }])
  }
  await api.streamChat('/chat/stream', {}, () => {}, {
    streamId: 'run-1', after: 2, onStream: value => { handshake = value },
  })
  assert.equal(request.url, '/api/v1/chat/streams/run-1/events?after=2')
  assert.equal(request.init.method, 'GET')
  assert.equal(request.init.body, undefined)
  assert.deepEqual(handshake, { streamId: 'run-1', conversationId: 'conversation-1' })
})

test('explicit server cancellation is terminal without reconnecting', async () => {
  let calls = 0
  const events = []
  globalThis.fetch = async () => {
    calls++
    return journalResponse([{ type: 'cancelled', sequence: 1 }])
  }
  await api.streamChat('/chat/stream', {}, event => events.push(event))
  assert.equal(calls, 1)
  assert.equal(events[0].type, 'cancelled')
})

test('aborting a reader never sends a server cancellation request', async () => {
  const controller = new AbortController()
  const requests = []
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init })
    return journalResponse([{ type: 'delta', content: 'partial', sequence: 1 }])
  }
  await assert.rejects(() => api.streamChat('/chat/stream', {}, () => controller.abort(), {
    signal: controller.signal, retryDelay: 0,
  }), { name: 'AbortError' })
  assert.equal(requests.length, 1)
  assert.equal(requests[0].url, '/api/v1/chat/stream')
})

test('permission failures and callback failures are surfaced without retrying', async () => {
  let calls = 0
  globalThis.fetch = async () => { calls++; return new Response('{"detail":"Forbidden"}', { status: 403 }) }
  await assert.rejects(() => api.streamChat('/chat/stream', {}, () => {}, { retryDelay: 0 }), /Forbidden/)
  assert.equal(calls, 1)
  globalThis.fetch = async () => { calls++; return journalResponse([{ type: 'delta', sequence: 1 }]) }
  await assert.rejects(() => api.streamChat('/chat/stream', {}, () => { throw new Error('render failed') }, { retryDelay: 0 }), /render failed/)
  assert.equal(calls, 2)
})

test('transient gateway errors retain the original request identity', async () => {
  const bodies = []
  globalThis.fetch = async (_url, init) => {
    bodies.push(init.body)
    return bodies.length === 1 ? new Response('temporarily down', { status: 503 }) : journalResponse([{ type: 'done', sequence: 1 }])
  }
  await api.streamChat('/chat/stream', { request_id: 'stable-request' }, () => {}, { retryDelay: 0 })
  assert.equal(bodies[0], bodies[1])
  assert.equal(JSON.parse(bodies[0]).request_id, 'stable-request')
})

test('split UTF-8 and CRLF boundaries preserve complete SSE payloads', async () => {
  const encoder = new TextEncoder()
  const payload = { type: 'done', content: '完成', sequence: 1 }
  const bytes = encoder.encode(`data: ${JSON.stringify(payload)}\r\n\r\n`)
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]))
      controller.close()
    },
  }), { headers: { 'X-Chat-Stream-Id': 'run-1' } })
  const received = []
  await api.streamChat('/chat/stream', {}, value => received.push(value))
  assert.deepEqual(received, [payload])
})
