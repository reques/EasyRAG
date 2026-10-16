// A browser connection subscribes to a durable execution. Aborting this reader
// detaches it; cancellation is an explicit authenticated server request.
class FatalStreamError extends Error {}

function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    signal?.throwIfAborted()
    const stop = () => { clearTimeout(timer); reject(signal.reason) }
    const timer = setTimeout(() => { signal?.removeEventListener('abort', stop); resolve() }, ms)
    signal?.addEventListener('abort', stop, { once: true })
  })
}

export async function streamChat(url, body, onEvent, options = {}) {
  if (typeof onEvent !== 'function') throw new TypeError('streamChat onEvent must be a function')
  const request = { ...body, request_id: body?.request_id || crypto.randomUUID() }
  let streamId = options.streamId || null
  let cursor = options.after || 0
  let attempt = 0
  let terminal = false
  while (!terminal) {
    options.signal?.throwIfAborted()
    let reader
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(streamId
        ? `/api/v1/chat/streams/${encodeURIComponent(streamId)}/events?after=${cursor}`
        : `/api/v1${url}`, {
        method: streamId ? 'GET' : 'POST',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        ...(streamId ? {} : { body: JSON.stringify(request) }),
        signal: options.signal,
      })
      if (response.status === 401) {
        localStorage.removeItem('token')
        localStorage.removeItem('user')
        window.location.href = '/login'
        throw new FatalStreamError('Unauthorized')
      }
      if (!response.ok) {
        let detail = await response.text()
        try { detail = JSON.parse(detail)?.detail || detail } catch { /* plain text */ }
        if (typeof detail === 'object') detail = detail.message || JSON.stringify(detail)
        // Only transient gateway failures can be retried safely with this ID.
        if ([502, 503, 504].includes(response.status)) throw new Error(detail || `HTTP ${response.status}`)
        throw new FatalStreamError(detail || `HTTP ${response.status}`)
      }
      streamId = response.headers.get('X-Chat-Stream-Id') || streamId
      const conversationId = response.headers.get('X-Conversation-Id')
      try {
        options.onStream?.({ streamId, conversationId })
        options.onConnection?.('connected')
      } catch (error) { throw new FatalStreamError(error.message, { cause: error }) }
      reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (!terminal) {
        const { value, done } = await reader.read()
        if (done) break
        buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n')
        let end
        while ((end = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, end)
          buffer = buffer.slice(end + 2)
          const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n')
          if (!data) continue
          let event
          try { event = JSON.parse(data) } catch { continue }
          if (event.sequence && event.sequence <= cursor) continue
          try { onEvent(event) } catch (error) { throw new FatalStreamError(error.message, { cause: error }) }
          cursor = event.sequence || cursor
          attempt = 0
          terminal = ['done', 'cancelled', 'error'].includes(event.type)
          if (terminal) break
        }
      }
      // Older endpoints without execution headers retain their one-shot contract.
      if (!streamId) return
      if (!terminal) throw new Error('Stream disconnected before completion')
    } catch (error) {
      if (options.signal?.aborted) throw options.signal.reason
      if (error instanceof FatalStreamError) throw error
      options.onConnection?.('reconnecting')
      const delay = options.retryDelay ?? Math.min(3000, 300 * (2 ** Math.min(attempt++, 4)))
      await wait(delay, options.signal)
    } finally {
      if (reader) {
        await reader.cancel().catch(() => {})
        reader.releaseLock()
      }
    }
  }
}
