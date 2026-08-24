import axios from 'axios'

const http = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,
})

// 自动附加 JWT token
http.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 401 自动跳转登录
http.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  },
)

export default {
  get: (url, params) => http.get(url, { params }).then((r) => r.data),
  post: (url, data) => http.post(url, data).then((r) => r.data),
  put: (url, data) => http.put(url, data).then((r) => r.data),
  patch: (url, data) => http.patch(url, data).then((r) => r.data),
  delete: (url) => http.delete(url).then((r) => r.data),
  upload: (url, formData, onUploadProgress) =>
    http.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      ...(onUploadProgress ? { onUploadProgress } : {}),
    }).then((r) => r.data),
  // 获取二进制数据（PDF/图片），返回 { data: Blob, contentType: string }
  getBlob: (url) =>
    http.get(url, { responseType: 'blob' }).then((r) => ({
      data: r.data,
      contentType: r.headers['content-type'] || 'application/octet-stream',
    })),

  // SSE 流式对话：fetch + ReadableStream 逐事件回调。
  // axios 不支持流式响应, 这里用原生 fetch 读 text/event-stream。
  // onEvent(payload) 每个 SSE data 事件回调一次, payload 为解析后的 JSON。
  // options.signal: AbortController.signal，用于"停止生成"（终止当前对话轮）。
  async streamChat(url, body, onEvent, options = {}) {
    const token = localStorage.getItem('token')
    const resp = await fetch(`/api/v1${url}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
      signal: options.signal || undefined,
    })
    if (resp.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
      throw new Error('Unauthorized')
    }
    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      let detail = text
      try {
        detail = JSON.parse(text)?.detail || text
      } catch { /* 非 JSON 错误响应保持原文 */ }
      throw new Error(detail || `HTTP ${resp.status}`)
    }
    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    // SSE 事件以 \n\n 分隔, 逐行解析 "data: {...}"
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        for (const line of rawEvent.split('\n')) {
          if (line.startsWith('data:')) {
            const data = line.slice(5).trim()
            if (!data) continue
            try {
              onEvent(JSON.parse(data))
            } catch { /* 忽略非 JSON 行 */ }
          }
        }
      }
    }
  },
}
