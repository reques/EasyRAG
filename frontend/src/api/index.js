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
  delete: (url) => http.delete(url).then((r) => r.data),
  upload: (url, formData) =>
    http.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data),
  // 获取二进制数据（PDF/图片），返回 { data: Blob, contentType: string }
  getBlob: (url) =>
    http.get(url, { responseType: 'blob' }).then((r) => ({
      data: r.data,
      contentType: r.headers['content-type'] || 'application/octet-stream',
    })),
}
