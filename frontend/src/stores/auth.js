import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const user = ref(JSON.parse(localStorage.getItem('user') || 'null'))

  const isAuthenticated = computed(() => !!token.value)
  const username = computed(() => user.value?.username || '')

  async function login(credentials) {
    const res = await api.post('/auth/login', credentials)
    token.value = res.access_token
    user.value = { username: res.username, user_id: res.user_id }
    localStorage.setItem('token', res.access_token)
    localStorage.setItem('user', JSON.stringify(user.value))
  }

  async function register(data) {
    const res = await api.post('/auth/register', data)
    token.value = res.access_token
    user.value = { username: res.username, user_id: res.user_id }
    localStorage.setItem('token', res.access_token)
    localStorage.setItem('user', JSON.stringify(user.value))
  }

  function logout() {
    token.value = ''
    user.value = null
    localStorage.removeItem('token')
    localStorage.removeItem('user')
  }

  return { token, user, isAuthenticated, username, login, register, logout }
})
