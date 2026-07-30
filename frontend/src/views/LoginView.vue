<template>
  <div class="auth-page">
    <div class="auth-card">
      <div class="auth-header">
        <span class="auth-logo">🧠</span>
        <h1>EasyRAG</h1>
        <p>企业知识库智能平台</p>
      </div>
      <form @submit.prevent="handleLogin" class="auth-form">
        <label>
          <span>用户名</span>
          <input v-model="form.username" placeholder="admin" required autofocus />
        </label>
        <label>
          <span>密码</span>
          <input v-model="form.password" type="password" placeholder="••••••" required />
        </label>
        <p v-if="error" class="auth-error">{{ error }}</p>
        <button type="submit" :disabled="loading" class="btn-primary">
          {{ loading ? '登录中…' : '登 录' }}
        </button>
      </form>
      <p class="auth-footer">
        还没有账户？<router-link to="/register">立即注册</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const auth = useAuthStore()
const form = reactive({ username: '', password: '' })
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  error.value = ''
  loading.value = true
  try {
    await auth.login({ username: form.username, password: form.password })
    router.push('/')
  } catch (e) {
    error.value = e.response?.data?.detail || '登录失败，请检查用户名和密码'
  } finally {
    loading.value = false
  }
}
</script>
