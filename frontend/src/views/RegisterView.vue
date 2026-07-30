<template>
  <div class="auth-page">
    <div class="auth-card">
      <div class="auth-header">
        <span class="auth-logo">🧠</span>
        <h1>创建账户</h1>
        <p>加入 EasyRAG 知识库平台</p>
      </div>
      <form @submit.prevent="handleRegister" class="auth-form">
        <label>
          <span>用户名</span>
          <input v-model="form.username" placeholder="3-64 个字符" required />
        </label>
        <label>
          <span>邮箱（选填）</span>
          <input v-model="form.email" type="email" placeholder="you@example.com" />
        </label>
        <label>
          <span>显示名称（选填）</span>
          <input v-model="form.display_name" placeholder="你的昵称" />
        </label>
        <label>
          <span>密码</span>
          <input v-model="form.password" type="password" placeholder="至少 6 位" required />
        </label>
        <p v-if="error" class="auth-error">{{ error }}</p>
        <p v-if="success" class="auth-success">{{ success }}</p>
        <button type="submit" :disabled="loading" class="btn-primary">
          {{ loading ? '注册中…' : '注 册' }}
        </button>
      </form>
      <p class="auth-footer">
        已有账户？<router-link to="/login">立即登录</router-link>
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
const form = reactive({ username: '', email: '', display_name: '', password: '' })
const loading = ref(false)
const error = ref('')
const success = ref('')

async function handleRegister() {
  error.value = ''
  success.value = ''
  if (form.password.length < 6) {
    error.value = '密码至少需要 6 个字符'
    return
  }
  loading.value = true
  try {
    await auth.register({ ...form })
    success.value = '注册成功，正在跳转…'
    setTimeout(() => router.push('/'), 800)
  } catch (e) {
    error.value = e.response?.data?.detail || '注册失败'
  } finally {
    loading.value = false
  }
}
</script>
