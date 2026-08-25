<template>
  <div class="auth-page">
    <aside class="auth-intro">
      <div class="auth-intro-brand"><span><Waypoints :size="20" /></span> EasyRAG</div>
      <div class="auth-intro-copy">
        <span class="auth-kicker">YOUR KNOWLEDGE, IN MOTION</span>
        <h2>让分散的知识，<br />成为可靠的答案。</h2>
        <p>检索、分析与多智能体协作，都在同一个清晰的工作空间里。</p>
      </div>
      <div class="auth-intro-foot"><span><ShieldCheck :size="15" /></span> 数据与访问权限由你的工作区掌控</div>
    </aside>
    <div class="auth-card">
      <div class="auth-header">
        <span class="auth-logo"><Waypoints :size="23" /></span>
        <span class="auth-eyebrow">WELCOME BACK</span>
        <h1>欢迎回来</h1>
        <p>登录后继续探索你的知识空间</p>
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
          {{ loading ? '登录中…' : '登录 EasyRAG' }}
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
import { ShieldCheck, Waypoints } from 'lucide-vue-next'
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
