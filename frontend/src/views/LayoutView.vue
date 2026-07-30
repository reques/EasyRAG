<template>
  <div class="layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-brand" @click="$router.push('/')">
        <span>🧠</span>
        <strong>EasyRAG</strong>
      </div>

      <nav class="sidebar-nav">
        <router-link to="/" class="nav-item" exact-active-class="active">
          <span class="nav-icon">💬</span> 对话
        </router-link>
        <router-link to="/knowledge" class="nav-item" active-class="active">
          <span class="nav-icon">📚</span> 知识库
        </router-link>
      </nav>

      <!-- 历史对话列表 -->
      <div class="sidebar-conversations">
        <div class="conv-header">
          <span>历史对话</span>
          <button @click="newChat" class="btn-new-chat" title="新对话">+</button>
        </div>
        <div class="conv-list" v-if="chatStore.conversations.length">
          <div
            v-for="conv in chatStore.conversations"
            :key="conv.id"
            :class="['conv-item', { active: chatStore.activeConversationId === conv.id }]"
            @click="selectConv(conv.id)"
          >
            <span class="conv-title">{{ conv.title || '新对话' }}</span>
            <span class="conv-date">{{ formatDate(conv.updated_at) }}</span>
          </div>
        </div>
        <div v-else-if="!chatStore.loading" class="conv-empty">
          暂无历史对话
        </div>
      </div>

      <div class="sidebar-footer">
        <span class="user-badge">{{ auth.username }}</span>
        <button @click="auth.logout(); $router.push('/login')" class="btn-logout">退出</button>
      </div>
    </aside>

    <!-- 主区域 -->
    <main class="main">
      <router-view v-slot="{ Component }">
        <keep-alive>
          <component :is="Component" />
        </keep-alive>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { useChatStore } from '../stores/chat'

const auth = useAuthStore()
const chatStore = useChatStore()
const router = useRouter()
const route = useRoute()

function selectConv(id) {
  chatStore.selectConversation(id)
  if (route.path !== '/') router.push('/')
}

function newChat() {
  chatStore.startNewConversation()
  if (route.path !== '/') router.push('/')
}

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const now = new Date()
  const diff = now - d
  if (diff < 86400000) return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  if (diff < 604800000) return ['周日','周一','周二','周三','周四','周五','周六'][d.getDay()]
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

onMounted(() => chatStore.loadConversations())
</script>
