<template>
  <div class="layout">
    <!-- 侧边栏: Yuxi 式纵向文字导航 -->
    <aside class="sidebar">
      <div class="sidebar-brand" @click="$router.push('/')">
        <span class="brand-logo"><Brain :size="16" /></span>
        <span class="brand-name">EasyRAG</span>
      </div>

      <nav class="sidebar-nav">
        <button class="nav-item primary-action" @click="newChat">
          <MessageCirclePlus :size="16" class="nav-icon" />
          <span class="nav-text">新建对话</span>
        </button>
        <router-link to="/" class="nav-item" exact-active-class="active">
          <MessageSquare :size="16" class="nav-icon" />
          <span class="nav-text">对话</span>
        </router-link>
        <router-link to="/knowledge" class="nav-item" active-class="active">
          <LibraryBig :size="16" class="nav-icon" />
          <span class="nav-text">知识库</span>
        </router-link>
      </nav>

      <!-- 最近对话列表 -->
      <div class="sidebar-conversations">
        <div class="conv-header"><span>最近</span></div>
        <div class="conv-list" v-if="chatStore.conversations.length">
          <div
            v-for="conv in chatStore.conversations"
            :key="conv.id"
            :class="['conv-item', { active: chatStore.activeConversationId === conv.id }]"
            @click="selectConv(conv.id)"
          >
            <span class="conv-title">{{ conv.title || '新的对话' }}</span>
            <button
              @click.stop="summarizeConv(conv.id)"
              class="btn-summarize"
              title="生成摘要"
              :disabled="summarizing === conv.id"
            ><Sparkles :size="14" /></button>
          </div>
        </div>
        <div v-else-if="!chatStore.loading" class="conv-empty">
          暂无对话历史
        </div>
      </div>

      <div class="sidebar-footer">
        <span class="user-avatar">{{ avatarLetter }}</span>
        <span class="user-badge">{{ auth.username }}</span>
        <button @click="auth.logout(); $router.push('/login')" class="btn-logout" title="退出登录">
          <LogOut :size="15" />
        </button>
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
import { onMounted, ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Brain, MessageCirclePlus, MessageSquare, LibraryBig, Sparkles, LogOut } from 'lucide-vue-next'
import { useAuthStore } from '../stores/auth'
import { useChatStore } from '../stores/chat'
import api from '../api'

const auth = useAuthStore()
const chatStore = useChatStore()
const router = useRouter()
const route = useRoute()
const summarizing = ref(null)

const avatarLetter = computed(() => (auth.username || '?').slice(0, 1).toUpperCase())

function selectConv(id) {
  chatStore.selectConversation(id)
  if (route.path !== '/') router.push('/')
}

function newChat() {
  chatStore.startNewConversation()
  if (route.path !== '/') router.push('/')
}

onMounted(() => chatStore.loadConversations())

async function summarizeConv(id) {
  summarizing.value = id
  try {
    const res = await api.post(`/chat/conversations/${id}/summarize`)
    // 更新本地列表中的标题
    const conv = chatStore.conversations.find(c => c.id === id)
    if (conv) conv.title = res.title
  } catch { /* ignore */ }
  finally { summarizing.value = null }
}
</script>
