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
              @click.stop="toggleConvMenu(conv.id, $event)"
              class="btn-conv-menu"
              title="更多"
            ><MoreHorizontal :size="15" /></button>
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

    <!-- 会话「⋯」弹出菜单 -->
    <Teleport to="body">
      <div
        v-if="menuConvId"
        class="conv-menu-overlay"
        @click="closeConvMenu"
        @contextmenu.prevent="closeConvMenu"
      >
        <div class="conv-menu" :style="menuStyle" @click.stop>
          <button class="conv-menu-item" :disabled="summarizing === menuConvId" @click="askSummarize">
            <Sparkles :size="14" /> {{ summarizing === menuConvId ? '生成中…' : '生成摘要' }}
          </button>
          <button class="conv-menu-item danger" @click="askDeleteConv">
            <Trash2 :size="14" /> 删除
          </button>
        </div>
      </div>
    </Teleport>

    <!-- 删除会话确认弹窗 -->
    <Teleport to="body">
      <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
        <div class="modal">
          <h3>删除对话</h3>
          <p class="delete-warning">
            确定要删除「<strong>{{ deleteTarget.title || '新的对话' }}</strong>」吗？
            该对话下的所有消息将一并删除，不可恢复。
          </p>
          <div class="modal-actions">
            <button @click="deleteTarget = null" class="btn-secondary">取消</button>
            <button @click="doDeleteConv" :disabled="deletingConv" class="btn-danger-sm">
              {{ deletingConv ? '删除中…' : '确认删除' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Brain, MessageCirclePlus, MessageSquare, LibraryBig, MoreHorizontal, Sparkles, Trash2, LogOut } from 'lucide-vue-next'
import { useAuthStore } from '../stores/auth'
import { useChatStore } from '../stores/chat'
import api from '../api'

const auth = useAuthStore()
const chatStore = useChatStore()
const router = useRouter()
const route = useRoute()

const avatarLetter = computed(() => (auth.username || '?').slice(0, 1).toUpperCase())

function selectConv(id) {
  chatStore.selectConversation(id)
  if (route.path !== '/') router.push('/')
}

function newChat() {
  chatStore.startNewConversation()
  if (route.path !== '/') router.push('/')
}

onMounted(() => {
  chatStore.loadConversations()
  window.addEventListener('scroll', closeConvMenu, true)
})
onUnmounted(() => window.removeEventListener('scroll', closeConvMenu, true))

// ── 会话「⋯」菜单 + 删除确认 ─────────────────────────────────────
const menuConvId = ref(null)
const menuStyle = ref({})
const deleteTarget = ref(null)
const deletingConv = ref(false)

function toggleConvMenu(id, e) {
  if (menuConvId.value === id) {
    closeConvMenu()
    return
  }
  const rect = e.currentTarget.getBoundingClientRect()
  menuStyle.value = {
    top: `${rect.bottom + 4}px`,
    left: `${Math.min(rect.left, window.innerWidth - 140)}px`,
  }
  menuConvId.value = id
}

function closeConvMenu() {
  menuConvId.value = null
}

function askDeleteConv() {
  const conv = chatStore.conversations.find(c => c.id === menuConvId.value)
  closeConvMenu()
  if (conv) deleteTarget.value = conv
}

// 生成摘要标题（原行内按钮功能，移入「⋯」菜单）
const summarizing = ref(null)

async function askSummarize() {
  const id = menuConvId.value
  closeConvMenu()
  if (!id) return
  summarizing.value = id
  try {
    const res = await api.post(`/chat/conversations/${id}/summarize`)
    const conv = chatStore.conversations.find(c => c.id === id)
    if (conv) conv.title = res.title
  } catch { /* ignore */ }
  finally { summarizing.value = null }
}

async function doDeleteConv() {
  if (!deleteTarget.value || deletingConv.value) return
  deletingConv.value = true
  const id = deleteTarget.value.id
  try {
    await api.delete(`/chat/conversations/${id}`)
    deleteTarget.value = null
    // 删除的是当前正在看的会话 → 跳到新对话状态（ChatView 会清空消息）
    if (chatStore.activeConversationId === id) {
      chatStore.startNewConversation()
      if (route.path !== '/') router.push('/')
    }
    await chatStore.loadConversations()
  } catch (e) {
    const detail = e.response?.data?.detail || e.message
    alert(`删除失败：${detail}`)
  } finally {
    deletingConv.value = false
  }
}
</script>
