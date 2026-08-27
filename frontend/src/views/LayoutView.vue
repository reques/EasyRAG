<template>
  <div class="layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-brand" title="EasyRAG 首页" @click="$router.push('/')">
        <span class="brand-logo"><Asterisk :size="20" :stroke-width="2.1" /></span>
        <span class="brand-copy">
          <span class="brand-name">EasyRAG</span>
          <small>KNOWLEDGE OS</small>
        </span>
      </div>

      <nav class="sidebar-nav">
        <span class="sidebar-section-label">工作区</span>
        <button class="nav-item primary-action" title="新建对话" @click="newChat">
          <Plus :size="17" class="nav-icon" />
          <span class="nav-text">新建对话</span>
          <span class="nav-shortcut">⌘ N</span>
        </button>
        <router-link to="/" class="nav-item" title="智能对话" exact-active-class="active">
          <MessagesSquare :size="17" class="nav-icon" />
          <span class="nav-text">对话</span>
        </router-link>
        <router-link to="/knowledge" class="nav-item" title="知识空间" active-class="active">
          <LibraryBig :size="17" class="nav-icon" />
          <span class="nav-text">知识库</span>
        </router-link>
      </nav>

      <!-- 最近对话列表 -->
      <div class="sidebar-conversations">
        <div class="conv-header"><Clock3 :size="13" /><span>最近对话</span></div>
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
        <span class="user-badge"><strong>{{ auth.username }}</strong><small>已登录</small></span>
        <button @click="auth.logout(); $router.push('/login')" class="btn-logout" title="退出登录">
          <LogOut :size="15" />
        </button>
      </div>
    </aside>

    <!-- 主区域 -->
    <main class="main">
      <header class="app-topbar">
        <div class="app-topbar-copy">
          <span class="app-topbar-name">{{ pageTitle }}</span>
          <span class="app-topbar-description">{{ pageDescription }}</span>
        </div>
        <div class="app-topbar-meta">
          <span class="system-status"><i></i> 服务正常</span>
          <span class="app-version-badge">{{ appVersion }}</span>
        </div>
      </header>
      <div class="main-content">
        <router-view v-slot="{ Component }">
          <keep-alive>
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </div>
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
import { Asterisk, Clock3, LibraryBig, LogOut, MessagesSquare, MoreHorizontal, Plus, Sparkles, Trash2 } from 'lucide-vue-next'
import { useAuthStore } from '../stores/auth'
import { useChatStore } from '../stores/chat'
import api from '../api'

const auth = useAuthStore()
const chatStore = useChatStore()
const router = useRouter()
const route = useRoute()
const appVersion = ref('v0.3.1')

const pageTitle = computed(() => route.path === '/knowledge' ? '知识空间' : '智能对话')
const pageDescription = computed(() => route.path === '/knowledge'
  ? '组织资料、检索与评估'
  : '从你的知识与工具中获得答案')

const avatarLetter = computed(() => (auth.username || '?').slice(0, 1).toUpperCase())

function selectConv(id) {
  chatStore.selectConversation(id)
  if (route.path !== '/') router.push('/')
}

function newChat() {
  chatStore.startNewConversation()
  if (route.path !== '/') router.push('/')
}

onMounted(async () => {
  chatStore.loadConversations()
  window.addEventListener('scroll', closeConvMenu, true)
  try {
    const info = await api.get('/health')
    if (info.version) appVersion.value = `v${String(info.version).replace(/^v/, '')}`
  } catch { /* 保留构建时版本兜底 */ }
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
