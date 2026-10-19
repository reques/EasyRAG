<template>
  <div class="layout" :class="{ 'is-sidebar-open': sidebarOpen }">
    <button v-if="sidebarOpen" class="sidebar-backdrop" aria-label="关闭侧边栏" @click="sidebarOpen = false"></button>
    <aside class="sidebar">
      <div class="sidebar-brand" title="EasyRAG 首页" @click="$router.push('/')">
        <span class="brand-logo"><Asterisk :size="20" :stroke-width="2.1" /></span>
        <span class="brand-copy"><span class="brand-name">EasyRAG</span><small>KNOWLEDGE OS</small></span>
      </div>
      <nav class="sidebar-nav">
        <button class="nav-item primary-action" title="在最近中新建会话" @click="sessions?.newChat(null)"><Plus :size="17" class="nav-icon" /><span class="nav-text">新建会话</span></button>
        <router-link :to="{ path: '/', query: chatStore.activeConversationId ? { session: chatStore.activeConversationId } : {} }" class="nav-item" title="智能对话" exact-active-class="active"><MessagesSquare :size="17" class="nav-icon" /><span class="nav-text">对话</span></router-link>
        <router-link to="/knowledge" class="nav-item" title="知识空间" active-class="active"><LibraryBig :size="17" class="nav-icon" /><span class="nav-text">知识库</span></router-link>
        <router-link to="/memory" class="nav-item" title="记忆管理" active-class="active"><BrainCircuit :size="17" class="nav-icon" /><span class="nav-text">记忆</span></router-link>
        <router-link to="/mcp" class="nav-item" title="MCP 工具中心" active-class="active"><Blocks :size="17" class="nav-icon" /><span class="nav-text">MCP 工具</span></router-link>
      </nav>
      <div class="sidebar-conversations"><WorkspaceSessions ref="sessions" @select="showChat" /></div>
      <div class="sidebar-footer">
        <span class="user-avatar">{{ avatarLetter }}</span>
        <span class="user-badge"><strong>{{ auth.username }}</strong><small>已登录</small></span>
        <button class="btn-logout" title="退出登录" @click="logout"><LogOut :size="15" /></button>
      </div>
    </aside>
    <main class="main">
      <header class="app-topbar">
        <button type="button" class="sidebar-toggle" title="工作区与会话" aria-label="工作区与会话" :aria-expanded="sidebarOpen" @click="sidebarOpen = !sidebarOpen"><PanelLeft :size="18" /></button>
        <div class="app-topbar-copy session-breadcrumb">
          <span v-if="route.path === '/' && chatStore.activeConversation" class="session-workspace-name"><Folder :size="14" /><span>{{ currentWorkspaceName }}</span><span class="session-breadcrumb-divider">/</span></span>
          <span class="app-topbar-name">{{ pageTitle }}</span>
          <span v-if="route.path !== '/' || !chatStore.activeConversationId" class="app-topbar-description">{{ pageDescription }}</span>
        </div>
        <div class="app-topbar-meta">
          <button v-if="route.path === '/' && chatStore.activeConversationId" type="button" class="session-id-badge" :title="`Session ID: ${chatStore.activeConversationId}（点击复制）`" @click="copySession"><Copy :size="12" /><span>{{ copied ? '已复制' : `Session ${chatStore.activeConversationId.slice(0, 8)}` }}</span></button>
          <span class="app-version-badge">{{ appVersion }}</span>
        </div>
      </header>
      <div class="main-content"><router-view v-slot="{ Component }"><keep-alive><component :is="Component" /></keep-alive></router-view></div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Asterisk, Blocks, BrainCircuit, Copy, Folder, LibraryBig, LogOut, MessagesSquare, PanelLeft, Plus } from 'lucide-vue-next'
import WorkspaceSessions from '../components/WorkspaceSessions.vue'
import { useAuthStore } from '../stores/auth'
import { useChatStore } from '../stores/chat'
import api from '../api'

const auth = useAuthStore()
const chatStore = useChatStore()
chatStore.resetForUser()
const router = useRouter()
const route = useRoute()
const sessions = ref(null)
const sidebarOpen = ref(false)
const copied = ref(false)
const appVersion = ref('v0.3.1')
const avatarLetter = computed(() => (auth.username || '?').slice(0, 1).toUpperCase())
const currentWorkspaceName = computed(() => chatStore.workspaces.find(w => w.id === chatStore.activeWorkspaceId)?.name || '最近')
const pageTitle = computed(() => ({ '/knowledge': '知识空间', '/memory': '记忆管理', '/mcp': 'MCP 工具中心' }[route.path] || chatStore.activeConversation?.title || '智能对话'))
const pageDescription = computed(() => ({ '/knowledge': '组织资料、检索与评估', '/memory': '查看和修正 Agent 记住的信息', '/mcp': '管理远程与本地扩展工具能力' }[route.path] || '从你的知识与工具中获得答案'))

watch(() => [route.path, route.query.session], ([path, id], previous = []) => {
  if (path !== '/') { sidebarOpen.value = false; return }
  if (typeof id === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
    if (id.toLowerCase() !== chatStore.activeConversationId) chatStore.selectConversation(id.toLowerCase())
  } else if (previous[0] === '/' && previous[1] && !id && chatStore.activeConversationId) chatStore.startNewConversation()
}, { immediate: true })
watch(() => [chatStore.activeConversationId, route.path], ([id]) => {
  copied.value = false
  if (route.path !== '/' || (route.query.session || null) === id) return
  const query = { ...route.query }
  if (id) query.session = id
  else delete query.session
  router.replace({ path: '/', query })
}, { immediate: true })

function showChat() {
  sidebarOpen.value = false
  router.push({ path: '/', query: chatStore.activeConversationId ? { session: chatStore.activeConversationId } : {} })
}
function logout() { auth.logout(); chatStore.resetForUser(); router.push('/login') }
async function copySession() {
  try { await navigator.clipboard.writeText(chatStore.activeConversationId); copied.value = true }
  catch { copied.value = false }
}
onMounted(async () => {
  void chatStore.loadConversations()
  try { const info = await api.get('/health'); if (info.version) appVersion.value = `v${String(info.version).replace(/^v/, '')}` } catch { /* use build version */ }
})
</script>

<style scoped>
.sidebar-conversations { min-height: 0; overflow: hidden; }
.session-breadcrumb { min-width: 0; flex: 1; }
.app-topbar-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-workspace-name { display: flex; align-items: center; gap: 8px; color: var(--gray-500); font-size: 12px; min-width: 0; max-width: 180px; flex-shrink: 0; }
.session-workspace-name > span:first-of-type { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-breadcrumb-divider { padding-inline: 3px; color: var(--gray-300); }
.session-id-badge { display: flex; align-items: center; gap: 6px; padding: 6px 9px; border: 1px solid var(--gray-150); border-radius: 6px; background: transparent; color: var(--gray-500); font: 10px ui-monospace, monospace; white-space: nowrap; cursor: pointer; }
.session-id-badge:hover { color: var(--gray-950); border-color: var(--gray-300); }
.sidebar-toggle { display: none; align-items: center; justify-content: center; width: 28px; height: 28px; padding: 0; margin-right: 10px; border: 0; border-radius: 5px; background: transparent; color: var(--gray-600); cursor: pointer; }
.sidebar-backdrop { display: none; }
@media (max-width: 1060px) {
  .sidebar-toggle { display: inline-flex; flex-shrink: 0; }
  .is-sidebar-open .sidebar { position: fixed; top: 0; bottom: 0; left: 0; z-index: 110; width: min(300px, 85vw); padding-inline: 13px; }
  .is-sidebar-open .sidebar-brand { justify-content: flex-start; }
  .is-sidebar-open .brand-copy, .is-sidebar-open .user-badge { display: flex; }
  .is-sidebar-open .nav-text, .is-sidebar-open .btn-logout { display: block; }
  .is-sidebar-open .sidebar-conversations { display: flex; }
  .is-sidebar-open .sidebar-nav { align-items: stretch; }
  .is-sidebar-open .nav-item { width: auto; justify-content: flex-start; padding: 0 12px; }
  .is-sidebar-open .sidebar-footer { justify-content: flex-start; padding: 0 8px; }
  .is-sidebar-open .sidebar-backdrop { display: block; position: fixed; inset: 0; z-index: 109; border: 0; background: rgba(20, 20, 20, .3); }
}
@media (max-width: 760px) {
  .session-workspace-name { max-width: 90px; font-size: 11px; }
  .session-breadcrumb { gap: 5px; }
  .app-topbar-name { font-size: 13px; }
  .session-id-badge span { display: none; }
  .app-version-badge { display: none; }
}
</style>
