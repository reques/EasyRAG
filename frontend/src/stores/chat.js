import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api'

// 刷新页面后恢复上次选中的会话
const STORAGE_KEY = 'easyrag_active_conversation'

export const useChatStore = defineStore('chat', () => {
  const conversations = ref([])
  const activeConversationId = ref(localStorage.getItem(STORAGE_KEY) || null)
  const loading = ref(false)

  async function loadConversations() {
    loading.value = true
    try {
      conversations.value = await api.get('/chat/conversations')
      // 若持久化的会话已被删除（列表里不存在），清掉避免卡在无效会话
      if (
        activeConversationId.value &&
        !conversations.value.some(c => c.id === activeConversationId.value)
      ) {
        startNewConversation()
      }
    } catch {
      conversations.value = []
    } finally {
      loading.value = false
    }
  }

  function selectConversation(id) {
    activeConversationId.value = id
    localStorage.setItem(STORAGE_KEY, id)
  }

  function startNewConversation() {
    activeConversationId.value = null
    localStorage.removeItem(STORAGE_KEY)
  }

  async function refreshAfterSend(convId) {
    // 发送消息后刷新列表（显示新标题）
    await loadConversations()
    if (!activeConversationId.value) {
      selectConversation(convId)
    }
    // 新会话标题由后端后台协程生成（LLM 语义摘要），SSE done 时可能未完成，
    // 延迟 3s 再刷一次让语义化标题就位，替换掉创建时的默认 "New Conversation"
    setTimeout(() => loadConversations(), 3000)
  }

  return {
    conversations,
    activeConversationId,
    loading,
    loadConversations,
    selectConversation,
    startNewConversation,
    refreshAfterSend,
  }
})
