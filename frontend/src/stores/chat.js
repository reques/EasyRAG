import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api'

export const useChatStore = defineStore('chat', () => {
  const conversations = ref([])
  const activeConversationId = ref(null)
  const loading = ref(false)

  async function loadConversations() {
    loading.value = true
    try {
      conversations.value = await api.get('/chat/conversations')
    } catch {
      conversations.value = []
    } finally {
      loading.value = false
    }
  }

  function selectConversation(id) {
    activeConversationId.value = id
  }

  function startNewConversation() {
    activeConversationId.value = null
  }

  async function refreshAfterSend(convId) {
    // 发送消息后刷新列表（显示新标题）
    await loadConversations()
    if (!activeConversationId.value) {
      activeConversationId.value = convId
    }
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
