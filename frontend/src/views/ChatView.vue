<template>
  <div class="chat-view">
    <!-- 顶栏 -->
    <header class="chat-header">
      <h2>💬 智能对话</h2>
      <div class="header-actions">
        <button @click="newConversation" class="btn-secondary">+ 新对话</button>
      </div>
    </header>

    <!-- 消息列表 -->
    <div class="chat-messages" ref="msgContainer">
      <div v-if="messages.length === 0" class="chat-empty">
        <span class="empty-icon">🧠</span>
        <h3>欢迎使用 EasyRAG</h3>
        <p>输入问题开始对话，Agent 将基于知识库为你解答</p>
      </div>

      <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.role]">
        <div class="message-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
        <div class="message-body">
          <div class="message-text">{{ msg.content }}</div>
          <div v-if="msg.meta" class="message-meta">
            <span v-if="msg.meta.intent">意图: {{ msg.meta.intent }}</span>
            <span v-if="msg.meta.elapsed">耗时: {{ msg.meta.elapsed }}s</span>
          </div>
        </div>
      </div>

      <div v-if="sending" class="message assistant">
        <div class="message-avatar">🤖</div>
        <div class="message-body">
          <div class="message-text typing">思考中<span>.</span><span>.</span><span>.</span></div>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="chat-input">
      <textarea
        v-model="input"
        @keydown.enter.exact.prevent="send"
        placeholder="输入你的问题，Enter 发送…"
        rows="1"
        ref="inputEl"
      ></textarea>
      <button @click="send" :disabled="!input.trim() || sending" class="btn-send">➤</button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onActivated } from 'vue'
import { useChatStore } from '../stores/chat'
import api from '../api'

const chatStore = useChatStore()
const messages = ref([])
const input = ref('')
const sending = ref(false)
const conversationId = ref(null)
const msgContainer = ref(null)
const inputEl = ref(null)

function scrollBottom() {
  nextTick(() => {
    if (msgContainer.value) {
      msgContainer.value.scrollTop = msgContainer.value.scrollHeight
    }
  })
}

// ── 监听会话切换，从 DB 加载历史 ─────────────────────────────────
watch(() => chatStore.activeConversationId, async (newId, oldId) => {
  if (newId === oldId) return
  conversationId.value = newId
  messages.value = []

  if (newId) {
    try {
      const res = await api.get(`/chat/conversations/${newId}/history`)
      messages.value = res.messages || []
      scrollBottom()
    } catch { /* 忽略 */ }
  }
  nextTick(() => inputEl.value?.focus())
}, { immediate: true })

async function send() {
  const text = input.value.trim()
  if (!text || sending.value) return
  input.value = ''
  sending.value = true

  messages.value.push({ role: 'user', content: text })
  scrollBottom()

  try {
    const res = await api.post('/chat/send', {
      query: text,
      conversation_id: conversationId.value,
    })
    conversationId.value = res.conversation_id
    messages.value.push({
      role: 'assistant',
      content: res.answer,
      meta: { intent: res.intent, elapsed: res.elapsed_seconds },
    })
    // 刷新侧边栏列表
    await chatStore.refreshAfterSend(res.conversation_id)
  } catch (e) {
    messages.value.push({
      role: 'assistant',
      content: `❌ 请求失败: ${e.response?.data?.detail || e.message}`,
    })
  } finally {
    sending.value = false
    scrollBottom()
    nextTick(() => inputEl.value?.focus())
  }
}

function newConversation() {
  chatStore.startNewConversation()
}

onActivated(() => {
  // 从知识库页切回时恢复焦点
  nextTick(() => inputEl.value?.focus())
})
</script>
