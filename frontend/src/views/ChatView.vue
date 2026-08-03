<template>
  <div class="chat-view">
    <!-- 顶栏 -->
    <header class="chat-header">
      <h2><MessageSquare :size="16" /> 智能对话</h2>
      <div class="header-actions">
        <button @click="newConversation" class="btn-secondary">
          <MessageCirclePlus :size="14" /> 新对话
        </button>
      </div>
    </header>

    <!-- 消息列表 -->
    <div class="chat-messages" ref="msgContainer" :class="{ 'is-empty': messages.length === 0 && !sending }">
      <div class="chat-column">
        <!-- 空状态：Yuxi greeting — 轻量大标题 + 说明 -->
        <div v-if="messages.length === 0 && !sending" class="chat-empty">
          <h3>有什么可以帮你？</h3>
          <p>基于知识库与联网搜索，为你解答</p>
        </div>

        <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.role]">
          <div class="message-body">
            <div class="message-text" v-html="renderContent(msg.content)"></div>
            <!-- 知识库 / 检索引用块 -->
            <div v-if="msg.sources && msg.sources.length" class="message-sources">
              <div class="sources-title">
                <BookOpen :size="13" /> 参考来源
              </div>
              <ol class="sources-list">
                <li v-for="(s, si) in msg.sources" :key="si">
                  <span v-if="s.type === 'kb'" class="source-tag kb">知识库</span>
                  <span v-else-if="s.type === 'knowledge_graph'" class="source-tag kg">图谱</span>
                  <span v-else-if="s.url" class="source-tag web">网页</span>
                  <a v-if="s.url" :href="s.url" target="_blank" rel="noopener noreferrer">{{ s.title || s.url }}</a>
                  <!-- 知识库引用: 有 file_id 时可点击跳转到文档详情 -->
                  <a
                    v-else-if="(s.type === 'kb' || s.type === 'knowledge_graph') && s.file_id"
                    class="source-link"
                    @click.prevent="goToSource(s)"
                    href="#"
                  >{{ s.title }}</a>
                  <span v-else>{{ s.title }}</span>
                </li>
              </ol>
            </div>
            <div v-if="msg.meta && (msg.meta.intent || msg.meta.elapsed)" class="message-meta">
              <span v-if="msg.meta.intent">意图: {{ msg.meta.intent }}</span>
              <span v-if="msg.meta.elapsed">耗时: {{ msg.meta.elapsed }}s</span>
            </div>
          </div>
        </div>

        <!-- 仅当正在等待首个 token(最后一条 assistant 还没内容)时显示思考中 -->
        <div v-if="sending && !lastAssistantHasContent" class="message assistant">
          <div class="message-body">
            <div class="message-text typing">思考中<span>.</span><span>.</span><span>.</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 输入区：空状态时垂直居中，有消息后固定底部 -->
    <div class="chat-input" :class="{ 'chat-input--center': messages.length === 0 && !sending }">
      <div class="chat-input-inner">
        <textarea
          v-model="input"
          @keydown.enter.exact.prevent="send"
          placeholder="输入你的问题，Enter 发送…"
          rows="1"
          ref="inputEl"
        ></textarea>
        <button @click="send" :disabled="!input.trim() || sending" class="btn-send" title="发送">
          <ArrowUp :size="16" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onActivated } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '../stores/chat'
import { marked } from 'marked'
import { MessageSquare, MessageCirclePlus, ArrowUp, BookOpen } from 'lucide-vue-next'
import api from '../api'

// Render LLM markdown (bold, lists, links) to HTML. Links get target=_blank
// and rel=noopener so external sources open safely in a new tab.
marked.setOptions({ breaks: true, gfm: true })

function renderContent(text) {
  if (!text) return ''
  const html = marked.parse(text)
  // Only allow http/https links to open externally; force safe attrs.
  return html.replace(
    /<a\s+href="(https?:\/\/[^"]+)"([^>]*)>/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer"$2>'
  )
}

const chatStore = useChatStore()
const router = useRouter()
const messages = ref([])

// 点击知识库引用 → 跳转到知识库页并定位到对应文档详情
function goToSource(s) {
  router.push({
    path: '/knowledge',
    query: { kb: s.knowledge_base_id, file: s.file_id },
  })
}
const input = ref('')
const sending = ref(false)
const conversationId = ref(null)
const msgContainer = ref(null)
const inputEl = ref(null)

// 流式进行中: 最后一条 assistant 消息是否已开始收到内容
const lastAssistantHasContent = computed(() => {
  const last = messages.value[messages.value.length - 1]
  return !!(last && last.role === 'assistant' && last.content)
})

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
      // 历史消息 meta 中持久化了 sources，重载时还原引用块
      messages.value = (res.messages || []).map(m => ({
        role: m.role,
        content: m.content,
        sources: m.meta?.sources || [],
        meta: m.meta?.intent ? { intent: m.meta.intent } : null,
      }))
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
  // 先插入一条空的 assistant 消息, 流式 delta 逐步填充其 content
  messages.value.push({ role: 'assistant', content: '', sources: [], meta: null })
  const msgIndex = messages.value.length - 1
  scrollBottom()

  let gotError = ''
  try {
    await api.streamChat('/chat/stream', {
      query: text,
      conversation_id: conversationId.value,
    }, (ev) => {
      if (ev.type === 'conversation_id') {
        conversationId.value = ev.conversation_id
      } else if (ev.type === 'delta') {
        // 触发响应式更新: 替换数组元素
        const m = messages.value[msgIndex]
        m.content += ev.content
        messages.value[msgIndex] = { ...m }
        scrollBottom()
      } else if (ev.type === 'done') {
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = {
          ...m,
          sources: ev.sources || [],
          meta: { intent: ev.intent, elapsed: ev.elapsed_seconds },
        }
      } else if (ev.type === 'error') {
        gotError = ev.detail || '生成失败'
      }
    })

    if (gotError) {
      const m = messages.value[msgIndex]
      messages.value[msgIndex] = { ...m, content: m.content || `❌ ${gotError}` }
    }
    // 刷新侧边栏列表
    await chatStore.refreshAfterSend(conversationId.value)
  } catch (e) {
    const m = messages.value[msgIndex]
    messages.value[msgIndex] = {
      ...m,
      content: m.content || `❌ 请求失败: ${e.message}`,
    }
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
  // 从知识库页切回时滚动到底部并恢复焦点
  scrollBottom()
  nextTick(() => inputEl.value?.focus())
})
</script>
