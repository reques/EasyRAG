<template>
  <div class="chat-view">
    <!-- 消息列表 -->
    <div class="chat-messages" ref="msgContainer" :class="{ 'is-empty': messages.length === 0 && !sending }">
      <div class="chat-column">
        <!-- 空状态：Yuxi greeting — 轻量大标题 + 说明 -->
        <div v-if="messages.length === 0 && !sending" class="chat-empty">
          <h3>有什么可以帮你？</h3>
          <p>基于知识库与联网搜索，为你解答</p>
        </div>

        <template v-for="(msg, i) in messages" :key="i">
          <!-- 等待首 token 时的空 assistant 占位不渲染，由下方「思考中」气泡代替，
               避免空灰条 + 思考中两个框同时出现 -->
          <div
            v-if="!(sending && msg.role === 'assistant' && !msg.content && !msg.steps?.length && i === messages.length - 1)"
            :class="['message', msg.role]"
          >
          <div class="message-body">
            <!-- 思考过程：绑定在该条消息上，渲染在答案上方，不被下一轮覆盖 -->
            <div v-if="msg.steps && msg.steps.length" class="status-panel">
              <div class="status-header" @click="msg.stepsExpanded = !msg.stepsExpanded">
                <span class="status-title">
                  <Loader2 v-if="msg.stepsLoading" :size="14" class="spin" />
                  <CheckCircle2 v-else :size="14" />
                  思考过程
                </span>
                <ChevronDown v-if="msg.stepsExpanded" :size="14" />
                <ChevronRight v-else :size="14" />
              </div>
              <div v-show="msg.stepsExpanded" class="status-steps">
                <div v-for="(stRaw, si) in msg.steps" :key="si" class="status-step" :class="{ active: si === msg.steps.length - 1 && msg.stepsLoading }">
                  <span class="step-dot"></span>
                  <span class="step-name">{{ stepLabel(normalizeStep(stRaw).step) }}</span>
                  <span class="step-detail">{{ normalizeStep(stRaw).detail }}</span>
                </div>
              </div>
            </div>
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
        </template>

        <!-- 思考中占位：还没有任何状态步骤时的等待气泡（有步骤后由消息内面板接管） -->
        <div v-if="sending && statusSteps.length === 0 && !lastAssistantHasContent" class="message assistant">
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
          @input="autoResize"
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
import { ArrowUp, BookOpen, ChevronDown, ChevronRight, Loader2, CheckCircle2 } from 'lucide-vue-next'
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

// 状态步骤面板（思考过程时间线）
// statusSteps 只是当前轮次的缓冲——status 事件实时落到当前 assistant 消息的
// msg.steps 上（随消息保留，渲染在答案上方，不会被下一轮清空覆盖）
const statusSteps = ref([])

// 把后端步骤 key 映射为友好的阶段名
const STEP_LABELS = {
  understand: '理解问题',
  understand_done: '问题理解',
  intent: '识别意图',
  intent_done: '意图',
  tool: '调用工具',
  tool_done: '工具结果',
  retrieve: '检索知识库',
  retrieve_done: '检索完成',
  generate: '生成回答',
  decompose: '拆解任务',
  decompose_done: '拆解完成',
  dispatch: '派发子任务',
  dispatch_done: '派发完成',
  synthesize: '汇总结果',
  synthesize_done: '汇总完成',
  fallback: '回退',
}
function stepLabel(key) {
  return STEP_LABELS[key] || key
}

// 兼容旧数据：早期版本 meta.steps 存的是 orchestrator 内部字符串日志
// （如 "orchestrator 接收查询: ..."），直接渲染成步骤名
function normalizeStep(st) {
  if (typeof st === 'string') return { step: '', detail: st }
  return st
}

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

// 输入框随内容自动撑高（超单行时），上限由 CSS max-height 控制
function autoResize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${el.scrollHeight}px`
}

function resetInputHeight() {
  if (inputEl.value) inputEl.value.style.height = 'auto'
}

// ── 监听会话切换，从 DB 加载历史 ─────────────────────────────────
watch(() => chatStore.activeConversationId, async (newId, oldId) => {
  if (newId === oldId) return
  conversationId.value = newId
  messages.value = []

  if (newId) {
    try {
      const res = await api.get(`/chat/conversations/${newId}/history`)
      // 历史消息 meta 中持久化了 sources 和 steps，重载时还原引用块 + 思考过程
      messages.value = (res.messages || []).map(m => ({
        role: m.role,
        content: m.content,
        sources: m.meta?.sources || [],
        meta: m.meta?.intent ? { intent: m.meta.intent } : null,
        steps: m.meta?.steps || [],
        stepsExpanded: true,
        stepsLoading: false,
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
  resetInputHeight()
  sending.value = true

  messages.value.push({ role: 'user', content: text })
  // 先插入一条空的 assistant 消息, 流式 delta 逐步填充其 content
  // steps: 本轮思考过程（绑定在这条消息上，不会被下一轮覆盖）
  messages.value.push({ role: 'assistant', content: '', sources: [], meta: null, steps: [], stepsExpanded: true, stepsLoading: true })
  const msgIndex = messages.value.length - 1
  scrollBottom()

  let gotError = ''
  // 重置当前轮次的状态缓冲
  statusSteps.value = []

  try {
    await api.streamChat('/chat/stream', {
      query: text,
      conversation_id: conversationId.value,
    }, (ev) => {
      if (ev.type === 'conversation_id') {
        conversationId.value = ev.conversation_id
      } else if (ev.type === 'status') {
        // 状态事件：落到当前 assistant 消息的 steps（随消息保留）
        const st = { step: ev.step, detail: ev.detail }
        statusSteps.value.push(st)
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = { ...m, steps: [...(m.steps || []), st] }
        scrollBottom()
      } else if (ev.type === 'worker_output') {
        // 子任务产出：作为中间结果追加到消息内容，边执行边输出
        const m = messages.value[msgIndex]
        const header = `\n\n---\n**子任务 ${ev.task_id}（${ev.worker}）产出：**\n\n`
        m.content += header + ev.content
        messages.value[msgIndex] = { ...m }
        scrollBottom()
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
          // 兜底：流式中断时 status 事件可能不全，用 done 携带的完整 steps 补齐
          steps: (ev.steps && ev.steps.length) ? ev.steps : (m.steps || []),
          stepsLoading: false,
        }
      } else if (ev.type === 'error') {
        gotError = ev.detail || '生成失败'
      }
    })

    if (gotError) {
      const m = messages.value[msgIndex]
      messages.value[msgIndex] = { ...m, content: m.content || `❌ ${gotError}`, stepsLoading: false }
    }
    // 刷新侧边栏列表
    await chatStore.refreshAfterSend(conversationId.value)
  } catch (e) {
    const m = messages.value[msgIndex]
    messages.value[msgIndex] = {
      ...m,
      content: m.content || `❌ 请求失败: ${e.message}`,
      stepsLoading: false,
    }
  } finally {
    sending.value = false
    scrollBottom()
    nextTick(() => inputEl.value?.focus())
  }
}

onActivated(() => {
  // 从知识库页切回时滚动到底部并恢复焦点
  scrollBottom()
  nextTick(() => inputEl.value?.focus())
})
</script>
