<template>
  <div class="chat-view" :class="{ 'has-task-panel': taskPanel.visible }">
    <!-- 消息区 -->
    <div class="chat-main">
    <!-- 消息列表 -->
    <div class="chat-messages" ref="msgContainer" :class="{ 'is-empty': messages.length === 0 && !sending }">
      <div class="chat-column">
        <!-- 空状态：Yuxi greeting — 轻量大标题 + 说明 -->
        <div v-if="messages.length === 0 && !sending" class="chat-empty">
          <h3>有什么可以帮你？</h3>
          <p>基于知识库与联网搜索，为你解答</p>
        </div>

        <template v-for="(msg, i) in messages" :key="msg.ts || i">
          <!-- 等待首 token 时的空 assistant 占位不渲染，由下方「思考中」气泡代替，
               避免空灰条 + 思考中两个框同时出现 -->
          <div
            v-if="!(sending && msg.role === 'assistant' && !msg.content && !msg.steps?.length && i === messages.length - 1)"
            :class="['message', msg.role]"
          >
          <div class="message-body">
            <!-- 时间分隔条: 首条消息 / 距上一条超过 10 分钟时, 居中显示在消息上方 -->
            <div v-if="msg.time && shouldShowTimeSeparator(i)" class="message-time-separator">{{ msg.time }}</div>
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
    </div><!-- /.chat-main -->

    <!-- 侧边任务进度面板（多智能体请求时显示） -->
    <aside v-if="taskPanel.visible" class="task-panel">
      <div class="task-panel-header">
        <span class="task-panel-title">
          <Loader2 v-if="sending" :size="14" class="spin" />
          <CheckCircle2 v-else :size="14" />
          任务进度
        </span>
        <span class="task-panel-count">{{ taskProgress.done }}/{{ taskProgress.total }} · {{ taskProgress.pct }}%</span>
      </div>
      <div class="task-panel-bar">
        <div class="task-panel-bar-fill" :style="{ width: taskProgress.pct + '%' }"></div>
      </div>
      <div class="task-list">
        <div
          v-for="t in taskPanel.tasks"
          :key="t.task_id"
          class="task-item"
          :class="'task-' + t.status"
        >
          <span class="task-status-icon">
            <CheckCircle2 v-if="t.status === 'done'" :size="14" />
            <span v-else-if="t.status === 'error'" class="task-error-mark">✕</span>
            <Loader2 v-else-if="t.status === 'running'" :size="14" class="spin" />
            <span v-else class="task-pending-dot"></span>
          </span>
          <div class="task-item-body">
            <div class="task-item-title">{{ t.task_id }} · {{ workerLabel(t.worker_hint) }}</div>
            <div class="task-item-goal">{{ t.goal }}</div>
            <div v-if="t.tools.length" class="task-tools">
              <div v-for="(tc, ti) in t.tools" :key="ti" class="task-tool-call">
                <span class="task-tool-name">Call</span>
                <span class="task-tool-args">{{ tc }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </aside>
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

// 侧边任务进度面板（多智能体）：子任务清单 + 每个子任务的状态
const taskPanel = ref({
  visible: false,
  tasks: [],      // [{task_id, goal, worker_hint, status: pending|running|done|error, tools: []}]
})
function findTask(taskId) {
  return taskPanel.value.tasks.find(t => t.task_id === taskId)
}
function setTaskStatus(taskId, status) {
  const t = findTask(taskId)
  if (t) t.status = status
}

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

// worker 名 → 友好标签
const WORKER_LABELS = { rag: '知识库', legal: '法律', code: '代码' }
function workerLabel(hint) {
  return WORKER_LABELS[hint] || hint || '通用'
}

// 任务面板进度：已完成数 / 总数
const taskProgress = computed(() => {
  const tasks = taskPanel.value.tasks
  if (!tasks.length) return { done: 0, total: 0, pct: 0 }
  const done = tasks.filter(t => t.status === 'done' || t.status === 'error').length
  return { done, total: tasks.length, pct: Math.round(done / tasks.length * 100) }
})

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

// 消息时间戳：当天显示 HH:MM，跨天显示 MM-DD HH:MM
function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const pad = n => String(n).padStart(2, '0')
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  const now = new Date()
  const sameDay = d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate()
  return sameDay ? hm : `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}`
}

// 时间分隔条: 首条消息显示; 之后距上一条超过 10 分钟再显示当前时间
const TIME_SEPARATOR_GAP_MS = 10 * 60 * 1000
function shouldShowTimeSeparator(i) {
  if (i === 0) return true
  const prev = messages.value[i - 1]
  const curr = messages.value[i]
  if (!prev?.ts || !curr?.ts) return false
  return curr.ts - prev.ts > TIME_SEPARATOR_GAP_MS
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
        time: formatTime(m.created_at),
        ts: m.created_at ? Date.parse(m.created_at) : null,
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

  const userTs = Date.now()
  messages.value.push({ role: 'user', content: text, time: formatTime(new Date(userTs).toISOString()), ts: userTs })
  // 先插入一条空的 assistant 消息, 流式 delta 逐步填充其 content
  // steps: 本轮思考过程（绑定在这条消息上，不会被下一轮覆盖）
  // 注意: assistant 的时间戳独立取"当前时刻"——不要复用 user 的 ts,
  // 否则同秒差值为 0, 分隔条条件 >10min 永不成立, 回复后就看不到新时间了。
  const asstTs = Date.now()
  messages.value.push({ role: 'assistant', content: '', sources: [], meta: null, steps: [], stepsExpanded: true, stepsLoading: true, time: formatTime(new Date(asstTs).toISOString()), ts: asstTs })
  const msgIndex = messages.value.length - 1
  scrollBottom()

  let gotError = ''
  // 重置当前轮次的状态缓冲 + 任务面板
  statusSteps.value = []
  taskPanel.value = { visible: false, tasks: [] }

  try {
    await api.streamChat('/chat/stream', {
      query: text,
      conversation_id: conversationId.value,
    }, (ev) => {
      if (ev.type === 'conversation_id') {
        conversationId.value = ev.conversation_id
      } else if (ev.type === 'sub_tasks') {
        // 拆解完成：初始化侧边任务面板的待办清单（全部 pending）
        taskPanel.value = {
          visible: true,
          tasks: (ev.tasks || []).map(t => ({
            task_id: t.task_id,
            goal: t.goal,
            worker_hint: t.worker_hint,
            status: 'pending',
            tools: [],
          })),
        }
      } else if (ev.type === 'tool_call') {
        // 工具调用记录：挂到当前运行中的子任务下
        const running = taskPanel.value.tasks.find(t => t.status === 'running')
        if (running) running.tools.push(ev.detail)
      } else if (ev.type === 'status') {
        // 状态事件：落到当前 assistant 消息的 steps（随消息保留）
        const st = { step: ev.step, detail: ev.detail }
        statusSteps.value.push(st)
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = { ...m, steps: [...(m.steps || []), st] }
        // 派发开始 → 所有子任务标记运行中
        if (ev.step === 'dispatch') {
          taskPanel.value.tasks.forEach(t => { t.status = 'running' })
        }
        scrollBottom()
      } else if (ev.type === 'worker_output') {
        // 子任务产出：作为中间结果追加到消息内容，边执行边输出
        const m = messages.value[msgIndex]
        const header = `\n\n---\n**子任务 ${ev.task_id}（${ev.worker}）产出：**\n\n`
        m.content += header + ev.content
        messages.value[msgIndex] = { ...m }
        // 任务面板：该子任务完成
        setTaskStatus(ev.task_id, 'done')
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
        // 任务面板：全部子任务标记完成
        taskPanel.value.tasks.forEach(t => {
          if (t.status === 'running' || t.status === 'pending') t.status = 'done'
        })
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
