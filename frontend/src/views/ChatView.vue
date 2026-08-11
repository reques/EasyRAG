<template>
  <div class="chat-view" :class="{ 'has-task-panel': taskPanel.visible }">
    <!-- 消息区 -->
    <div class="chat-main">
    <!-- 消息列表 -->
    <div class="chat-messages" ref="msgContainer" :class="{ 'is-empty': messages.length === 0 && !sending }">
      <!-- 任务面板开关：本轮有任务记录但面板被手动关闭时，提供重新打开的入口 -->
      <div v-if="taskPanel.tasks.length && !taskPanel.visible" class="task-panel-toggle">
        <button class="task-panel-toggle-btn" @click="taskPanel.visible = true">
          <ListChecks :size="14" />
          任务进度（{{ taskProgress.done }}/{{ taskProgress.total }}）
        </button>
      </div>
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
          <!-- 用户消息时间: 显示在气泡外上方, 左对齐时间标签, 不放进气泡里 -->
          <div v-if="msg.role === 'user' && msg.time && shouldShowTimeSeparator(i)" class="message-time-separator">{{ msg.time }}</div>
          <div class="message-body">
            <!-- AI 消息时间分隔条: 首条消息 / 距上一条超过 10 分钟时, 居中显示在消息上方 -->
            <div v-if="msg.role === 'assistant' && msg.time && shouldShowTimeSeparator(i)" class="message-time-separator">{{ msg.time }}</div>
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
            <div v-if="msg.meta && (msg.meta.intent || msg.meta.elapsed || msg.meta.modelName)" class="message-meta">
              <span v-if="msg.meta.modelName">模型: {{ msg.meta.modelName }}</span>
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
        <div class="chat-input-actions">
          <div ref="modelPickerEl" class="model-picker-shell">
            <button
              type="button"
              class="model-picker"
              :class="{ 'has-error': modelLoadError, open: modelMenuOpen }"
              :title="modelPickerTitle"
              :disabled="sending || modelsLoading"
              @click="modelMenuOpen = !modelMenuOpen"
            >
              <span class="model-status-dot" :class="{ ready: selectedModel?.available }"></span>
              <span class="model-picker-label">
                {{ modelsLoading ? '加载模型中…' : (selectedModel?.name || '添加自定义模型') }}
              </span>
              <ChevronDown :size="13" />
            </button>
            <div v-if="modelMenuOpen" class="model-dropdown">
              <div class="model-dropdown-title">对话模型</div>
              <button
                v-for="model in modelOptions"
                :key="model.id"
                type="button"
                class="model-dropdown-item"
                :class="{ selected: model.id === selectedModelId }"
                :disabled="!model.available"
                @click="selectModel(model)"
              >
                <span class="model-status-dot" :class="{ ready: model.available }"></span>
                <span class="model-option-copy">
                  <strong>{{ model.name }}</strong>
                  <small>{{ model.provider_type === 'local' ? '本地' : model.provider }}</small>
                </span>
                <span v-if="!model.available" class="model-unavailable">未配置</span>
                <CheckCircle2 v-else-if="model.id === selectedModelId" :size="14" />
              </button>
              <button type="button" class="model-dropdown-add" @click="openCustomModelModal">
                <Plus :size="14" /> 添加自定义模型
              </button>
            </div>
          </div>
          <button @click="send" :disabled="!input.trim() || sending || !selectedModelId" class="btn-send" title="发送">
            <ArrowUp :size="16" />
          </button>
        </div>
      </div>
    </div>
    </div><!-- /.chat-main -->

    <!-- 侧边任务进度面板（多智能体请求时显示，可手动开/关） -->
    <aside v-if="taskPanel.visible" class="task-panel">
      <div class="task-panel-header">
        <span class="task-panel-title">
          <Loader2 v-if="sending" :size="14" class="spin" />
          <CheckCircle2 v-else :size="14" />
          任务进度
        </span>
        <span class="task-panel-actions">
          <span class="task-panel-count">{{ taskProgress.done }}/{{ taskProgress.total }} · {{ taskProgress.pct }}%</span>
          <button class="task-panel-close" title="关闭任务面板" @click="taskPanel.visible = false">
            <X :size="14" />
          </button>
        </span>
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

    <Teleport to="body">
      <div v-if="customModelModalOpen" class="modal-overlay" @click.self="closeCustomModelModal">
        <div class="modal model-config-modal">
          <div class="model-config-header">
            <div>
              <h3>添加自定义模型</h3>
              <p>支持 OpenAI 兼容的本地服务或云端接口，配置会安全保存到后端。</p>
            </div>
            <button type="button" class="model-config-close" @click="closeCustomModelModal">×</button>
          </div>

          <form class="model-config-form" @submit.prevent="saveCustomModel">
            <div class="model-type-grid">
              <button
                type="button"
                :class="['model-type-card', { active: customModelForm.provider_type === 'local' }]"
                @click="setCustomModelType('local')"
              >
                <HardDrive :size="18" />
                <span><strong>本地模型</strong><small>Ollama、LM Studio 等</small></span>
              </button>
              <button
                type="button"
                :class="['model-type-card', { active: customModelForm.provider_type === 'cloud' }]"
                @click="setCustomModelType('cloud')"
              >
                <Cloud :size="18" />
                <span><strong>云端模型</strong><small>OpenAI 兼容 API</small></span>
              </button>
            </div>

            <div class="model-form-grid">
              <label>
                <span>显示名称</span>
                <input v-model="customModelForm.name" type="text" maxlength="80" placeholder="例如：本地 Qwen 32B" required />
              </label>
              <label>
                <span>供应商名称</span>
                <input v-model="customModelForm.provider_name" type="text" maxlength="80" placeholder="例如：Ollama / OpenRouter" />
              </label>
            </div>
            <label>
              <span>API Base URL</span>
              <input v-model="customModelForm.base_url" type="url" maxlength="512" :placeholder="customModelUrlPlaceholder" required />
              <small class="model-form-hint">填写 OpenAI 兼容接口根地址，通常以 /v1 结尾。</small>
            </label>
            <div class="model-form-grid">
              <label>
                <span>模型 ID</span>
                <input v-model="customModelForm.model_name" type="text" maxlength="160" placeholder="例如：qwen3:32b" required />
              </label>
              <label>
                <span>Temperature</span>
                <input v-model.number="customModelForm.temperature" type="number" min="0" max="2" step="0.1" />
              </label>
            </div>
            <label class="model-key-toggle">
              <input v-model="customModelForm.requires_api_key" type="checkbox" />
              <span>此接口需要 API Key</span>
            </label>
            <label v-if="customModelForm.requires_api_key">
              <span>API Key</span>
              <input v-model="customModelForm.api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="仅加密保存到后端" required />
            </label>

            <div v-if="customModelError" class="model-config-error">{{ customModelError }}</div>

            <div v-if="customModels.length" class="custom-model-list">
              <div class="custom-model-list-title">已添加</div>
              <div v-for="model in customModels" :key="model.id" class="custom-model-row">
                <span class="model-option-copy">
                  <strong>{{ model.name }}</strong>
                  <small>{{ model.provider_type === 'local' ? '本地模型' : model.provider }}</small>
                </span>
                <button type="button" title="删除模型" @click="deleteCustomModel(model)">
                  <Trash2 :size="14" />
                </button>
              </div>
            </div>

            <div class="modal-actions">
              <button type="button" class="btn-secondary" @click="closeCustomModelModal">取消</button>
              <button type="submit" class="btn-primary-sm" :disabled="savingCustomModel">
                {{ savingCustomModel ? '保存中…' : '保存并使用' }}
              </button>
            </div>
          </form>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick, onActivated, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '../stores/chat'
import { marked } from 'marked'
import {
  ArrowUp,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cloud,
  HardDrive,
  ListChecks,
  Loader2,
  Plus,
  Trash2,
  X,
} from 'lucide-vue-next'
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

// 对话模型目录由后端环境配置生成；浏览器只保存公开 model_id，不接触供应商密钥。
const MODEL_STORAGE_KEY = 'easyrag-chat-model-id'
const modelOptions = ref([])
const selectedModelId = ref(localStorage.getItem(MODEL_STORAGE_KEY) || '')
const modelsLoading = ref(true)
const modelLoadError = ref('')
const modelMenuOpen = ref(false)
const modelPickerEl = ref(null)
const customModelModalOpen = ref(false)
const savingCustomModel = ref(false)
const customModelError = ref('')
const customModelForm = reactive({
  provider_type: 'local',
  name: '',
  provider_name: 'Ollama',
  base_url: 'http://localhost:11434/v1',
  model_name: '',
  api_key: '',
  requires_api_key: false,
  temperature: 0,
})
const selectedModel = computed(() => (
  modelOptions.value.find(model => model.id === selectedModelId.value) || null
))
const customModels = computed(() => (
  modelOptions.value.filter(model => model.source === 'custom')
))
const customModelUrlPlaceholder = computed(() => (
  customModelForm.provider_type === 'local'
    ? 'http://localhost:11434/v1'
    : 'https://api.example.com/v1'
))
const modelPickerTitle = computed(() => {
  if (modelLoadError.value) return modelLoadError.value
  if (selectedModel.value) return `${selectedModel.value.provider} · ${selectedModel.value.name}`
  return '添加自定义模型'
})

async function loadModels(preferredModelId = '') {
  modelsLoading.value = true
  modelLoadError.value = ''
  try {
    const data = await api.get('/chat/models')
    modelOptions.value = data.models || []
    const available = modelOptions.value.filter(model => model.available)
    const requested = available.find(model => model.id === preferredModelId)
    const saved = available.find(model => model.id === selectedModelId.value)
    const preferred = available.find(model => model.id === data.default_model_id)
    selectedModelId.value = (requested || saved || preferred || available[0])?.id || ''
  } catch (error) {
    modelOptions.value = []
    selectedModelId.value = ''
    modelLoadError.value = `模型列表加载失败：${error.message}`
  } finally {
    modelsLoading.value = false
  }
}

watch(selectedModelId, (modelId) => {
  if (modelId) localStorage.setItem(MODEL_STORAGE_KEY, modelId)
  else localStorage.removeItem(MODEL_STORAGE_KEY)
})

function selectModel(model) {
  if (!model.available) return
  selectedModelId.value = model.id
  modelMenuOpen.value = false
}

function setCustomModelType(type) {
  customModelForm.provider_type = type
  customModelForm.api_key = ''
  if (type === 'local') {
    customModelForm.provider_name = 'Ollama'
    customModelForm.base_url = 'http://localhost:11434/v1'
    customModelForm.requires_api_key = false
    customModelForm.temperature = 0
  } else {
    customModelForm.provider_name = ''
    customModelForm.base_url = ''
    customModelForm.requires_api_key = true
    customModelForm.temperature = 0.7
  }
}

function openCustomModelModal() {
  modelMenuOpen.value = false
  customModelError.value = ''
  Object.assign(customModelForm, {
    provider_type: 'local',
    name: '',
    provider_name: 'Ollama',
    base_url: 'http://localhost:11434/v1',
    model_name: '',
    api_key: '',
    requires_api_key: false,
    temperature: 0,
  })
  customModelModalOpen.value = true
}

function closeCustomModelModal() {
  if (savingCustomModel.value) return
  customModelModalOpen.value = false
  customModelError.value = ''
}

async function saveCustomModel() {
  if (savingCustomModel.value) return
  savingCustomModel.value = true
  customModelError.value = ''
  try {
    const created = await api.post('/chat/models', {
      provider_type: customModelForm.provider_type,
      name: customModelForm.name.trim(),
      provider_name: customModelForm.provider_name.trim(),
      base_url: customModelForm.base_url.trim(),
      model_name: customModelForm.model_name.trim(),
      api_key: customModelForm.api_key.trim(),
      requires_api_key: customModelForm.requires_api_key,
      temperature: customModelForm.temperature,
    })
    await loadModels(created.id)
    customModelModalOpen.value = false
  } catch (error) {
    customModelError.value = error.response?.data?.detail || error.message || '保存失败'
  } finally {
    savingCustomModel.value = false
  }
}

async function deleteCustomModel(model) {
  if (!window.confirm(`确定删除自定义模型“${model.name}”吗？`)) return
  customModelError.value = ''
  try {
    await api.delete(`/chat/models/${model.id}`)
    if (selectedModelId.value === model.id) selectedModelId.value = ''
    await loadModels()
  } catch (error) {
    customModelError.value = error.response?.data?.detail || error.message || '删除失败'
  }
}

function closeModelMenuOnOutsideClick(event) {
  if (modelPickerEl.value && !modelPickerEl.value.contains(event.target)) {
    modelMenuOpen.value = false
  }
}

// 状态步骤面板（思考过程时间线）
// statusSteps 只是当前轮次的缓冲——status 事件实时落到当前 assistant 消息的
// msg.steps 上（随消息保留，渲染在答案上方，不会被下一轮清空覆盖）
const statusSteps = ref([])

// 侧边任务进度面板（多智能体）：子任务清单 + 每个子任务的状态
const taskPanel = ref({
  visible: false,
  run_id: '',
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
  // 切换会话 → 清空上一会话的任务面板（否则面板残留 pin 在右边）
  taskPanel.value = { visible: false, run_id: '', tasks: [] }

  if (newId) {
    try {
      const res = await api.get(`/chat/conversations/${newId}/history`)
      // 历史消息 meta 中持久化了 sources 和 steps，重载时还原引用块 + 思考过程
      messages.value = (res.messages || []).map(m => ({
        role: m.role,
        content: m.content,
        sources: m.meta?.sources || [],
        meta: (m.meta?.intent || m.meta?.model_name || m.meta?.run_id) ? {
          intent: m.meta?.intent || '',
          modelName: m.meta?.model_name || '',
          runId: m.meta?.run_id || '',
        } : null,
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
  messages.value.push({
    role: 'assistant',
    content: '',
    sources: [],
    meta: { modelName: selectedModel.value?.name || '' },
    steps: [],
    stepsExpanded: true,
    stepsLoading: true,
    time: formatTime(new Date(asstTs).toISOString()),
    ts: asstTs,
  })
  const msgIndex = messages.value.length - 1
  scrollBottom()

  let gotError = ''
  // 重置当前轮次的状态缓冲 + 任务面板
  statusSteps.value = []
  taskPanel.value = { visible: false, run_id: '', tasks: [] }

  try {
    await api.streamChat('/chat/stream', {
      query: text,
      conversation_id: conversationId.value,
      model_id: selectedModelId.value,
    }, (ev) => {
      if (ev.type === 'conversation_id') {
        conversationId.value = ev.conversation_id
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = {
          ...m,
          meta: {
            ...(m.meta || {}),
            runId: ev.run_id || m.meta?.runId || '',
            modelName: ev.model_name || m.meta?.modelName || '',
          },
        }
        taskPanel.value.run_id = ev.run_id || ''
      } else if (ev.type === 'sub_tasks') {
        // 拆解完成：初始化侧边任务面板的待办清单（全部 pending）
        taskPanel.value = {
          visible: true,
          run_id: ev.run_id || taskPanel.value.run_id || '',
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
          meta: {
            intent: ev.intent,
            elapsed: ev.elapsed_seconds,
            runId: ev.run_id || m.meta?.runId || '',
            modelName: ev.model_name || m.meta?.modelName || '',
          },
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

onMounted(() => {
  loadModels()
  document.addEventListener('click', closeModelMenuOnOutsideClick)
})

onUnmounted(() => {
  document.removeEventListener('click', closeModelMenuOnOutsideClick)
})
</script>
