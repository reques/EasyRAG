<template>
  <div class="activity-stream" :class="{ 'is-running': running, 'is-error': !!error }">
    <!-- 错误提示 -->
    <div v-if="error" class="activity-error-line">{{ error }}</div>

    <!-- 操作流：每个操作一个条目（图标 + 英文动词 + 对象），实时追加，完成后打勾 -->
    <TransitionGroup name="activity-list" tag="div" class="activity-list" ref="wrapEl">
      <div
        v-for="st in displayStagesWithDuration"
        :key="st.uid"
        class="activity-row"
        :class="{
          'is-active': st.status === 'running',
          'is-done': st.status === 'done',
          'is-fallback': st.isFallback,
        }"
      >
        <span class="activity-badge" :class="`kind-${st.kind}`">
          <component :is="st.icon" :size="13" />
        </span>
        <span class="activity-copy">
          <span class="activity-verb">{{ st.label }}</span>
          <span v-if="st.subagent" class="activity-subagent">{{ st.subagent }}</span>
          <span v-if="st.object" class="activity-object" :class="`obj-${st.kind}`">{{ st.object }}</span>
        </span>
        <span class="activity-row-side">
          <span v-if="st.status === 'running'" class="activity-row-spinner">
            <Loader2 :size="12" class="spin" />
          </span>
          <CheckCircle2 v-else-if="st.status === 'done' && !st.isFallback" :size="13" class="activity-check" />
          <AlertTriangle v-else-if="st.isFallback" :size="13" class="activity-warn" />
          <span v-if="st.durationMs != null" class="activity-duration">{{ formatDuration(st.durationMs) }}</span>
        </span>
      </div>
    </TransitionGroup>

    <!-- 过程产出流：思维链 / 检索片段 / 工具输入输出 / 委派产出，实时追加 -->
    <TransitionGroup
      v-if="feedArtifacts.length"
      name="artifact-list"
      tag="div"
      class="activity-feed"
    >
      <div
        v-for="a in feedArtifacts"
        :key="a.id"
        class="artifact-card"
        :class="[`artifact-${a.kind}`, { 'is-streaming': a.streaming }]"
      >
        <span class="artifact-head">
          <component :is="artifactIcon(a.kind)" :size="12" />
          <span class="artifact-title">{{ a.title }}</span>
          <span v-if="stageTag(a.stage)" class="artifact-stage">{{ stageTag(a.stage) }}</span>
          <span v-if="a.streaming" class="artifact-caret" aria-hidden="true"></span>
        </span>
        <div v-if="a.content" class="artifact-content" :class="{ 'is-code': a.kind === 'tool' || a.kind === 'tool_result' }">
          {{ a.content }}
        </div>
      </div>
    </TransitionGroup>
    <div v-if="!stages.length && running" class="activity-empty">正在分析你的问题…</div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import {
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Target,
  Brain,
  BookOpen,
  Search,
  Wrench,
  PenLine,
  Zap,
  ShieldCheck,
  Workflow,
  Send,
  Merge,
  ListChecks,
  Info,
  FileText,
  Terminal,
  GitBranch,
} from 'lucide-vue-next'

const props = defineProps({
  /** 原始步骤数组：[{step, detail, task_id?, _ts?}] 或历史中的旧字符串格式 */
  steps: { type: Array, default: () => [] },
  /** 中间产出流：{id, kind, stage, title, content, streaming} */
  artifacts: { type: Array, default: () => [] },
  /** 本轮是否仍在进行（未收到 done/error） */
  running: { type: Boolean, default: false },
  /** 本轮错误信息（error 事件携带） */
  error: { type: String, default: '' },
  /** 兼容旧用法：保留 prop 但不再控制折叠（操作流始终铺开） */
  expanded: { type: Boolean, default: true },
})
const emit = defineEmits(['update:expanded'])

/* ── 操作（action）映射：后端 step key → 截图式英文动词 + 图标 ─────────
   2026-08-21：操作流铺开 + 英文动词式（Think / Read / Run / Search /
   Write / Check / Plan / Delegate / Merge ...），贴近 Cursor / Copilot
   风格的 agent 操作流展示。SSE 事件仍是成对的 X / X_done：
   X 到达 → status=running；X_done 到达 → status=done。 */
const ACTION_MAP = {
  // 查询改写 / 意图
  query_rewrite: { label: 'Rewrite', icon: RefreshCw, kind: 'rewrite' },
  rewrite: { label: 'Rewrite', icon: RefreshCw, kind: 'rewrite' },
  intent: { label: 'Classify', icon: Target, kind: 'intent' },
  intent_recognition: { label: 'Classify', icon: Target, kind: 'intent' },
  // 思考
  understand: { label: 'Think', icon: Brain, kind: 'reason' },
  agent_reasoning: { label: 'Think', icon: Brain, kind: 'reason' },
  react: { label: 'Think', icon: Brain, kind: 'reason' },
  reason: { label: 'Think', icon: Brain, kind: 'reason' },
  // 检索（Read = 读取知识库）
  retrieve: { label: 'Read', icon: BookOpen, kind: 'retrieve' },
  // 工具（Run；kb/web 搜索动态改为 Search）
  tool: { label: 'Run', icon: Wrench, kind: 'tool' },
  // 生成（Write = 产出回答）
  generate: { label: 'Write', icon: PenLine, kind: 'generate' },
  // 直接回答
  direct: { label: 'Answer', icon: Zap, kind: 'direct' },
  chitchat: { label: 'Answer', icon: Zap, kind: 'direct' },
  degenerate: { label: 'Answer', icon: Zap, kind: 'direct' },
  // 校验
  validate: { label: 'Check', icon: ShieldCheck, kind: 'validate' },
  answer_validation: { label: 'Check', icon: ShieldCheck, kind: 'validate' },
  // 编排
  decompose: { label: 'Plan', icon: Workflow, kind: 'orchestrate' },
  dispatch: { label: 'Delegate', icon: Send, kind: 'orchestrate' },
  synthesize: { label: 'Merge', icon: Merge, kind: 'orchestrate' },
  task_started: { label: 'Delegate', icon: ListChecks, kind: 'orchestrate' },
  // 回退
  fallback: { label: 'Fallback', icon: AlertTriangle, kind: 'fallback' },
  // 兜底
  info: { label: 'Step', icon: Info, kind: 'info' },
}
const SEARCH_TOOLS = new Set(['kb_search', 'web_search', 'search', 'tavily_search', 'duckduckgo'])

/** tool 步骤：detail 含工具名（"调用 kb_search(...)"）→ 搜索类工具用 Search 动词 */
function toolAction(detail) {
  const m = String(detail || '').match(/调用\s*([a-zA-Z_]+)/)
  const name = m ? m[1] : ''
  if (SEARCH_TOOLS.has(name)) return { label: 'Search', icon: Search, kind: 'retrieve' }
  return ACTION_MAP.tool
}

/** detail → 对象（去掉"调用 "等动作前缀，突出对象本身） */
function cleanObject(detail) {
  let d = String(detail || '').trim()
  d = d.replace(/^调用\s*/, '')
  return d
}

let _stageUid = 0
function makeAction(raw) {
  _stageUid += 1
  const s = typeof raw === 'string' ? { step: '', detail: raw } : (raw || {})
  const key = s.step || 'info'
  const done = key.endsWith('_done')
  const base = done ? key.slice(0, -5) : key
  // 子 Agent 步骤（S3）："research-agent/tool" → subagent=research-agent, action=tool
  let subagent = ''
  let actionKey = base
  if (base.includes('/')) {
    const parts = base.split('/')
    subagent = parts.slice(0, -1).join('/')
    actionKey = parts[parts.length - 1]
  }
  const action = actionKey === 'tool'
    ? toolAction(s.detail)
    : (ACTION_MAP[actionKey] || { label: actionKey || 'Step', icon: Info, kind: 'info' })
  return {
    uid: _stageUid,
    key: base,
    rawKey: key,
    done,
    actionKey,
    subagent,
    label: action.label,
    icon: action.icon,
    kind: action.kind,
    object: cleanObject(s.detail),
    status: done ? 'done' : 'running',
    startedAt: s._ts || null,
    finishedAt: done ? (s._ts || null) : null,
    isFallback: base === 'fallback',
  }
}

const stages = computed(() => {
  const result = []
  const byKey = new Map()
  for (const raw of props.steps) {
    const act = makeAction(raw)
    if (act.done) {
      // X_done 到达 → 把同 key 的 running 步骤置 done（瞬时步骤无配对则直接呈现 done）
      const stage = byKey.get(act.key)
      if (stage) {
        stage.status = 'done'
        stage.finishedAt = act.finishedAt
        // tool 配对保留调用信息（动作+参数），工具返回内容由 artifact 卡片承载；
        // 其余步骤（retrieve_done 等）用 done 的 detail 补充结果信息
        if (act.object && stage.actionKey !== 'tool') stage.object = act.object
      } else {
        result.push(act)
        byKey.set(act.key, act)
      }
    } else {
      result.push(act)
      byKey.set(act.key, act)
    }
  }
  return result
})

/* ── 展示状态：running 时最后阶段保持 active；结束后全部置 done ── */
const displayStages = computed(() => {
  if (props.running) return stages.value
  return stages.value.map((st) => (
    st.status === 'running' ? { ...st, status: 'done', finishedAt: st.finishedAt || (st.startedAt || null) } : st
  ))
})

const activeStage = computed(() => {
  if (!props.running) return null
  const list = displayStages.value
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i].status === 'running') return list[i]
  }
  return null
})

/* ── 耗时：阶段耗时 + 总耗时计时器 ────────────────────────────────── */
const now = ref(Date.now())
let timer = null
watch(
  () => props.running,
  (running) => {
    if (timer) { clearInterval(timer); timer = null }
    if (running) {
      now.value = Date.now()
      timer = setInterval(() => { now.value = Date.now() }, 250)
    }
  },
  { immediate: true },
)
onBeforeUnmount(() => { if (timer) clearInterval(timer) })

const totalMs = computed(() => {
  const list = displayStages.value
  if (!list.length) return null
  const first = list[0].startedAt
  if (!first) return null
  const last = props.running
    ? now.value
    : (list[list.length - 1].finishedAt || now.value)
  return Math.max(0, last - first)
})

function stageDuration(st) {
  if (st.startedAt == null) return null
  if (st.status === 'done' && st.finishedAt != null) {
    const d = st.finishedAt - st.startedAt
    return d > 0 ? d : null
  }
  if (st.status === 'running') return now.value - st.startedAt
  return null
}
function formatDuration(ms) {
  if (ms == null || ms < 0) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
const displayStagesWithDuration = computed(() =>
  displayStages.value.map((st) => ({ ...st, durationMs: stageDuration(st) })),
)

/* ── 活动行滚动可见 ───────────────────────────────────────────────── */
const wrapEl = ref(null)
function scrollActiveIntoView() {
  nextTick(() => {
    const wrap = wrapEl.value
    const active = wrap && wrap.querySelector('.activity-row.is-active')
    if (!wrap || !active) return
    const wr = wrap.getBoundingClientRect()
    const ar = active.getBoundingClientRect()
    if (ar.top < wr.top || ar.bottom > wr.bottom) {
      active.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  })
}
watch(
  [() => displayStages.value.length, () => activeStage.value?.key],
  () => { if (props.running) scrollActiveIntoView() },
  { flush: 'post' },
)

/* ── 过程产出流（artifacts）────────────────────────────────────────── */
const feedArtifacts = computed(() => (props.artifacts || []).map((a) => ({
  id: a.id || `art-${Math.random().toString(36).slice(2)}`,
  kind: a.kind || 'info',
  stage: a.stage || '',
  title: a.title || '',
  content: a.content || '',
  streaming: !!a.streaming,
})))
const ARTIFACT_ICONS = {
  thought: Brain,
  tool: Wrench,
  tool_result: Terminal,
  retrieve: FileText,
  worker: ListChecks,
  delegate: GitBranch,
  info: Info,
}
function artifactIcon(kind) {
  return ARTIFACT_ICONS[kind] || Info
}
const STAGE_TAGS = {
  reason: '推理',
  tool: '工具',
  retrieve: '检索',
  generate: '生成',
  synthesize: '汇总',
  plan: '计划',
}
function stageTag(stage) {
  return STAGE_TAGS[stage] || ''
}
// 产出流更新（新增卡片 / 流式内容增长）→ 滚动到面板底部
watch(
  [
    () => feedArtifacts.value.length,
    () => feedArtifacts.value[feedArtifacts.value.length - 1]?.content?.length || 0,
  ],
  () => {
    if (!props.running) return
    nextTick(() => {
      if (wrapEl.value) wrapEl.value.scrollTop = wrapEl.value.scrollHeight
    })
  },
  { flush: 'post' },
)
</script>
