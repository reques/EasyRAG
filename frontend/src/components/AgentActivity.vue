<template>
  <div class="activity-panel" :class="{ 'is-running': running, 'is-error': !!error }">
    <!-- 头部：运行中展示实时状态，完成后展示汇总 -->
    <button
      type="button"
      class="activity-header"
      :class="{ expanded }"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      <span class="activity-title">
        <!-- 运行中：脉冲点 + 当前动作；完成后：对勾/错误 -->
        <span v-if="running" class="activity-live-dot" aria-hidden="true"></span>
        <CheckCircle2 v-else-if="!error" :size="14" class="activity-done-icon" />
        <XCircle v-else :size="14" class="activity-error-icon" />
        <span class="activity-headline">
          <template v-if="running">
            {{ activeStage ? activeStage.label : '处理中' }}
          </template>
          <template v-else-if="error">生成失败</template>
          <template v-else>思考过程</template>
        </span>
        <span v-if="running" class="activity-now-badge">进行中</span>
      </span>
      <span class="activity-meta">
        <span v-if="running" class="activity-timer">
          <Loader2 :size="11" class="spin" />
          {{ formatDuration(totalMs) }}
        </span>
        <span v-else-if="stages.length" class="activity-timer">
          {{ stages.length }} 步<span v-if="totalMs"> · {{ formatDuration(totalMs) }}</span>
        </span>
        <span v-else class="activity-timer">—</span>
        <ChevronDown v-if="expanded" :size="14" class="activity-chevron" />
        <ChevronRight v-else :size="14" class="activity-chevron" />
      </span>
    </button>

    <!-- 运行中的细进度线（indeterminate shimmer） -->
    <div v-if="running" class="activity-progress" aria-hidden="true"></div>
    <div v-if="error" class="activity-error-line">{{ error }}</div>

    <!-- 步骤时间线：grid-rows 折叠动画 -->
    <div class="activity-collapse" :class="{ open: expanded }">
      <div class="activity-steps" ref="wrapEl">
        <TransitionGroup name="activity-list" tag="div" class="activity-list">
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
              <span class="activity-row-name">
                {{ st.label }}
                <span v-if="st.detail" class="activity-row-detail">{{ st.detail }}</span>
              </span>
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

        <!-- 过程产出流：思维链 / 检索片段 / 工具输入输出 / 子任务产出，实时追加 -->
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
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import {
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  MessageSquareText,
  Target,
  Wrench,
  Search,
  PenLine,
  Brain,
  ShieldCheck,
  Workflow,
  Layers,
  Zap,
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
  /** 面板是否展开 */
  expanded: { type: Boolean, default: true },
})
const emit = defineEmits(['update:expanded'])

/* ── 步骤 → 阶段（stage）归一化 ─────────────────────────────────────
   SSE 事件是成对的 X / X_done，这里折叠成单个阶段：
   X 到达 → status=running；X_done 到达 → status=done + 更新 detail。
   面板事件（tool_call / worker_output / sub_tasks）不进时间线，
   它们由侧边任务面板承载。 */
const BASE_LABELS = {
  understand: '理解问题',
  intent: '识别意图',
  tool: '调用工具',
  retrieve: '检索知识库',
  generate: '生成回答',
  agent_reasoning: '推理思考',
  react: '推理思考',
  validate: '校验回答',
  answer_validation: '校验回答',
  decompose: '拆解任务',
  dispatch: '派发子任务',
  synthesize: '汇总结果',
  degenerate: '走快速路径',
  task_started: '子任务执行',
  fallback: '回退处理',
}
const BASE_ICONS = {
  understand: MessageSquareText,
  intent: Target,
  tool: Wrench,
  retrieve: Search,
  generate: PenLine,
  agent_reasoning: Brain,
  react: Brain,
  validate: ShieldCheck,
  answer_validation: ShieldCheck,
  decompose: Workflow,
  dispatch: Layers,
  synthesize: Layers,
  degenerate: Zap,
  task_started: ListChecks,
  fallback: AlertTriangle,
}
function baseOf(key) {
  return String(key || 'info').replace(/_done$/, '')
}
function kindOf(base) {
  if (['agent_reasoning', 'react'].includes(base)) return 'reason'
  if (['decompose', 'dispatch', 'synthesize', 'degenerate'].includes(base)) return 'orchestrate'
  if (['validate', 'answer_validation'].includes(base)) return 'validate'
  if (['tool_call', 'worker_output'].includes(base)) return 'panel'
  return base
}
let _stageUid = 0
function makeStage(base, raw, status) {
  _stageUid += 1
  return {
    uid: _stageUid,
    key: base,
    label: BASE_LABELS[base] || (raw.step || '信息'),
    icon: BASE_ICONS[base] || Info,
    kind: kindOf(base),
    detail: raw.detail || '',
    status,
    startedAt: raw._ts || null,
    finishedAt: null,
    isFallback: base === 'fallback',
  }
}

const stages = computed(() => {
  const result = []
  const byKey = new Map()
  for (const raw of props.steps) {
    const s = typeof raw === 'string' ? { step: '', detail: raw } : (raw || {})
    const key = s.step || 'info'
    const base = baseOf(key)
    // 面板事件不进入时间线
    if (kindOf(base) === 'panel') continue
    if (key.endsWith('_done')) {
      const stage = byKey.get(base)
      if (stage) {
        stage.status = 'done'
        stage.finishedAt = s._ts || null
        if (s.detail) stage.detail = s.detail
      } else {
        const orphan = makeStage(base, s, 'done')
        orphan.finishedAt = s._ts || null
        result.push(orphan)
        byKey.set(base, orphan)
      }
    } else {
      const stage = makeStage(base, s, 'running')
      result.push(stage)
      byKey.set(base, stage)
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
  // 取最后一条 running：DeepAgents 循环中"推理思考"与"调用工具"可能同时 running，
  // 头部应显示最新动作
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
  if (!first) return null // 历史恢复的步骤没有时间戳，不展示耗时
  const last = props.running
    ? now.value
    : (list[list.length - 1].finishedAt || now.value)
  return Math.max(0, last - first)
})

function stageDuration(st) {
  if (st.startedAt == null) return null
  if (st.status === 'done' && st.finishedAt != null) {
    const d = st.finishedAt - st.startedAt
    return d > 0 ? d : null // 无 _done 的瞬时阶段（degenerate 等）不显示 0ms
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

/* ── 交互：展开控制 + 运行中自动展开 + 活动行滚动可见 ─────────────── */
const expanded = computed({
  get: () => props.expanded,
  set: (value) => emit('update:expanded', value),
})
const wrapEl = ref(null)

watch(
  () => props.running,
  (running) => {
    // immediate：组件挂载时 running 已为 true（首步事件到达即挂载），也要自动展开
    if (running && !props.expanded) expanded.value = true
  },
  { immediate: true },
)

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
