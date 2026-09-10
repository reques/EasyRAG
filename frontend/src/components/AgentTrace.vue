<template>
  <section v-if="events.length" class="trace-panel" aria-live="polite">
    <button class="trace-heading" type="button" @click="panelOpen = !panelOpen">
      <Activity :size="13" />
      <span>执行过程</span>
      <span class="trace-summary">{{ events.length }} 个步骤</span>
      <span v-if="tokenUsage.total_tokens" class="trace-summary">{{ formatTokens(tokenUsage.total_tokens) }} tokens</span>
      <ChevronDown :size="13" :class="{ flipped: !panelOpen }" />
    </button>

    <div v-if="panelOpen" class="trace-tree">
      <div
        v-for="row in visibleRows"
        :key="row.event.id"
        class="trace-node"
        :class="[`status-${effectiveStatus(row.event)}`, `type-${row.event.type}`]"
        :style="{ '--depth': row.depth }"
      >
        <button class="trace-node-head" type="button" @click="toggle(row.event.id)">
          <span class="trace-guide"></span>
          <ChevronRight
            v-if="row.hasChildren || hasDetails(row.event)"
            :size="12"
            :class="{ expanded: expanded.has(row.event.id) }"
          />
          <span v-else class="trace-spacer"></span>
          <Loader2 v-if="effectiveStatus(row.event) === 'running'" :size="12" class="spin" />
          <CircleAlert v-else-if="effectiveStatus(row.event) === 'error'" :size="12" />
          <CheckCircle2 v-else-if="effectiveStatus(row.event) === 'completed'" :size="12" />
          <Circle v-else :size="10" />
          <strong>{{ label(row.event) }}</strong>
          <span class="trace-title">{{ displayTitle(row.event) }}</span>
          <span v-if="duration(row.event)" class="trace-duration">{{ duration(row.event) }}</span>
        </button>
        <div v-if="expanded.has(row.event.id) && hasDetails(row.event)" class="trace-details">
          <div v-if="row.event.input != null">
            <b>输入</b><pre>{{ formatValue(row.event.input) }}</pre>
          </div>
          <div v-if="row.event.output != null">
            <b>输出</b><pre>{{ formatValue(row.event.output) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import {
  Activity, CheckCircle2, ChevronDown, ChevronRight, Circle, CircleAlert, Loader2,
} from 'lucide-vue-next'

const props = defineProps({
  events: { type: Array, default: () => [] },
  running: { type: Boolean, default: false },
  tokenUsage: { type: Object, default: () => ({}) },
})

const panelOpen = ref(false)
const expanded = reactive(new Set())

watch(() => props.running, (running) => {
  if (running) panelOpen.value = true
  else {
    panelOpen.value = false
    expanded.clear()
  }
}, { immediate: true })
watch(() => props.events.length, () => {
  if (props.running) {
    const root = props.events.find(event => event.type === 'agent_start')
    if (root) expanded.add(root.id)
  }
})

const rows = computed(() => {
  const ordered = [...props.events].sort((a, b) =>
    (a.sequence || 0) - (b.sequence || 0)
    || String(a.timestamp || '').localeCompare(String(b.timestamp || ''))
  )
  const byParent = new Map()
  for (const event of ordered) {
    const parent = event.parent_id || ''
    if (!byParent.has(parent)) byParent.set(parent, [])
    byParent.get(parent).push(event)
  }
  const ids = new Set(ordered.map(event => event.id))
  const result = []
  const seen = new Set()
  const visit = (event, depth) => {
    if (seen.has(event.id)) return
    seen.add(event.id)
    const children = byParent.get(event.id) || []
    result.push({ event, depth, hasChildren: children.length > 0 })
    if (expanded.has(event.id)) children.forEach(child => visit(child, depth + 1))
  }
  ordered
    .filter(event => !event.parent_id || !ids.has(event.parent_id))
    .forEach(event => visit(event, 0))
  return result
})

const visibleRows = rows
const labels = {
  agent_start: '开始处理',
  planning: '任务进度',
  reasoning_summary: '分析摘要',
  tool_call: '调用工具',
  tool_result: '工具结果',
  file_operation: '文件操作',
  code_execution: '执行代码',
  error: '执行异常',
  final_response: '回复完成',
}
const stageTitles = {
  understand: '分析任务',
  generate: '正在生成回复',
  generate_done: '回复生成完成',
  reason: '分析下一步操作',
  tool: '执行所需工具',
  tool_done: '工具执行完成',
}

function label(event) { return labels[event.type] || event.type }
function displayTitle(event) {
  if (event.type === 'agent_start') return ''
  if (event.type === 'final_response') return ''
  const stage = event.metadata?.stage || ''
  const title = event.metadata?.title || ''
  return stageTitles[stage] || stageTitles[title] || title
}
function effectiveStatus(event) {
  if (!props.running && event.status === 'running') return 'completed'
  return event.status
}
function hasDetails(event) {
  if (event.type === 'agent_start') return false
  return event.input != null || event.output != null
}
function toggle(id) {
  if (expanded.has(id)) expanded.delete(id)
  else expanded.add(id)
}
function duration(event) {
  const ms = event.metadata?.elapsed_ms
  if (ms == null) return ''
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}
function formatValue(value) {
  if (typeof value === 'string') return value
  try { return JSON.stringify(value, null, 2) } catch { return String(value) }
}
function formatTokens(value) { return Number(value || 0).toLocaleString() }
</script>

<style scoped>
.trace-panel { margin: 4px 0 12px; color: var(--text-secondary, #68707d); font-size: 12px; }
.trace-heading, .trace-node-head { width: 100%; border: 0; background: transparent; color: inherit; display: flex; align-items: center; gap: 7px; cursor: pointer; text-align: left; }
.trace-heading { padding: 5px 0; font-weight: 650; color: var(--text-primary, #2d333b); }
.trace-heading .trace-summary { margin-left: 3px; font-weight: 400; color: var(--text-tertiary, #9098a5); }
.trace-heading svg:last-child { margin-left: auto; transition: transform .15s; }
.trace-heading svg.flipped { transform: rotate(-90deg); }
.trace-tree { margin: 2px 0 5px; }
.trace-node { --depth: 0; }
.trace-node-head { min-height: 27px; padding-left: calc(var(--depth) * 18px); }
.trace-node-head strong { color: var(--text-primary, #343a43); font-size: 11px; white-space: nowrap; }
.trace-node-head > svg:first-of-type { transition: transform .15s; }
.trace-node-head > svg.expanded { transform: rotate(90deg); }
.trace-spacer { width: 12px; }
.trace-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trace-duration { margin-left: auto; color: var(--text-tertiary, #959da8); font-variant-numeric: tabular-nums; }
.status-error .trace-node-head { color: #d14343; }
.status-error .trace-node-head strong { color: #b83232; }
.trace-details { margin: 0 0 6px calc(var(--depth) * 18px + 46px); padding: 7px 9px; border-left: 2px solid var(--border-color, #e3e6ea); background: rgba(127,127,127,.045); border-radius: 4px; }
.trace-details b { display: block; margin: 2px 0 3px; font-size: 10px; text-transform: uppercase; letter-spacing: .04em; }
.trace-details pre { margin: 0 0 7px; max-height: 180px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; color: var(--text-primary, #343a43); }
.trace-details code { font-size: 11px; }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
