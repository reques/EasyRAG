<template>
  <section v-if="rows.length || running || error" class="trace-process" aria-label="执行过程" :aria-busy="running">
    <div v-if="!rows.length && running" class="trace-preparing" role="status">
      <Loader2 :size="13" class="spin" /> 正在处理
    </div>
    <ol class="trace-flow">
      <li v-for="row in rows" :key="row.id" :class="[`trace-${row.kind}`, `status-${status(row)}`]">
        <!-- Public action summaries stay readable after completion. -->
        <p v-if="row.kind === 'commentary'" class="trace-commentary-text">{{ row.text }}<span v-if="running && row.event.metadata?.streaming" class="trace-caret" aria-hidden="true"></span></p>
        <template v-else>
          <button type="button" class="trace-operation-head" :aria-expanded="expanded.has(row.id)" @click="toggle(row.id)">
            <Loader2 v-if="status(row) === 'running'" :size="13" class="spin" />
            <CircleAlert v-else-if="['error', 'stopped'].includes(status(row))" :size="13" />
            <component v-else :is="operationIcon(row)" :size="13" />
            <span>{{ operationLabel(row) }}</span>
            <span v-if="worker(row)" class="trace-worker">{{ worker(row) }}</span>
            <span v-if="duration(row)" class="trace-duration">{{ duration(row) }}</span>
            <ChevronRight :size="12" class="trace-chevron" :class="{ expanded: expanded.has(row.id) }" />
          </button>
          <div v-if="expanded.has(row.id)" class="trace-operation-details">
            <div v-if="row.event.input != null"><b>输入</b><pre>{{ formatTraceValue(row.event.input) }}</pre></div>
            <div v-if="row.event.output != null"><b>输出</b><pre>{{ formatTraceValue(row.event.output) }}</pre></div>
            <div v-for="result in row.results" :key="result.id"><b>{{ result.status === 'error' ? '错误' : '结果' }}</b><pre>{{ formatTraceValue(result.output) }}</pre></div>
            <p v-if="row.event.input == null && row.event.output == null && !row.results.length">{{ status(row) === 'running' ? '正在执行…' : '没有更多详情' }}</p>
          </div>
        </template>
      </li>
    </ol>
    <p v-if="error" class="trace-error" role="alert"><CircleAlert :size="13" /> {{ error }}</p>
  </section>
</template>

<script setup>
import { computed, reactive } from 'vue'
import { Activity, BookOpen, ChevronRight, CircleAlert, FilePenLine, Loader2, Search, Terminal, Wrench } from 'lucide-vue-next'
import { buildTraceRows, formatTraceValue, traceRowStatus } from '../utils/agent-trace.js'

const props = defineProps({
  events: { type: Array, default: () => [] },
  running: { type: Boolean, default: false },
  error: { type: String, default: '' },
  stopped: { type: Boolean, default: false },
})
const rows = computed(() => buildTraceRows(props.events))
const expanded = reactive(new Set())
const status = row => traceRowStatus(row, props)
function toggle(id) {
  if (expanded.has(id)) expanded.delete(id)
  else expanded.add(id)
}
function toolName(row) {
  return row.event.metadata?.tool || (row.event.metadata?.title || '').replace(/^调用\s+/, '')
}
function operationLabel(row) {
  if (row.event.type === 'file_operation') return `操作文件 · ${toolName(row)}`
  if (row.event.type === 'code_execution') return `执行命令 · ${toolName(row)}`
  if (row.event.type === 'tool_call') {
    const name = toolName(row)
    const labels = { kb_search: '检索知识库', web_search: '搜索网页', read_skill: '读取技能', calculator: '计算', task: '执行子任务', spawn_tasks: '执行子任务' }
    return labels[name] || `调用工具 · ${name || '工具'}`
  }
  if (row.event.type === 'tool_result') return row.event.metadata?.title || '工具返回结果'
  const stage = row.event.metadata?.stage || ''
  return ({ task_start: '执行子任务', task_end: '子任务完成', task_error: '子任务失败', task_skip: '跳过子任务' })[stage]
    || row.event.metadata?.title || (row.event.type === 'error' ? '执行遇到问题' : '处理任务')
}
function operationIcon(row) {
  if (row.event.type === 'code_execution') return Terminal
  if (row.event.type === 'file_operation') return FilePenLine
  if (toolName(row) === 'kb_search') return BookOpen
  if (toolName(row) === 'web_search') return Search
  return row.event.type.startsWith('tool') ? Wrench : Activity
}
function worker(row) {
  const span = row.event.metadata?.span
  return span && span !== 'main' ? span : ''
}
function duration(row) {
  const ms = [...row.results, row.event].find(event => event.metadata?.elapsed_ms != null)?.metadata.elapsed_ms
  return ms == null ? '' : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}
</script>

<style scoped>
.trace-process { margin: 4px 0 22px; color: var(--gray-600, #666); font-size: 13px; line-height: 1.8; }
.trace-flow { margin: 0; padding: 0; list-style: none; }
.trace-flow > li + li { margin-top: 13px; }
.trace-commentary-text { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.trace-operation-head { display: flex; align-items: center; gap: 7px; max-width: 100%; padding: 3px 0; border: 0; background: transparent; color: var(--gray-500, #7d7d7d); font: inherit; font-size: 12px; line-height: 1.6; cursor: pointer; text-align: left; }
.trace-operation-head > svg { flex-shrink: 0; }
.trace-operation-head > span { min-width: 0; overflow-wrap: anywhere; }
.trace-operation-head:hover { color: var(--gray-900, #1a1a1a); }
.trace-operation-head:focus-visible { outline: 2px solid var(--gray-500, #7d7d7d); outline-offset: 4px; border-radius: 3px; }
.trace-worker, .trace-duration { font-size: 11px; color: var(--gray-500, #7d7d7d); }
.trace-chevron { transition: transform .15s; }
.trace-chevron.expanded { transform: rotate(90deg); }
.trace-operation-details { margin: 7px 0 0 20px; padding: 8px 12px; border-left: 2px solid var(--gray-150, #e4e4e4); background: var(--gray-50, #f5f5f5); border-radius: 3px; }
.trace-operation-details b { font-size: 11px; font-weight: 500; }
.trace-operation-details pre { margin: 4px 0 10px; max-height: 260px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.7 ui-monospace, SFMono-Regular, Consolas, monospace; }
.trace-operation-details p { margin: 0; }
.trace-error, .trace-preparing { display: flex; align-items: center; gap: 7px; }
.trace-error, .status-error .trace-operation-head { color: var(--color-error-700, #a53333); }
.trace-caret { display: inline-block; width: 2px; height: 1em; margin-left: 3px; vertical-align: -.1em; background: currentColor; animation: blink 1s step-end infinite; }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes blink { 50% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .spin, .trace-caret { animation: none; } }
</style>
