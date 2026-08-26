<template>
  <section class="progress-journal" :class="{ 'is-running': running, 'is-error': !!error }" aria-live="polite">
    <header class="progress-journal-head">
      <span class="progress-journal-title"><Activity :size="14" /> 工作进度</span>
      <span v-if="running" class="progress-journal-state"><i></i> 进行中</span>
      <span v-else-if="error" class="progress-journal-state is-warning"><CircleAlert :size="12" /> 遇到问题</span>
      <span v-else-if="stopped" class="progress-journal-state is-warning"><CircleAlert :size="12" /> 已停止</span>
      <span v-else class="progress-journal-state is-complete"><CheckCircle2 :size="12" /> 已完成</span>
    </header>

    <ol class="progress-journal-list">
      <li
        v-for="(item, index) in items"
        :key="item.id || `${item.sequence || index}-${item.text}`"
        :class="[`phase-${item.phase || 'info'}`, `status-${item.status || 'running'}`]"
      >
        <span class="progress-journal-marker">
          <component :is="phaseIcon(item.phase)" :size="13" />
        </span>
        <div class="progress-journal-entry">
          <p>{{ item.text }}</p>
          <time v-if="formatTime(item.created_at || item._ts)">{{ formatTime(item.created_at || item._ts) }}</time>
        </div>
      </li>
    </ol>

    <div v-if="error" class="progress-journal-error">
      <CircleAlert :size="13" /> {{ error }}
    </div>
  </section>
</template>

<script setup>
import {
  Activity,
  BookOpen,
  CheckCircle2,
  CircleAlert,
  FileSearch2,
  ListTree,
  Search,
  Sparkles,
  Wrench,
} from 'lucide-vue-next'

defineProps({
  items: { type: Array, default: () => [] },
  running: { type: Boolean, default: false },
  error: { type: String, default: '' },
  stopped: { type: Boolean, default: false },
})

const ICONS = {
  planning: ListTree,
  retrieval: BookOpen,
  search: Search,
  action: Wrench,
  delegation: ListTree,
  analysis: FileSearch2,
  synthesis: Sparkles,
  complete: CheckCircle2,
  warning: CircleAlert,
}

function phaseIcon(phase) {
  return ICONS[phase] || Activity
}

function formatTime(value) {
  if (!value) return ''
  const numeric = Number(value)
  const millis = Number.isFinite(numeric) && numeric < 1e12 ? numeric * 1000 : numeric
  const date = Number.isFinite(millis) ? new Date(millis) : new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}
</script>

<style scoped>
.progress-journal {
  width: min(100%, 720px);
  margin: 2px 0 18px;
  overflow: hidden;
  border: 1px solid var(--gray-150);
  border-radius: 14px;
  background: linear-gradient(145deg, rgba(255,255,255,.96), var(--main-20));
  box-shadow: 0 8px 28px rgba(23, 60, 52, .05);
}

.progress-journal-head {
  min-height: 42px;
  padding: 0 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--gray-100);
}
.progress-journal-title { display: inline-flex; align-items: center; gap: 7px; color: var(--gray-900); font-size: 12px; font-weight: 720; }
.progress-journal-title svg { color: var(--main-600); }
.progress-journal-state { display: inline-flex; align-items: center; gap: 6px; color: var(--main-600); font-size: 10px; font-weight: 650; }
.progress-journal-state i { width: 6px; height: 6px; border-radius: 50%; background: var(--main-500); box-shadow: 0 0 0 3px var(--main-50); animation: progress-pulse 1.8s ease-in-out infinite; }
.progress-journal-state.is-complete { color: var(--gray-500); }
.progress-journal-state.is-warning { color: var(--color-warning-900); }

.progress-journal-list { margin: 0; padding: 12px 14px 13px; list-style: none; }
.progress-journal-list li { position: relative; min-height: 46px; display: grid; grid-template-columns: 26px minmax(0, 1fr); gap: 9px; }
.progress-journal-list li:not(:last-child)::after { content: ''; position: absolute; top: 25px; bottom: 2px; left: 12px; width: 1px; background: var(--gray-150); }
.progress-journal-marker { position: relative; z-index: 1; width: 25px; height: 25px; display: grid; place-items: center; border: 1px solid var(--main-100); border-radius: 8px; background: #fff; color: var(--main-600); }
.progress-journal-entry { min-width: 0; padding: 2px 0 10px; display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.progress-journal-entry p { margin: 0; color: var(--gray-700); font-size: 12px; line-height: 1.65; }
.progress-journal-entry time { flex: 0 0 auto; padding-top: 2px; color: var(--gray-400); font-size: 9px; font-variant-numeric: tabular-nums; }
.progress-journal-list li.status-warning .progress-journal-marker { border-color: #f0dcae; background: var(--color-warning-50); color: var(--color-warning-900); }
.progress-journal-list li.status-completed .progress-journal-marker { background: var(--main-50); }
.progress-journal-error { margin: -2px 14px 13px; padding: 8px 10px; display: flex; align-items: center; gap: 6px; border-radius: 8px; background: var(--color-error-50); color: var(--color-error-700); font-size: 11px; }

@keyframes progress-pulse {
  0%, 100% { opacity: .55; transform: scale(.92); }
  50% { opacity: 1; transform: scale(1); }
}

@media (max-width: 640px) {
  .progress-journal { border-radius: 12px; }
  .progress-journal-entry { flex-direction: column; gap: 2px; }
  .progress-journal-entry time { padding: 0; }
}

@media (prefers-reduced-motion: reduce) {
  .progress-journal-state i { animation: none; }
}
</style>
