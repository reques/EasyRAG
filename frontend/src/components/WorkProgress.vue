<template>
  <section class="work-progress" :class="{ 'is-running': running, 'is-error': !!error, 'panel-collapsed': !panelOpen }" aria-live="polite">
    <header
      class="work-progress-head"
      role="button"
      tabindex="0"
      :aria-expanded="panelOpen"
      @click="panelOpen = !panelOpen"
      @keydown.enter="panelOpen = !panelOpen"
    >
      <span class="work-progress-title"><Activity :size="14" /> 工作进度</span>
      <span class="work-progress-state">
        <span v-if="running" class="wp-state is-running"><i></i> 进行中</span>
        <span v-else-if="error" class="wp-state is-warning"><CircleAlert :size="12" /> 遇到问题</span>
        <span v-else-if="stopped" class="wp-state is-warning"><CircleAlert :size="12" /> 已停止</span>
        <span v-else class="wp-state is-done"><CheckCircle2 :size="12" /> 已完成</span>
      </span>
      <ChevronDown :size="15" class="work-progress-chevron" :class="{ flipped: !panelOpen }" />
    </header>

    <div v-if="!panelOpen" class="work-progress-collapsed">
      <span class="work-progress-collapsed-summary">{{ collapsedSummary }}</span>
    </div>

    <div v-else class="work-progress-body">
      <div v-if="!chain.length && running" class="work-progress-empty">正在分析你的问题…</div>
      <!-- 旧消息兜底：无步骤时间线时退化为工作日志列表 -->
      <ol v-else-if="!chain.length && summaries.length" class="wp-journal">
        <li v-for="(s, si) in summaries" :key="s.id || si" :class="`phase-${s.phase || 'info'} status-${s.status || 'running'}`">
          <span class="wp-journal-marker"><component :is="phaseIcon(s.phase)" :size="12" /></span>
          <span class="wp-journal-text">{{ s.text }}</span>
        </li>
      </ol>

      <ol v-if="chain.length" class="work-chain" ref="chainEl">
        <li
          v-for="st in displayChain"
          :key="st.uid"
          class="work-step"
          :class="[
            `wk-${st.kind}`,
            {
              'is-expanded': st.expanded,
              'is-running': st.status === 'running',
              'is-done': st.status === 'done',
              'is-fallback': st.isFallback,
            },
          ]"
        >
          <div
            class="work-step-head"
            role="button"
            tabindex="0"
            :aria-expanded="st.expanded"
            @click="toggleStep(st)"
            @keydown.enter="toggleStep(st)"
          >
            <span class="work-step-badge" :class="`kind-${st.kind}`">
              <component :is="st.icon" :size="13" />
            </span>
            <span class="work-step-label">
              <span class="work-step-name">{{ st.label }}</span>
              <span v-if="st.subagent" class="work-step-subagent">{{ st.subagent }}</span>
              <span v-if="st.object" class="work-step-object" :title="st.object">{{ st.object }}</span>
              <span v-if="!st.expanded && previewOf(st)" class="work-step-preview" :title="previewOf(st)">{{ previewOf(st) }}</span>
            </span>
            <span class="work-step-side">
              <span v-if="st.status === 'running'" class="work-step-spinner"><Loader2 :size="12" class="spin" /></span>
              <CheckCircle2 v-else-if="st.status === 'done' && !st.isFallback" :size="13" class="work-step-check" />
              <AlertTriangle v-else-if="st.isFallback" :size="13" class="work-step-warn" />
              <span v-if="st.streaming" class="work-step-caret" aria-hidden="true"></span>
              <span v-if="st.durationMs != null" class="work-step-duration">{{ formatDuration(st.durationMs) }}</span>
              <ChevronDown :size="13" class="work-step-chevron" :class="{ flipped: !st.expanded }" />
            </span>
          </div>
          <div v-show="st.expanded" class="work-step-body" :ref="el => setBodyRef(st.uid, el)">
            <div v-if="st.content" class="work-step-content" :class="{ 'is-code': st.kind === 'tool' }">{{ st.content }}</div>
            <div v-if="st.results.length" class="work-step-results">
              <div v-for="(r, ri) in st.results" :key="ri" class="work-step-result" :class="{ 'is-error': r.isError }">
                <span v-if="r.label" class="work-step-result-label">{{ r.label }}</span>
                <span class="work-step-result-text">{{ r.text }}</span>
              </div>
            </div>
            <div v-if="!st.content && !st.results.length && st.status === 'running'" class="work-step-placeholder">执行中…</div>
          </div>
        </li>
      </ol>
    </div>

    <div v-if="error" class="work-progress-error"><CircleAlert :size="13" /> {{ error }}</div>
  </section>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import {
  Activity,
  Loader2,
  CheckCircle2,
  CircleAlert,
  AlertTriangle,
  Brain,
  Target,
  BookOpen,
  Search,
  Wrench,
  PenLine,
  GitBranch,
  ListChecks,
  Merge,
  Workflow,
  Info,
  Terminal,
} from 'lucide-vue-next'

const props = defineProps({
  /** 有序执行时间线：[{t:'step',...}|{t:'artifact',...}]，由 ChatView 按 SSE 到达顺序维护 */
  items: { type: Array, default: () => [] },
  /** 旧消息兜底：无时间线时展示工作日志 */
  summaries: { type: Array, default: () => [] },
  /** 本轮是否仍在进行（未收到 done/error） */
  running: { type: Boolean, default: false },
  error: { type: String, default: '' },
  stopped: { type: Boolean, default: false },
})

const panelOpen = ref(true)

/* ── 步骤/产出 → 链上行定义 ─────────────────────────────────────────── */
const STEP_DEFS = {
  understand: { key: 'understand', label: '理解问题', icon: Brain, kind: 'reason' },
  reason: { key: 'reason', label: '推理', icon: Brain, kind: 'reason' },
  intent: { key: 'intent', label: '意图识别', icon: Target, kind: 'intent' },
  retrieve: { key: 'retrieve', label: '检索', icon: BookOpen, kind: 'retrieve' },
  tool: { key: 'tool', label: '工具', icon: Wrench, kind: 'tool' },
  delegate: { key: 'delegate', label: '委派', icon: GitBranch, kind: 'delegate' },
  generate: { key: 'generate', label: '生成', icon: PenLine, kind: 'generate' },
  synthesize: { key: 'synthesize', label: '汇总', icon: Merge, kind: 'synthesize' },
  fallback: { key: 'fallback', label: '回退', icon: AlertTriangle, kind: 'fallback' },
  info: { key: 'info', label: '步骤', icon: Info, kind: 'info' },
}
const SEARCH_TOOLS = new Set(['kb_search', 'web_search', 'search', 'tavily_search', 'duckduckgo'])
const DELEGATE_TOOLS = new Set(['task', 'spawn_tasks'])

function stepDef(action) {
  if (action === 'agent_reasoning' || action === 'react' || action === 'reason' || action === 'thought') {
    return STEP_DEFS.reason
  }
  if (action === 'task_started' || action === 'dispatch' || action === 'decompose') return STEP_DEFS.delegate
  if (action === 'intent_recognition') return STEP_DEFS.intent
  return STEP_DEFS[action] || { key: action || 'info', label: action || '步骤', icon: Info, kind: 'info' }
}

function toolNameOf(text) {
  const m = String(text || '').match(/调用\s*([a-zA-Z_]+)/)
  return m ? m[1] : ''
}

function artifactToDef(kind, item) {
  if (kind === 'thought') return { ...STEP_DEFS.reason, label: '推理' }
  if (kind === 'delegate') return STEP_DEFS.delegate
  if (kind === 'tool') {
    const name = toolNameOf(item.title || item.content || '')
    if (DELEGATE_TOOLS.has(name)) return STEP_DEFS.delegate
    if (SEARCH_TOOLS.has(name)) return { ...STEP_DEFS.tool, label: '搜索', icon: Search }
    return STEP_DEFS.tool
  }
  if (kind === 'tool_result') return { ...STEP_DEFS.tool, label: '工具返回', icon: Terminal }
  if (kind === 'retrieve') return STEP_DEFS.retrieve
  if (kind === 'worker') return { ...STEP_DEFS.delegate, label: '子任务', icon: ListChecks, kind: 'worker' }
  return STEP_DEFS.info
}

function parseStep(step) {
  let key = String(step || '')
  const done = /_done$/.test(key)
  const base = done ? key.slice(0, -5) : key
  let subagent = ''
  let action = base
  if (base.includes('/')) {
    const parts = base.split('/')
    subagent = parts.slice(0, -1).join('/')
    action = parts[parts.length - 1]
  }
  return { action, done, subagent }
}

function subagentFromStage(stage) {
  const s = String(stage || '')
  if (!s.includes('/')) return ''
  return s.split('/').slice(0, -1).join('/')
}

function cleanObject(text) {
  let d = String(text || '').trim()
  d = d.replace(/^调用\s*/, '')
  return d
}

function previewText(text, n = 56) {
  const flat = String(text || '').replace(/\s+/g, ' ').trim()
  return flat.length > n ? `${flat.slice(0, n)}…` : flat
}

/* ── 持久状态：行每次重建都是全新 reactive 对象，状态按 ownerWid 存这里 ── */
const stateMap = new Map()
let _uid = 0

function stateFor(wid) {
  let s = stateMap.get(wid)
  if (!s) {
    s = { startedAt: null, finishedAt: null, expanded: true, auto: true, streaming: false, streamSrc: '', uid: ++_uid }
    stateMap.set(wid, s)
  }
  return s
}

function newRow(def, wid, startedAt, createdBy = 'artifact') {
  const s = stateFor(wid)
  if (s.startedAt == null) s.startedAt = startedAt || Date.now()
  const row = reactive({
    ownerWid: wid,
    uid: s.uid,
    createdBy,
    key: def.key,
    label: def.label,
    icon: def.icon,
    kind: def.kind,
    subagent: def.subagent || '',
    object: '',
    content: '',
    results: [],
    status: 'running',
    startedAt: s.startedAt,
    finishedAt: s.finishedAt,
    isFallback: def.key === 'fallback',
    expanded: s.expanded,
    auto: s.auto,
    streaming: s.streaming,
    streamSrc: s.streamSrc,
  })
  return row
}

function markDone(row, ts) {
  row.status = 'done'
  const s = stateFor(row.ownerWid)
  if (s.finishedAt == null) s.finishedAt = ts || Date.now()
  row.finishedAt = s.finishedAt
}

function findRow(rows, key, subagent = '') {
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i]
    if (r.key !== key) continue
    if (r.subagent !== subagent) continue
    return r
  }
  return null
}

function closeOthers(rows, except) {
  for (const r of rows) {
    if (r !== except && r.status === 'running') markDone(r)
  }
}

function appendText(base, extra) {
  return [base, extra].filter(Boolean).join('\n')
}

function setStream(row, streamId) {
  const s = stateFor(row.ownerWid)
  s.streaming = true
  s.streamSrc = streamId
  row.streaming = true
  row.streamSrc = streamId
}

/* ── 链构建：按到达顺序合并步骤与产出（确定性重建，状态经 stateMap 持久） ── */
const chain = computed(() => {
  const rows = []
  const seen = new Set()
  for (const item of props.items) {
    if (!item || seen.has(item.wid)) continue
    seen.add(item.wid)
    if (item.t === 'step') {
      const { action, done, subagent } = parseStep(item.step)
      const def = stepDef(action)
      // task_started 属于主 Agent 的委派动作，归入主链路（subagent=''）
      const sg = action === 'task_started' ? '' : (item.task_id || subagent)
      if (done) {
        const row = findRow(rows, def.key, sg)
        if (row) {
          markDone(row, item._ts || Date.now())
          if (def.key !== 'tool' && item.detail) row.object = cleanObject(item.detail) || row.object
        } else {
          const r = newRow({ ...def, subagent: sg }, item.wid, item._ts || Date.now(), 'step')
          markDone(r, item._ts || Date.now())
          r.object = cleanObject(item.detail)
          rows.push(r)
        }
      } else {
        const row = findRow(rows, def.key, sg)
        // 同阶段回显（artifact 先建行，step 后到达）只更新不重复建行；
        // 仅当既有行由 step 创建且已完成后，才视为新阶段新建行
        if (row && (row.status === 'running' || row.createdBy === 'artifact')) {
          if (item.detail) row.object = cleanObject(item.detail) || row.object
        } else {
          const r = newRow({ ...def, subagent: sg }, item.wid, item._ts || Date.now(), 'step')
          r.object = cleanObject(item.detail)
          rows.push(r)
          closeOthers(rows, r)
        }
      }
    } else {
      const kind = item.kind || 'info'
      const sg = subagentFromStage(item.stage)
      const def = artifactToDef(kind, item)
      if (kind === 'thought') {
        if (item.streaming || item.streamed) {
          const streamId = item.id || item.wid
          const target = findRow(rows, 'generate') || findRow(rows, 'reason', sg)
          if (target && !target.streamSrc) {
            setStream(target, streamId)
            if (!item.streaming) target.streaming = false
            target.content = appendText(target.content, item.content)
          } else if (target && target.streamSrc === streamId) {
            if (item.content) target.content += item.content
            if (!item.streaming) target.streaming = false
          } else {
            const r = newRow({ ...STEP_DEFS.reason, subagent: sg, label: '推理' }, item.wid, Date.now())
            setStream(r, streamId)
            if (!item.streaming) r.streaming = false
            r.content = item.content || ''
            rows.push(r)
            closeOthers(rows, r)
          }
        } else {
          const target = findRow(rows, 'reason', sg)
          if (target && target.status === 'running' && !target.streaming) {
            target.content = appendText(target.content, item.content)
          } else {
            const r = newRow({ ...STEP_DEFS.reason, subagent: sg, label: '推理' }, item.wid, Date.now())
            r.content = item.content || ''
            r.object = item.title || ''
            rows.push(r)
            closeOthers(rows, r)
          }
        }
      } else if (kind === 'tool' || kind === 'delegate') {
        const target = findRow(rows, def.key, sg)
        if (target && target.status === 'running') {
          target.object = target.object || cleanObject(item.title || '')
          target.content = appendText(target.content, item.content)
        } else {
          const r = newRow({ ...def, subagent: sg }, item.wid, Date.now())
          r.object = cleanObject(item.title || '')
          r.content = item.content || ''
          rows.push(r)
          closeOthers(rows, r)
        }
      } else if (kind === 'tool_result') {
        const target = findRow(rows, 'tool', sg) || findRow(rows, 'delegate', sg)
        if (target) {
          target.results.push({
            label: item.title || '',
            text: item.content || '',
            isError: /失败|错误|error/i.test(item.title || ''),
          })
          if (target.status === 'running') markDone(target)
        } else {
          const r = newRow({ ...STEP_DEFS.tool, subagent: sg, label: '工具返回', icon: Terminal }, item.wid, Date.now())
          markDone(r)
          r.results.push({ label: item.title || '', text: item.content || '' })
          rows.push(r)
        }
      } else if (kind === 'retrieve') {
        const target = findRow(rows, 'retrieve', sg)
        if (target && target.status === 'running') {
          target.content = appendText(target.content, item.content)
        } else {
          const r = newRow({ ...STEP_DEFS.retrieve, subagent: sg }, item.wid, Date.now())
          r.content = item.content || ''
          r.object = item.title || ''
          rows.push(r)
          closeOthers(rows, r)
        }
      } else if (kind === 'worker') {
        const target = findRow(rows, 'delegate')
        if (target) {
          target.results.push({ label: item.title || '', text: item.content || '', isError: false })
          if (target.status === 'running') markDone(target)
        } else {
          const r = newRow({ ...STEP_DEFS.delegate, label: '子任务', icon: ListChecks, kind: 'worker' }, item.wid, Date.now())
          markDone(r)
          r.results.push({ label: item.title || '', text: item.content || '' })
          rows.push(r)
        }
      } else {
        const r = newRow({ ...STEP_DEFS.info, subagent: sg, label: item.title || '步骤' }, item.wid, Date.now())
        r.content = item.content || ''
        rows.push(r)
        closeOthers(rows, r)
      }
    }
  }
  return rows
})

/* ── 自动展开/折叠：执行中的步骤展开，完成后折叠，随后展开下一步 ── */
watch(
  () => [props.running, chain.value.length, chain.value.map(r => r.status).join('')],
  () => applyAutoExpand(),
  { flush: 'post', immediate: true },
)

function applyAutoExpand() {
  const rows = chain.value
  if (!props.running) {
    for (const r of rows) {
      if (!r.auto) continue
      const s = stateFor(r.ownerWid)
      s.expanded = false
      r.expanded = false
    }
    return
  }
  const actives = rows.filter(r => r.status === 'running')
  const active = actives[actives.length - 1] || null
  for (const r of rows) {
    const s = stateFor(r.ownerWid)
    if (r.status === 'running') {
      if (s.auto) {
        s.expanded = r === active
        r.expanded = s.expanded
      }
    } else if (r.status === 'done') {
      if (s.auto) {
        s.expanded = false
        r.expanded = false
      }
    }
  }
  scrollActiveBody()
}

function toggleStep(st) {
  const s = stateFor(st.ownerWid)
  s.expanded = !st.expanded
  s.auto = false
  st.expanded = s.expanded
  st.auto = false
}

const displayChain = computed(() => {
  for (const st of chain.value) {
    // 整轮结束后，未配对收尾的 running 步骤统一置为 done
    if (!props.running && st.status === 'running') {
      st.finishedAt = st.finishedAt || st.startedAt
      st.status = 'done'
    }
    st.durationMs = stageDuration(st)
  }
  return chain.value
})

/* ── 耗时计时 ──────────────────────────────────────────────────────── */
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

/* ── 流式滚动：活动步骤内容增长时自动滚到底部 ──────────────────────── */
const bodyRefs = new Map()
function setBodyRef(uid, el) {
  if (el) bodyRefs.set(uid, el)
  else bodyRefs.delete(uid)
}
function scrollActiveBody() {
  nextTick(() => {
    const actives = chain.value.filter(r => r.status === 'running')
    const active = actives[actives.length - 1]
    const el = active && bodyRefs.get(active.uid)
    if (el) el.scrollTop = el.scrollHeight
  })
}
watch(
  () => chain.value.map(r => `${r.uid}:${r.content.length}:${r.results.length}`).join('|'),
  () => { if (props.running) scrollActiveBody() },
)

/* ── 摘要与兜底日志 ────────────────────────────────────────────────── */
function previewOf(st) {
  if (st.results.length) return previewText(st.results[st.results.length - 1].text)
  return previewText(st.content)
}

const collapsedSummary = computed(() => {
  const rows = chain.value
  if (!rows.length) {
    const last = props.summaries[props.summaries.length - 1]
    if (last) return props.running ? `${last.text}…` : last.text
    return props.running ? '正在执行…' : '无执行记录'
  }
  const done = rows.filter(r => r.status === 'done').length
  return `共 ${rows.length} 步 · 已完成 ${done} 步${props.running ? ' · 进行中' : ''}`
})

const PHASE_ICONS = {
  planning: Workflow,
  retrieval: BookOpen,
  search: Search,
  action: Wrench,
  delegation: ListChecks,
  analysis: Brain,
  synthesis: Merge,
  complete: CheckCircle2,
  warning: CircleAlert,
}
function phaseIcon(phase) {
  return PHASE_ICONS[phase] || Activity
}
</script>

<style scoped>
.work-progress {
  width: min(100%, 720px);
  margin: 2px 0 16px;
  overflow: hidden;
  border: 1px solid var(--gray-150);
  border-radius: 14px;
  background: rgba(255,255,255,.94);
  box-shadow: 0 4px 18px rgba(58, 48, 40, .045);
}

.work-progress-head {
  min-height: 42px;
  padding: 0 14px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--gray-100);
  cursor: pointer;
  user-select: none;
}
.work-progress-title {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--gray-900);
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: var(--font-display-weight);
}
.work-progress-title svg { color: var(--main-600); }
.work-progress-state { display: inline-flex; align-items: center; margin-left: auto; }
.wp-state { display: inline-flex; align-items: center; gap: 6px; color: var(--main-600); font-size: 10px; font-weight: 650; }
.wp-state i { width: 6px; height: 6px; border-radius: 50%; background: var(--main-500); box-shadow: 0 0 0 3px var(--main-50); animation: wp-pulse 1.8s ease-in-out infinite; }
.wp-state.is-done { color: var(--gray-500); }
.wp-state.is-warning { color: var(--color-warning-900); }
.work-progress-chevron { color: var(--gray-400); transition: transform .2s ease; }
.work-progress-chevron.flipped { transform: rotate(180deg); }
.work-progress-head:hover .work-progress-chevron { color: var(--gray-600); }

.work-progress-collapsed { padding: 8px 14px; font-size: 11px; color: var(--gray-500); }
.work-progress-collapsed-summary { display: inline-flex; align-items: center; gap: 6px; }

.work-progress-body { padding: 8px 0 6px; }
.work-progress-empty { padding: 10px 14px; color: var(--gray-400); font-size: 12px; }

/* 旧消息兜底日志 */
.wp-journal { margin: 0; padding: 2px 14px 6px; list-style: none; }
.wp-journal li { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 12px; color: var(--gray-700); }
.wp-journal-marker { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 6px; border: 1px solid var(--main-100); color: var(--main-600); }
.wp-journal-text { min-width: 0; }

/* 步骤链 */
.work-chain { margin: 0; padding: 0 10px; list-style: none; }
.work-step { position: relative; }
.work-step:not(:last-child)::after {
  content: '';
  position: absolute;
  top: 34px;
  bottom: -4px;
  left: 19px;
  width: 1px;
  background: var(--gray-150);
}
.work-step-head {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 6px;
  border-radius: 9px;
  cursor: pointer;
  user-select: none;
  transition: background .15s ease;
}
.work-step-head:hover { background: var(--gray-25); }
.work-step-badge {
  position: relative;
  z-index: 1;
  flex: 0 0 auto;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: 1px solid var(--main-100);
  border-radius: 8px;
  background: #fff;
  color: var(--main-600);
}
.work-step.is-running .work-step-badge { border-color: var(--main-200); color: var(--main-700); box-shadow: 0 0 0 3px var(--main-50); }
.work-step.is-done .work-step-badge { background: var(--main-50); }
.work-step.is-fallback .work-step-badge { border-color: #f0dcae; background: var(--color-warning-50); color: var(--color-warning-900); }

.work-step-label {
  min-width: 0;
  flex: 1 1 auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--gray-800);
  white-space: nowrap;
}
.work-step-name { font-weight: 650; }
.work-step-subagent { flex: 0 0 auto; padding: 1px 6px; border-radius: 999px; background: var(--main-50); color: var(--main-700); font-size: 9.5px; }
.work-step-object {
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--gray-500);
  font-size: 11px;
}
.work-step-preview {
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--gray-400);
  font-size: 10.5px;
  max-width: 240px;
}
.work-step-side { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 7px; }
.work-step-spinner { color: var(--main-500); display: inline-flex; }
.work-step-check { color: var(--color-success-600, #3f9d5f); }
.work-step-warn { color: var(--color-warning-900); }
.work-step-caret {
  width: 2px;
  height: 11px;
  border-radius: 1px;
  background: var(--main-400);
  animation: wp-caret-blink .9s steps(1) infinite;
}
.work-step-duration { color: var(--gray-400); font-size: 9px; font-variant-numeric: tabular-nums; }
.work-step-chevron { color: var(--gray-400); transition: transform .2s ease; }
.work-step-chevron.flipped { transform: rotate(-90deg); }

.work-step-body {
  margin: 2px 6px 6px 40px;
  max-height: 220px;
  overflow-y: auto;
  padding: 8px 10px;
  border-radius: 10px;
  background: var(--gray-25);
  font-size: 12px;
  line-height: 1.6;
  color: var(--gray-600);
}
.work-step-content { white-space: pre-wrap; word-break: break-word; }
.work-step-content.is-code {
  font-family: var(--font-mono);
  font-size: 11px;
  background: var(--gray-50);
  border: 1px solid var(--gray-100);
  border-radius: 6px;
  padding: 6px 8px;
  color: var(--gray-700);
}
.work-step-results { display: flex; flex-direction: column; gap: 6px; }
.work-step-result { display: flex; align-items: flex-start; gap: 7px; }
.work-step-result-label { flex: 0 0 auto; color: var(--gray-400); font-size: 10.5px; padding-top: 1px; }
.work-step-result-text { min-width: 0; color: var(--gray-600); }
.work-step-result.is-error .work-step-result-text { color: var(--color-error-700); }
.work-step-placeholder { color: var(--gray-400); font-style: italic; }

.work-progress-error { margin: 0 14px 12px; padding: 8px 10px; display: flex; align-items: center; gap: 6px; border-radius: 8px; background: var(--color-error-50); color: var(--color-error-700); font-size: 11px; }

.spin { animation: wp-spin 1s linear infinite; }
@keyframes wp-spin { to { transform: rotate(360deg); } }
@keyframes wp-pulse {
  0%, 100% { opacity: .55; transform: scale(.92); }
  50% { opacity: 1; transform: scale(1); }
}
@keyframes wp-caret-blink { 50% { opacity: 0; } }

@media (max-width: 640px) {
  .work-progress { border-radius: 12px; }
  .work-step-object, .work-step-preview { max-width: 120px; }
  .work-step-body { margin-left: 34px; }
}

@media (prefers-reduced-motion: reduce) {
  .wp-state i { animation: none; }
}
</style>
