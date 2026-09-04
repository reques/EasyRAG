<template>
  <section class="agent-process" :class="{ 'is-running': running, 'is-error': !!error, 'is-collapsed': !panelOpen }" aria-live="polite">
    <!-- 折叠态：单行摘要，点击展开回看完整过程 -->
    <button
      v-if="!panelOpen"
      type="button"
      class="process-summary-line"
      :aria-expanded="false"
      @click="togglePanel"
    >
      <span class="process-summary-marker" :class="{ 'is-warning': !!error || stopped }">
        <CircleAlert v-if="error || stopped" :size="13" />
        <CheckCircle2 v-else :size="13" />
      </span>
      <span class="process-summary-text">{{ collapsedSummary }}</span>
      <ChevronRight :size="13" class="process-summary-chevron" />
    </button>

    <!-- 展开态：无边框日志流 -->
    <template v-else>
      <div v-if="error" class="process-error-line"><CircleAlert :size="12" /> {{ error }}</div>
      <div v-if="!timeline.rows.length && running" class="process-empty">正在分析你的问题…</div>

      <!-- 旧消息兜底：无步骤时间线时退化为工作日志列表 -->
      <ol v-else-if="!timeline.rows.length && summaries.length" class="process-journal">
        <li v-for="(s, si) in summaries" :key="s.id || si" :class="`phase-${s.phase || 'info'} status-${s.status || 'running'}`">
          <span class="process-journal-marker"><component :is="phaseIcon(s.phase)" :size="12" /></span>
          <span class="process-journal-text">{{ s.text }}</span>
        </li>
      </ol>

      <ol v-if="timeline.rows.length || timeline.thoughts.length" class="process-log" ref="logEl">
        <template v-for="row in displayRows" :key="row.uid">
          <!-- 思考流：弱化灰字段落，不属于步骤行 -->
          <li v-if="row.isThought" class="process-thought" :class="{ 'is-streaming': row.streaming }">
            <span class="process-thought-marker"><Brain :size="11" /></span>
            <span class="process-thought-text">{{ row.content }}<span v-if="row.streaming" class="process-thought-caret" aria-hidden="true"></span></span>
          </li>
          <!-- 步骤/工具行：⏺ 标记 + 动词 + 对象，可展开结果 -->
          <li
            v-else
            class="process-line"
            :class="[
              `wk-${row.kind}`,
              {
                'is-expanded': row.expanded,
                'is-running': row.status === 'running',
                'is-done': row.status === 'done',
                'is-fallback': row.isFallback,
              },
            ]"
          >
            <div
              class="process-line-head"
              role="button"
              tabindex="0"
              :aria-expanded="row.expanded"
              @click="toggleStep(row)"
              @keydown.enter.prevent="toggleStep(row)"
            >
              <span class="process-marker" :class="`kind-${row.kind}`">
                <Loader2 v-if="row.status === 'running'" :size="12" class="spin" />
                <component :is="row.icon" v-else :size="12" />
              </span>
              <span class="process-verb">{{ row.label }}</span>
              <span v-if="row.subagent" class="process-subagent">{{ row.subagent }}</span>
              <span v-if="row.object" class="process-object" :title="row.object">{{ row.object }}</span>
              <span class="process-side">
                <CheckCircle2 v-if="row.status === 'done' && !row.isFallback" :size="12" class="process-check" />
                <AlertTriangle v-else-if="row.isFallback" :size="12" class="process-warn" />
                <span v-if="row.durationMs != null" class="process-duration">{{ formatDuration(row.durationMs) }}</span>
                <ChevronDown v-if="row.content || row.results.length" :size="12" class="process-chevron" :class="{ flipped: !row.expanded }" />
              </span>
            </div>
            <div v-show="row.expanded && (row.content || row.results.length)" class="process-line-body" :ref="el => setBodyRef(row.uid, el)">
              <div v-if="row.content" class="process-line-content" :class="{ 'is-code': row.kind === 'tool' }">{{ row.content }}</div>
              <div v-if="row.results.length" class="process-line-results">
                <div v-for="(r, ri) in row.results" :key="ri" class="process-result" :class="{ 'is-error': r.isError }">
                  <span v-if="r.label" class="process-result-label">{{ r.label }}</span>
                  <span class="process-result-text">{{ r.text }}</span>
                </div>
              </div>
            </div>
          </li>
        </template>
      </ol>

      <!-- 收起入口：运行结束后显示 -->
      <button v-if="!running && (timeline.rows.length || timeline.thoughts.length)" type="button" class="process-collapse-entry" @click="togglePanel">
        <ChevronLeft :size="12" /> 收起过程
      </button>
    </template>
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
  WandSparkles,
  ChevronRight,
  ChevronDown,
  ChevronLeft,
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

/* ── 折叠行为：运行中强制展开；结束后自动收起为摘要；历史消息默认折叠 ── */
const panelOpen = ref(false)
const userToggled = ref(false)

watch(
  () => props.running,
  (running) => {
    if (running) {
      // 新一轮开始 → 展开实时日志（用户手动收起状态不跨轮保留，重置）
      userToggled.value = false
      panelOpen.value = true
    } else if (!userToggled.value) {
      // 运行结束 → 自动折叠为摘要（用户已手动操作则尊重其选择）
      panelOpen.value = false
    }
  },
  { immediate: true },
)

function togglePanel() {
  userToggled.value = true
  panelOpen.value = !panelOpen.value
}

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
  // 渐进式披露：模型读取 SKILL.md 激活了某个 Skill（2026-09-04）
  skill: { key: 'skill', label: '激活 Skill', icon: WandSparkles, kind: 'skill' },
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
  if (action === 'skill_activated') return STEP_DEFS.skill
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

function newRow(def, wid, startedAt, createdBy = 'artifact', order = null) {
  const s = stateFor(wid)
  if (s.startedAt == null) s.startedAt = startedAt || Date.now()
  const row = reactive({
    ownerWid: wid,
    uid: s.uid,
    createdBy,
    isThought: false,
    _order: order,
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

/* ── 链构建：按到达顺序合并步骤与产出（确定性重建，状态经 stateMap 持久）──
   思考段落独立于步骤链，收进 thoughts 弱化展示。
   ⚠ 纯派生约定：本计算内所有写操作都只发生在本轮新建的行对象上
   （构建期尚无订阅者），计算结束后不再触碰任何外部 reactive ref/行属性。
   旧版在 displayRows / applyAutoExpand watcher 里回写行的
   status/durationMs/expanded，属于「effect 修改自身依赖」，会触发
   Vue "Maximum recursive updates exceeded" 并打爆 ChatView 渲染队列
   （症状：一轮回复完成后按钮卡死、下一轮无任何输出）。 */
const timeline = computed(() => {
  const rows = []
  const thoughts = []
  const seen = new Set()
  props.items.forEach((item, order) => {
    if (!item || seen.has(item.wid)) return
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
          const r = newRow({ ...def, subagent: sg }, item.wid, item._ts || Date.now(), 'step', order)
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
          const r = newRow({ ...def, subagent: sg }, item.wid, item._ts || Date.now(), 'step', order)
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
        // 思考流 → 弱化文本段落（独立于步骤行），记录 _order 用于时序交织
        const streamId = item.id || item.wid
        const last = thoughts[thoughts.length - 1]
        if (item.streaming && last && last.streamSrc === streamId) {
          last.content += item.content || ''
        } else if (item.streaming === false && last && last.streamSrc === streamId) {
          last.streaming = false
        } else {
          thoughts.push(reactive({
            isThought: true,
            uid: `th-${streamId}`,
            streamId,
            streamSrc: streamId,
            _order: order,
            content: item.content || '',
            streaming: !!item.streaming,
          }))
        }
      } else if (kind === 'tool' || kind === 'delegate') {
        const target = findRow(rows, def.key, sg)
        if (target && target.status === 'running') {
          target.object = target.object || cleanObject(item.title || '')
          target.content = appendText(target.content, item.content)
        } else {
          const r = newRow({ ...def, subagent: sg }, item.wid, Date.now(), 'artifact', order)
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
          const r = newRow({ ...STEP_DEFS.tool, subagent: sg, label: '工具返回', icon: Terminal }, item.wid, Date.now(), 'artifact', order)
          markDone(r)
          r.results.push({ label: item.title || '', text: item.content || '' })
          rows.push(r)
        }
      } else if (kind === 'retrieve') {
        const target = findRow(rows, 'retrieve', sg)
        if (target && target.status === 'running') {
          target.content = appendText(target.content, item.content)
        } else {
          const r = newRow({ ...STEP_DEFS.retrieve, subagent: sg }, item.wid, Date.now(), 'artifact', order)
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
          const r = newRow({ ...STEP_DEFS.delegate, label: '子任务', icon: ListChecks, kind: 'worker' }, item.wid, Date.now(), 'artifact', order)
          markDone(r)
          r.results.push({ label: item.title || '', text: item.content || '' })
          rows.push(r)
        }
      } else {
        const r = newRow({ ...STEP_DEFS.info, subagent: sg, label: item.title || '步骤' }, item.wid, Date.now(), 'artifact', order)
        r.content = item.content || ''
        rows.push(r)
        closeOthers(rows, r)
      }
    }
  })
  // 收尾归一化（仅写本轮新建的行对象；stateMap 是普通 Map，不触发响应式）：
  // 整轮结束后，未配对收尾的 running 步骤统一置为 done，并冻结时长。
  const running = props.running
  for (const r of rows) {
    if (!running && r.status === 'running') {
      const s = stateFor(r.ownerWid)
      if (s.finishedAt == null) s.finishedAt = s.startedAt || Date.now()
      r.finishedAt = s.finishedAt
      r.status = 'done'
    }
    r.durationMs = stageDuration(r)
  }
  syncExpandState(rows, running)
  return { rows, thoughts }
})

/* ── 展示行合并：步骤行 + 思考段落按 items 到达顺序交织（纯派生，无回写）── */
const displayRows = computed(() => {
  const { rows, thoughts } = timeline.value
  const merged = [...rows, ...thoughts]
  merged.sort((a, b) => (a._order ?? 0) - (b._order ?? 0))
  return merged
})

/* ── 自动展开：执行中只展开最新 running 行，完成后折叠 auto 行 ──────────
   在 timeline 构建期内联调用（写入对象均为本轮新建 + 普通 stateMap），
   取代旧版 flush:'post' watcher 对行属性的回写。 */
function syncExpandState(rows, running) {
  let active = null
  if (running) {
    for (const r of rows) if (r.status === 'running') active = r
  }
  for (const r of rows) {
    const s = stateFor(r.ownerWid)
    if (!s.auto) continue
    s.expanded = running ? r === active : false
    r.expanded = s.expanded
  }
}

function toggleStep(st) {
  const s = stateFor(st.ownerWid)
  s.expanded = !s.expanded
  s.auto = false
  st.expanded = s.expanded
  st.auto = false
}

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

/* ── 流式滚动：日志增长时自动滚到底部 ──────────────────────────────── */
const logEl = ref(null)
const bodyRefs = new Map()
function setBodyRef(uid, el) {
  if (el) bodyRefs.set(uid, el)
  else bodyRefs.delete(uid)
}
function scrollLogToBottom() {
  if (!props.running) return
  nextTick(() => {
    const el = logEl.value
    if (el) {
      const scroller = el.closest('.chat-messages') || null
      if (scroller) scroller.scrollTop = scroller.scrollHeight
    }
  })
}
watch(
  () => {
    const t = timeline.value
    return [
      t.rows.length,
      t.thoughts.length,
      t.thoughts[t.thoughts.length - 1]?.content?.length || 0,
      t.rows.map(r => r.content.length).join(''),
    ]
  },
  () => scrollLogToBottom(),
)

/* ── 摘要：完成后折叠行的文案 ──────────────────────────────────────── */
const KIND_LABELS = { retrieve: '检索', tool: '工具', reason: '推理', delegate: '委派', generate: '生成' }
const collapsedSummary = computed(() => {
  const rows = timeline.value.rows
  if (!rows.length) {
    const last = props.summaries[props.summaries.length - 1]
    if (last) return props.running ? `${last.text}…` : last.text
    return props.running ? '正在执行…' : '无执行记录'
  }
  const counts = {}
  for (const r of rows) {
    const label = KIND_LABELS[r.kind]
    if (label) counts[label] = (counts[label] || 0) + 1
  }
  const parts = Object.entries(counts).map(([label, n]) => `${label} ${n} 次`)
  // 总耗时：首行开始 → 末行结束（无时间数据时省略）
  const starts = rows.map(r => r.startedAt).filter(Boolean)
  const ends = rows.map(r => r.finishedAt).filter(Boolean)
  if (starts.length && ends.length) {
    const total = Math.max(0, Math.max(...ends) - Math.min(...starts))
    if (total > 0) parts.push(formatDuration(total))
  }
  const head = props.running ? '进行中' : (props.error ? '遇到问题' : (props.stopped ? '已停止' : '已完成'))
  const body = parts.length ? parts.join(' · ') : `${rows.length} 步`
  return props.running ? `${head} · ${body}` : `${head} · ${body}`
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
/* 无边框日志流：去卡片边框/背景/阴影，贴近终端 agent 过程输出 */
.agent-process {
  width: min(100%, 720px);
  margin: 2px 0 14px;
}

/* ── 折叠态摘要行 ── */
.process-summary-line {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  max-width: 100%;
  padding: 4px 8px;
  margin-left: -8px;
  border: none;
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  font-size: 11.5px;
  color: var(--gray-500);
  transition: background .15s ease;
}
.process-summary-line:hover { background: var(--gray-25); }
.process-summary-marker { display: inline-flex; color: var(--gray-400); }
.process-summary-marker.is-warning { color: var(--color-warning-900); }
.process-summary-text {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.process-summary-chevron { flex: 0 0 auto; color: var(--gray-400); }

/* ── 展开态日志流 ── */
.process-log { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 1px; }
.process-empty { padding: 2px 0 6px; color: var(--gray-400); font-size: 12px; }
.process-error-line {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 8px;
  padding: 6px 10px;
  border-radius: 8px;
  background: var(--color-error-50);
  color: var(--color-error-700);
  font-size: 11.5px;
}

/* 步骤行 */
.process-line-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  margin-left: -8px;
  border-radius: 8px;
  cursor: pointer;
  user-select: none;
  transition: background .15s ease;
}
.process-line-head:hover { background: var(--gray-25); }
.process-marker {
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  color: var(--gray-400);
  background: var(--gray-25);
}
.process-line.is-running .process-marker { color: var(--main-600); background: var(--main-50); }
.process-line.is-fallback .process-marker { color: var(--color-warning-900); background: var(--color-warning-50); }
.process-verb { flex: 0 0 auto; font-size: 12px; font-weight: 650; color: var(--gray-800); }
.process-subagent { flex: 0 0 auto; padding: 1px 6px; border-radius: 999px; background: var(--main-50); color: var(--main-700); font-size: 9.5px; }
.process-object {
  min-width: 0;
  flex: 1 1 auto;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--gray-500);
  font-size: 11px;
}
.process-side { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 6px; }
.process-check { color: var(--color-success-600, #3f9d5f); }
.process-warn { color: var(--color-warning-900); }
.process-duration { color: var(--gray-400); font-size: 9.5px; font-variant-numeric: tabular-nums; }
.process-chevron { color: var(--gray-400); transition: transform .2s ease; }
.process-chevron.flipped { transform: rotate(-90deg); }

/* 展开体：左侧细竖线日志缩进，无底色块 */
.process-line-body {
  margin: 2px 0 6px 26px;
  padding: 2px 0 2px 10px;
  border-left: 2px solid var(--gray-150);
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 220px;
  overflow-y: auto;
  font-size: 11.5px;
  line-height: 1.6;
  color: var(--gray-600);
}
.process-line-content { white-space: pre-wrap; word-break: break-word; }
.process-line-content.is-code {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--gray-600);
}
.process-line-results { display: flex; flex-direction: column; gap: 4px; }
.process-result { display: flex; align-items: flex-start; gap: 7px; }
.process-result-label { flex: 0 0 auto; color: var(--gray-400); font-size: 10.5px; padding-top: 1px; }
.process-result-text { min-width: 0; white-space: pre-wrap; word-break: break-word; color: var(--gray-600); }
.process-result.is-error .process-result-text { color: var(--color-error-700); }

/* ── 思考流：弱化灰字段落 ── */
.process-thought {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 3px 0;
  margin-left: 10px;
}
.process-thought-marker { flex: 0 0 auto; display: inline-flex; padding-top: 3px; color: var(--gray-300); }
.process-thought-text {
  min-width: 0;
  color: var(--gray-400);
  font-size: 11.5px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}
.process-thought.is-streaming .process-thought-text { color: var(--gray-500); }
.process-thought-caret {
  display: inline-block;
  width: 2px;
  height: 11px;
  margin-left: 2px;
  vertical-align: -1px;
  border-radius: 1px;
  background: var(--gray-400);
  animation: process-caret-blink .9s steps(1) infinite;
}

/* 收起入口 */
.process-collapse-entry {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-top: 6px;
  padding: 3px 8px;
  margin-left: -8px;
  border: none;
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  font-size: 11px;
  color: var(--gray-400);
  transition: background .15s ease, color .15s ease;
}
.process-collapse-entry:hover { background: var(--gray-25); color: var(--gray-600); }

/* 旧消息兜底日志 */
.process-journal { margin: 0; padding: 0; list-style: none; }
.process-journal li { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; color: var(--gray-600); }
.process-journal-marker { display: grid; place-items: center; width: 20px; height: 20px; border-radius: 6px; color: var(--gray-400); }
.process-journal-text { min-width: 0; }

.spin { animation: process-spin 1s linear infinite; }
@keyframes process-spin { to { transform: rotate(360deg); } }
@keyframes process-caret-blink { 50% { opacity: 0; } }

@media (max-width: 640px) {
  .process-object { max-width: 140px; }
  .process-line-body { margin-left: 24px; }
}

@media (prefers-reduced-motion: reduce) {
  .process-thought-caret { animation: none; }
}
</style>
