<template>
  <section class="ob-page">
    <header class="ob-page-bar">
      <nav class="ob-tabs" aria-label="观测与评测">
        <button
          v-for="item in tabs"
          :key="item.id"
          :class="{ active: tab === item.id }"
          :aria-current="tab === item.id ? 'page' : undefined"
          @click="switchTab(item.id)"
        >
          <component :is="item.icon" :size="16" />
          {{ item.name }}
        </button>
      </nav>
      <span class="ob-owner">
        <ShieldCheck :size="14" />
        我的数据
      </span>
    </header>
    <div class="ob-content">
      <TraceDetail
        v-if="traceId"
        :key="traceId"
        :trace-id="traceId"
        :models="models"
        @back="backToList('traces')"
      />
      <EvaluationRunDetail
        v-else-if="runId"
        :key="runId"
        :run-id="runId"
        :item-index="itemIndex"
        @back="backToList('evaluations')"
        @item="openItem"
      />
      <template v-else>
        <div class="ob-page-heading">
          <div>
            <span class="ob-kicker">
              {{
                tab === 'workspace' || tab === 'evaluations'
                  ? 'EVALUATION'
                  : 'OBSERVABILITY'
              }}
            </span>
            <h1>{{ tabs.find((t) => t.id === tab)?.title }}</h1>
            <p>{{ tabs.find((t) => t.id === tab)?.description }}</p>
          </div>
          <button
            v-if="tab !== 'workspace'"
            class="ob-button"
            :disabled="loading"
            @click="refresh"
          >
            <RefreshCw :size="15" :class="{ spin: loading }" />
            刷新数据
          </button>
        </div>
        <p v-if="error" class="ob-error" role="alert">
          {{ error }}
          <button class="ob-text-button" @click="refresh">重试</button>
        </p>
        <template v-if="['overview', 'traces'].includes(tab)">
          <div class="ob-filters">
            <label class="ob-search">
              <Search :size="16" />
              <input
                v-model="filters.q"
                type="search"
                placeholder="搜索问题、会话或 Trace ID…"
                aria-label="搜索执行记录"
              />
            </label>
            <select v-model="filters.window" aria-label="时间范围">
              <option value="7d">最近 7 天</option>
              <option value="30d">最近 30 天</option>
              <option value="all">全部时间</option>
            </select>
            <select v-model="filters.status" aria-label="执行状态">
              <option value="">全部状态</option>
              <option
                v-for="s in [
                  'completed',
                  'running',
                  'cancel_requested',
                  'cancelled',
                  'failed',
                  'error',
                  'interrupted',
                  'timed_out',
                ]"
                :key="s"
                :value="s"
              >
                {{ statusLabel(s) }}
              </option>
            </select>
            <select v-model="filters.model" aria-label="模型筛选">
              <option value="">全部模型</option>
              <option v-for="m in modelOptions" :key="m.id" :value="m.id">
                {{ m.name }}
              </option>
            </select>
            <select v-model="filters.agent" aria-label="智能体筛选">
              <option value="">全部智能体</option>
              <option v-for="a in agents" :key="a.id" :value="a.id">
                {{ a.name }}
              </option>
            </select>
          </div>
          <p v-if="filters.conversation_id" class="ob-filter-chip">
            当前会话 {{ filters.conversation_id.slice(0, 8) }}
            <button
              aria-label="清除会话筛选"
              @click="filters.conversation_id = ''"
            >
              ×
            </button>
          </p>
          <div v-if="loading && !loaded" class="ob-empty">
            <LoaderCircle :size="26" class="spin" />
            <strong>正在读取观测数据</strong>
          </div>
          <template v-else>
            <template v-if="tab === 'overview'">
              <div class="ob-metric-grid">
                <article class="ob-stat">
                  <span>
                    执行次数
                    <Workflow :size="16" />
                  </span>
                  <strong>{{ formatNumber(summary?.total) }}</strong>
                  <small>
                    {{ summary?.status_counts?.running || 0 }} 次执行中 ·
                    {{
                      (summary?.status_counts?.failed || 0) +
                      (summary?.status_counts?.error || 0)
                    }}
                    次失败
                  </small>
                </article>
                <article class="ob-stat">
                  <span>
                    执行成功率
                    <CheckCheck :size="16" />
                  </span>
                  <strong>
                    {{
                      summary?.success_rate == null
                        ? '—'
                        : `${(summary.success_rate * 100).toFixed(1)}%`
                    }}
                  </strong>
                  <small>已完成 / 已结束的执行</small>
                </article>
                <article class="ob-stat">
                  <span>
                    平均耗时
                    <Timer :size="16" />
                  </span>
                  <strong>{{ formatMs(summary?.avg_duration_ms) }}</strong>
                  <small>按已结束执行统计</small>
                </article>
                <article class="ob-stat">
                  <span>
                    记录的 Token 用量
                    <ChartNoAxesCombined :size="16" />
                  </span>
                  <strong>{{ formatNumber(summary?.total_tokens) }}</strong>
                  <small>包含执行中的模型用量记录</small>
                </article>
              </div>
              <div class="ob-feedback-band">
                <span>
                  <ThumbsUp :size="15" />
                  <b>{{ summary?.feedback?.like || 0 }}</b>
                  次有帮助
                </span>
                <span>
                  <ThumbsDown :size="15" />
                  <b>{{ summary?.feedback?.dislike || 0 }}</b>
                  次需改进
                </span>
                <small>
                  费用未配置可靠价格，暂不汇总金额。步骤详情可查看 Langfuse
                  提供的估算。
                </small>
              </div>
              <article class="ob-card flush">
                <div class="ob-section-head padded">
                  <div>
                    <h2>最近执行</h2>
                    <p>从一个问题，追踪到每次检索与调用。</p>
                  </div>
                  <button class="ob-text-button" @click="switchTab('traces')">
                    查看全部
                    <ArrowRight :size="14" />
                  </button>
                </div>
                <TraceTable
                  :items="traces.slice(0, 5)"
                  :model-name="modelName"
                  @open="openTrace"
                />
              </article>
              <div class="ob-bottom-grid">
                <article class="ob-card">
                  <div class="ob-section-head">
                    <h2>模型用量</h2>
                    <Cpu :size="18" class="ob-accent" />
                  </div>
                  <div v-if="summary?.models?.length" class="ob-usage-list">
                    <button
                      v-for="m in summary.models"
                      :key="m.model_id"
                      @click="filterByModel(m.model_id)"
                    >
                      <span>
                        <strong>{{ modelName(m.model_id) }}</strong>
                        <small>{{ m.calls }} 次执行</small>
                      </span>
                      <span class="ob-usage-bar">
                        <i
                          :style="{
                            width: `${Math.max(2, (m.total_tokens / Math.max(1, summary.total_tokens)) * 100)}%`,
                          }"
                        ></i>
                      </span>
                      <b>{{ formatNumber(m.total_tokens) }}</b>
                    </button>
                  </div>
                  <p v-else class="ob-muted">还没有记录到模型用量。</p>
                </article>
                <article class="ob-card">
                  <div class="ob-section-head">
                    <h2>工具调用</h2>
                    <Wrench :size="18" class="ob-accent" />
                  </div>
                  <div v-if="summary?.tools?.length" class="ob-tool-list">
                    <div v-for="tool in summary.tools" :key="tool.name">
                      <code>{{ tool.name }}</code>
                      <b>{{ tool.calls }} 次</b>
                    </div>
                  </div>
                  <p v-else class="ob-muted">当前范围内没有工具调用。</p>
                  <p class="ob-muted">
                    {{
                      summary?.tools_truncated
                        ? '按最近 10,000 条调用统计。'
                        : '按本地记录的工具调用事件统计。'
                    }}失败定位可展开执行详情。
                  </p>
                </article>
              </div>
            </template>
            <article v-else class="ob-card flush">
              <div class="ob-section-head padded">
                <h2>
                  执行记录
                  <span>{{ total }} 条</span>
                </h2>
                <span class="ob-muted">每次执行拥有独立 Trace</span>
              </div>
              <TraceTable
                :items="traces"
                :model-name="modelName"
                @open="openTrace"
              />
              <div class="ob-pagination">
                <span>
                  第 {{ page }} / {{ Math.max(1, Math.ceil(total / 20)) }} 页
                </span>
                <button
                  class="ob-button"
                  :disabled="page <= 1 || loading"
                  @click="changePage(-1)"
                >
                  上一页
                </button>
                <button
                  class="ob-button"
                  :disabled="page * 20 >= total || loading"
                  @click="changePage(1)"
                >
                  下一页
                </button>
              </div>
            </article>
          </template>
        </template>
        <template v-else-if="tab === 'evaluations'">
          <div class="ob-filters">
            <label class="ob-search">
              <Search :size="16" />
              <input
                v-model="runSearch"
                type="search"
                placeholder="搜索评测名称…"
                aria-label="搜索评测运行"
              />
            </label>
            <select v-model="runKb" aria-label="评测知识库筛选">
              <option value="">全部知识库</option>
              <option v-for="kb in knowledgeBases" :key="kb.id" :value="kb.id">
                {{ kb.name }}
              </option>
            </select>
            <button class="ob-button primary" @click="switchTab('workspace')">
              <Plus :size="15" />
              新建评测
            </button>
          </div>
          <article v-if="comparison.length" class="ob-card ob-comparison">
            <div class="ob-section-head">
              <h2>
                运行对比
                <span>{{ comparison.length }} / 3</span>
              </h2>
              <button class="ob-text-button" @click="compareIds = []">
                清空选择
              </button>
            </div>
            <p class="ob-notice">
              {{
                comparableRuns(comparison) || '再选择一次运行，可并列比较指标。'
              }}
            </p>
            <div class="ob-table-wrap">
              <table class="ob-table">
                <thead>
                  <tr>
                    <th>指标</th>
                    <th v-for="run in comparison" :key="run.id">
                      <button class="ob-text-button" @click="openRun(run.id)">
                        {{ run.name }}
                      </button>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Top K / 样例数</td>
                    <td v-for="run in comparison" :key="run.id">
                      {{ run.top_k }} / {{ run.query_count }}
                    </td>
                  </tr>
                  <tr v-for="m in compareMetrics" :key="m[1]">
                    <td>{{ m[0] }}</td>
                    <td v-for="run in comparison" :key="run.id">
                      {{ formatMetric(run[m[1]]) }}
                    </td>
                  </tr>
                  <tr>
                    <td>数据集快照</td>
                    <td v-for="run in comparison" :key="run.id">
                      <code :title="run.observability?.dataset_id">
                        {{
                          run.observability?.dataset_id?.slice(0, 12) ||
                          '未记录'
                        }}
                      </code>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </article>
          <article class="ob-card flush">
            <div class="ob-section-head padded">
              <h2>
                评测运行
                <span>{{ visibleRuns.length }} 条</span>
              </h2>
              <span class="ob-muted">勾选最多 3 次运行进行对比</span>
            </div>
            <div v-if="visibleRuns.length" class="ob-table-wrap">
              <table class="ob-table">
                <thead>
                  <tr>
                    <th>对比</th>
                    <th>名称 / 知识库</th>
                    <th>样例</th>
                    <th>Recall@K</th>
                    <th>MRR@K</th>
                    <th>Experiment</th>
                    <th>时间</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="run in visibleRuns" :key="run.id">
                    <td>
                      <input
                        v-model="compareIds"
                        type="checkbox"
                        :value="run.id"
                        :disabled="
                          compareIds.length >= 3 && !compareIds.includes(run.id)
                        "
                        :aria-label="`对比${run.name}`"
                      />
                    </td>
                    <td>
                      <button class="ob-trace-link" @click="openRun(run.id)">
                        <strong>{{ run.name }}</strong>
                        <small>{{ kbName(run.knowledge_base_id) }}</small>
                        <small v-if="run.failed_items" class="ob-inline-error">
                          {{ run.failed_items }} 条样例执行失败
                        </small>
                      </button>
                    </td>
                    <td>{{ run.query_count }}</td>
                    <td>{{ formatMetric(run.recall_at_k) }}</td>
                    <td>{{ formatMetric(run.mrr_at_k) }}</td>
                    <td>
                      <span
                        class="ob-experiment-badge"
                        :class="{ linked: run.observability?.linked_items }"
                      >
                        <FlaskConical :size="13" />
                        {{
                          run.observability?.linked_items
                            ? `${run.observability.linked_items} 条已关联`
                            : '未关联'
                        }}
                      </span>
                    </td>
                    <td>{{ formatTime(run.created_at) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-else class="ob-empty">
              <FlaskConical :size="29" />
              <strong>
                {{ loading ? '正在读取评测记录' : '还没有匹配的评测运行' }}
              </strong>
              <p>在评测工作台准备数据集，并运行检索评测。</p>
              <button class="ob-button" @click="switchTab('workspace')">
                打开评测工作台
                <ArrowRight :size="14" />
              </button>
            </div>
          </article>
        </template>
        <div v-else class="ob-evaluation-workspace"><EvaluationView /></div>
      </template>
    </div>
  </section>
</template>

<script setup>
import {
  computed,
  onActivated,
  onBeforeUnmount,
  onDeactivated,
  onMounted,
  reactive,
  ref,
  watch,
} from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  ArrowRight,
  ChartNoAxesCombined,
  CheckCheck,
  Cpu,
  FlaskConical,
  LayoutDashboard,
  ListTree,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
  Timer,
  Workflow,
  Wrench,
} from 'lucide-vue-next'
import api from '../api'
import TraceDetail from '../components/TraceDetail.vue'
import TraceTable from '../components/TraceTable.vue'
import EvaluationRunDetail from '../components/EvaluationRunDetail.vue'
import EvaluationView from './EvaluationView.vue'
import {
  comparableRuns,
  formatMetric,
  formatMs,
  formatNumber,
  formatTime,
  statusLabel,
  traceFilters,
} from '../utils/observability.js'
import '../styles/observability.css'
const route = useRoute(),
  router = useRouter()
const tabs = [
  {
    id: 'overview',
    name: '总览',
    icon: LayoutDashboard,
    title: '看见每一次智能执行',
    description: '了解智能体做了什么、消耗多少用量，以及回答是否有帮助。',
  },
  {
    id: 'traces',
    name: '执行记录',
    icon: ListTree,
    title: '执行记录',
    description: '从问题定位到模型、检索、工具和子智能体的每一步。',
  },
  {
    id: 'evaluations',
    name: '评测结果',
    icon: FlaskConical,
    title: '让每次改动都有依据',
    description: '查看真实评测结果、比较历史运行，并追踪到逐题实验观测。',
  },
  {
    id: 'workspace',
    name: '评测工作台',
    icon: Settings2,
    title: '评测工作台',
    description: '准备数据集、配置基准并执行评测，复用知识库中的现有配置。',
  },
]
const tab = computed(() =>
  tabs.some((t) => t.id === route.query.tab) ? route.query.tab : 'overview',
)
const traceId = computed(() =>
  typeof route.query.trace === 'string' ? route.query.trace : '',
)
const runId = computed(() =>
  typeof route.query.run === 'string' ? route.query.run : '',
)
const itemIndex = computed(() =>
  typeof route.query.item === 'string' && /^\d+$/.test(route.query.item)
    ? Number(route.query.item)
    : null,
)
const filters = reactive({
  window: '7d',
  q: '',
  status: '',
  model: '',
  agent: '',
  conversation_id:
    typeof route.query.conversation === 'string'
      ? route.query.conversation
      : '',
})
const summary = ref(null),
  traces = ref([]),
  total = ref(0),
  page = ref(1),
  models = ref([]),
  agents = ref([]),
  runs = ref([]),
  knowledgeBases = ref([]),
  compareIds = ref([]),
  runSearch = ref(''),
  runKb = ref(''),
  loading = ref(false),
  loaded = ref(false),
  error = ref('')
let generation = 0,
  timer,
  active = true,
  activated = false
const modelOptions = computed(() => {
  const list = [...models.value]
  for (const m of summary.value?.models || [])
    if (m.model_id && !list.some((x) => x.id === m.model_id))
      list.push({ id: m.model_id, name: m.model_id })
  return list
})
const modelName = (id) =>
  models.value.find((m) => m.id === id)?.name || id || '未记录模型'
const kbName = (id) => knowledgeBases.value.find((k) => k.id === id)?.name || id
const visibleRuns = computed(() =>
  runs.value.filter(
    (r) =>
      (!runKb.value || r.knowledge_base_id === runKb.value) &&
      r.name
        .toLocaleLowerCase()
        .includes(runSearch.value.trim().toLocaleLowerCase()),
  ),
)
const comparison = computed(() =>
  compareIds.value
    .map((id) => runs.value.find((r) => r.id === id))
    .filter(Boolean),
)
const compareMetrics = [
  ['HitRate@K', 'hit_rate_at_k'],
  ['Recall@K', 'recall_at_k'],
  ['Precision@K', 'precision_at_k'],
  ['MRR@K', 'mrr_at_k'],
  ['nDCG@K', 'ndcg_at_k'],
]
function switchTab(value) {
  router.push({
    path: '/observability',
    query: value === 'overview' ? {} : { tab: value },
  })
}
function filterByModel(id) {
  filters.model = id
  switchTab('traces')
}
function changePage(delta) {
  page.value += delta
  loadTraces()
}
function openTrace(id) {
  router.push({ path: '/observability', query: { tab: 'traces', trace: id } })
}
function openRun(id) {
  router.push({
    path: '/observability',
    query: { tab: 'evaluations', run: id },
  })
}
function openItem(index) {
  router.replace({
    query: { ...route.query, item: index == null ? undefined : String(index) },
  })
}
function backToList(value) {
  router.push({ path: '/observability', query: { tab: value } })
  refresh()
}
async function loadTraces() {
  const current = ++generation
  loading.value = true
  error.value = ''
  const selected = traceFilters(filters)
  try {
    const [overview, list] = await Promise.all([
      api.get('/observability/overview', selected),
      api.get('/observability/traces', {
        ...selected,
        page: page.value,
        page_size: 20,
      }),
    ])
    if (current !== generation) return
    summary.value = overview
    traces.value = list.items
    total.value = list.total
    loaded.value = true
  } catch {
    if (current === generation)
      error.value = '观测数据读取失败，请检查连接后重试。'
  } finally {
    if (current === generation) loading.value = false
  }
}
async function loadRuns() {
  const current = ++generation
  loading.value = true
  error.value = ''
  try {
    const [list, kbs] = await Promise.all([
      api.get('/evaluation/runs'),
      api.get('/knowledge/bases'),
    ])
    if (current !== generation) return
    runs.value = list
    knowledgeBases.value = kbs
    compareIds.value = compareIds.value.filter((id) =>
      list.some((r) => r.id === id),
    )
  } catch {
    if (current === generation) error.value = '评测结果读取失败，请重试。'
  } finally {
    if (current === generation) loading.value = false
  }
}
function refresh() {
  if (traceId.value || runId.value || tab.value === 'workspace') return
  if (tab.value === 'evaluations') loadRuns()
  else loadTraces()
}
watch(
  filters,
  () => {
    page.value = 1
    clearTimeout(timer)
    timer = setTimeout(() => {
      if (
        active &&
        !traceId.value &&
        ['overview', 'traces'].includes(tab.value)
      )
        loadTraces()
    }, 250)
  },
  { deep: true },
)
watch(
  () => [route.query.tab, route.query.trace, route.query.run],
  () => {
    if (route.path === '/observability') {
      clearTimeout(timer)
      refresh()
    }
  },
)
async function loadCatalogs() {
  const results = await Promise.allSettled([
    api.get('/chat/models'),
    api.get('/agents'),
  ])
  if (!active) return
  if (results[0].status === 'fulfilled') models.value = results[0].value.models
  if (results[1].status === 'fulfilled') agents.value = results[1].value.agents
}
onMounted(() => {
  refresh()
  loadCatalogs()
})
onActivated(() => {
  active = true
  if (activated) {
    refresh()
    if (!models.value.length || !agents.value.length) loadCatalogs()
  }
  activated = true
})
onDeactivated(() => {
  active = false
  generation++
  clearTimeout(timer)
})
onBeforeUnmount(() => {
  generation++
  clearTimeout(timer)
})
</script>
