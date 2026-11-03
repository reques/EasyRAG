<template>
  <section class="ob-detail">
    <div class="ob-detail-toolbar">
      <button class="ob-button" @click="$emit('back')">
        <ArrowLeft :size="15" />
        返回评测结果
      </button>
      <button class="ob-button" :disabled="loading" @click="load">
        <RefreshCw :size="15" :class="{ spin: loading }" />
        刷新
      </button>
    </div>
    <p v-if="error" class="ob-error" role="alert">{{ error }}</p>
    <div v-if="loading && !run" class="ob-empty">
      <LoaderCircle :size="25" class="spin" />
      <strong>正在读取评测运行</strong>
    </div>
    <template v-if="run">
      <header class="ob-detail-header">
        <div>
          <span class="ob-kicker">EVALUATION / {{ run.id }}</span>
          <h1>{{ run.name }}</h1>
          <p>
            {{ formatTime(run.created_at) }} · {{ run.query_count }} 条样例 ·
            Top K {{ run.top_k }} · {{ statusLabel(run.status) }}
          </p>
        </div>
        <button class="ob-button" @click="download">
          <Download :size="15" />
          导出报告
        </button>
      </header>
      <p v-if="downloadError" class="ob-error">{{ downloadError }}</p>
      <EvaluationObservability
        :value="run.metrics?.observability"
        :show-native-link="false"
      />
      <p v-if="run.failed_items" class="ob-error">
        本次运行有
        {{
          run.failed_items
        }}
        条样例执行失败，汇总指标包含这些失败结果。请结合逐题错误判断指标变化。
      </p>
      <div class="ob-metric-grid six">
        <article v-for="metric in metrics" :key="metric[0]" class="ob-stat">
          <span>{{ metric[0] }}</span>
          <strong>
            {{ formatMetric(run[metric[1]] ?? run.metrics?.[metric[1]]) }}
          </strong>
        </article>
      </div>
      <article
        v-if="Object.keys(run.metrics?.ragas?.metrics || {}).length"
        class="ob-card"
      >
        <div class="ob-section-head">
          <h2>RAGAs 指标</h2>
          <span>{{ run.metrics.ragas.status }}</span>
        </div>
        <div class="ob-detail-metrics">
          <span v-for="(value, key) in run.metrics.ragas.metrics" :key="key">
            {{ key }}
            <b>{{ formatMetric(value) }}</b>
          </span>
        </div>
      </article>
      <p v-else class="ob-muted">
        RAGAs：{{ run.ragas_status || '本次运行未记录语义评分' }}
      </p>
      <article class="ob-card flush">
        <div class="ob-section-head padded">
          <div>
            <h2>逐题结果</h2>
            <p>从每条样例打开它实际运行时采集的观测与评分。</p>
          </div>
          <span class="ob-muted">{{ run.details.length }} 条</span>
        </div>
        <div class="ob-table-wrap">
          <table class="ob-table">
            <thead>
              <tr>
                <th>#</th>
                <th>问题</th>
                <th>命中排名</th>
                <th>Recall@K</th>
                <th>MRR@K</th>
                <th>观测</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(item, index) in run.details"
                :key="index"
                :class="{ 'ob-selected-row': itemIndex === index }"
              >
                <td>{{ index + 1 }}</td>
                <td class="ob-case-question">
                  {{ item.question }}
                  <small
                    v-if="item.retrieval_error || item.generation_error"
                    class="ob-error"
                  >
                    {{ item.retrieval_error || item.generation_error }}
                  </small>
                </td>
                <td>
                  {{
                    item.chunk_hit_rank
                      ? `#${item.chunk_hit_rank}`
                      : item.file_hit_rank
                        ? `文件 #${item.file_hit_rank}`
                        : '未命中'
                  }}
                </td>
                <td>{{ formatMetric(item.chunk_metrics?.recall_at_k) }}</td>
                <td>
                  {{ formatMetric(item.chunk_metrics?.reciprocal_rank_at_k) }}
                </td>
                <td>
                  <button class="ob-text-button" @click="$emit('item', index)">
                    查看步骤
                    <ArrowUpRight :size="13" />
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
      <section v-if="itemIndex != null" class="ob-eval-item">
        <div class="ob-section-head">
          <h2>样例 {{ itemIndex + 1 }} 的执行观测</h2>
          <button class="ob-text-button" @click="$emit('item', null)">
            收起
          </button>
        </div>
        <p v-if="itemError" class="ob-error" role="alert">{{ itemError }}</p>
        <ObservationWaterfall
          :remote="itemData?.remote || { state: 'loading' }"
        />
        <article v-if="itemData?.scores?.length" class="ob-card">
          <div class="ob-section-head">
            <h2>样例评分</h2>
            <span class="ob-muted">对应本次 Experiment 的样例根节点</span>
          </div>
          <div class="ob-score-list">
            <div v-for="score in itemData.scores" :key="score.id">
              <strong>{{ score.name }}</strong>
              <b>{{ formatMetric(score.value) }}</b>
              <em>
                {{
                  score.delivery_state === 'accepted'
                    ? '远端已接收'
                    : '等待同步'
                }}
              </em>
            </div>
          </div>
        </article>
        <details v-if="itemData?.case" class="ob-card">
          <summary>样例参考与结果数据</summary>
          <pre class="ob-json">{{ pretty(itemData.case) }}</pre>
        </details>
      </section>
    </template>
  </section>
</template>
<script setup>
import { onBeforeUnmount, ref, watch } from 'vue'
import {
  ArrowLeft,
  ArrowUpRight,
  Download,
  LoaderCircle,
  RefreshCw,
} from 'lucide-vue-next'
import api from '../api'
import ObservationWaterfall from './ObservationWaterfall.vue'
import EvaluationObservability from './EvaluationObservability.vue'
import {
  formatMetric,
  formatTime,
  pretty,
  statusLabel,
} from '../utils/observability.js'
const props = defineProps({
  runId: { type: String, required: true },
  itemIndex: { type: Number, default: null },
})
defineEmits(['back', 'item'])
const run = ref(null),
  loading = ref(false),
  error = ref(''),
  itemData = ref(null),
  itemError = ref(''),
  downloadError = ref('')
const metrics = [
  ['HitRate@K', 'hit_rate_at_k'],
  ['MRR@K', 'mrr_at_k'],
  ['Recall@K', 'recall_at_k'],
  ['Precision@K', 'precision_at_k'],
  ['nDCG@K', 'ndcg_at_k'],
  ['平均检索得分', 'avg_score'],
]
let generation = 0,
  itemGeneration = 0
async function load() {
  const current = ++generation
  loading.value = true
  error.value = ''
  try {
    const value = await api.get(
      `/evaluation/runs/${encodeURIComponent(props.runId)}`,
    )
    if (current !== generation) return
    run.value = value
    await loadItem()
  } catch {
    if (current === generation) error.value = '无法读取此评测运行'
  } finally {
    if (current === generation) loading.value = false
  }
}
async function loadItem() {
  const current = ++itemGeneration
  itemData.value = null
  itemError.value = ''
  if (props.itemIndex == null) return
  try {
    const value = await api.get(
      `/observability/evaluations/${encodeURIComponent(props.runId)}/items/${props.itemIndex}`,
    )
    if (current === itemGeneration) itemData.value = value
  } catch {
    if (current === itemGeneration) {
      itemError.value = '无法读取此样例的观测'
      itemData.value = { remote: { state: 'unavailable' } }
    }
  }
}
async function download() {
  downloadError.value = ''
  try {
    const { data } = await api.getBlob(
      `/evaluation/runs/${encodeURIComponent(props.runId)}/report`,
    )
    const url = URL.createObjectURL(data),
      a = document.createElement('a')
    a.href = url
    a.download = `evaluation-${props.runId}.md`
    a.click()
    URL.revokeObjectURL(url)
  } catch {
    downloadError.value = '报告导出失败，请重试'
  }
}
watch(
  () => props.runId,
  () => {
    run.value = null
    itemGeneration++
    load()
  },
  { immediate: true },
)
watch(() => props.itemIndex, loadItem)
onBeforeUnmount(() => {
  generation++
  itemGeneration++
})
</script>
