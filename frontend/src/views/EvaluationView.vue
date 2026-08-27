<template>
  <div class="ev-shell" :class="{ 'ev-embedded': embedded }">
    <!-- ── 页头 ── -->
    <header v-if="!embedded" class="ev-header">
      <div>
        <span class="ev-eyebrow">RETRIEVAL EVALUATION</span>
        <h1>检索评估</h1>
        <p>对知识库逐条执行真实检索，输出命中率 / MRR / RAGAs 指标，并沉淀为可对比的历史运行。</p>
      </div>
    </header>

    <!-- ── 配置区 ── -->
    <section class="ev-card">
      <div class="ev-card-head">
        <h2>评估配置</h2>
        <span class="ev-hint">选择知识库后，可为每条用例指定期望命中的文件</span>
      </div>
      <div class="ev-config-grid">
        <label v-if="!embedded" class="ev-field">
          <span>知识库</span>
          <select v-model="form.kbId" @change="onKbChange" :disabled="running">
            <option value="" disabled>请选择知识库</option>
            <option v-for="kb in kbList" :key="kb.id" :value="kb.id">{{ kb.name }}</option>
          </select>
        </label>
        <label v-else class="ev-field">
          <input :value="kbNameLabel" disabled />
        </label>
        <label class="ev-field">
          <span>返回条数 top_k</span>
          <input v-model.number="form.topK" type="number" min="1" max="20" :disabled="running" />
        </label>
        <label class="ev-field ev-field-wide">
          <span>运行名称</span>
          <input v-model="form.name" placeholder="例如：legal-recursive-v2" :disabled="running" />
        </label>
      </div>
      <div class="ev-dataset-bar">
        <label class="ev-field">
          <span>评测集（Golden Set）</span>
          <select v-model="datasetSel" @change="onDatasetSelect" :disabled="running">
            <option value="" disabled>选择已保存评测集</option>
            <option v-for="ds in datasets" :key="ds.id" :value="ds.id">
              {{ ds.name }}（v{{ ds.version }} · {{ ds.case_count }} 条）
            </option>
          </select>
        </label>
        <button class="ev-btn-ghost" @click="saveDataset" :disabled="running || !validCases || !form.kbId">
          <Save :size="14" /> 保存当前用例为评测集
        </button>
        <span v-if="datasetHint" class="ev-hint">{{ datasetHint }}</span>
      </div>
    </section>

    <!-- ── 用例编辑 ── -->
    <section class="ev-card">
      <div class="ev-card-head">
        <h2>测试用例</h2>
        <button class="ev-btn-ghost" @click="addCase" :disabled="running || !form.kbId">
          <Plus :size="14" /> 添加用例
        </button>
      </div>

      <div v-if="!form.kbId" class="ev-empty">请先选择知识库，再编辑用例</div>

      <div v-else-if="cases.length === 0" class="ev-empty">还没有用例，点击「添加用例」开始</div>

      <div v-else class="ev-case-list">
        <div v-for="(c, idx) in cases" :key="idx" class="ev-case-row">
          <div class="ev-case-index">{{ idx + 1 }}</div>
          <label class="ev-field ev-field-grow">
            <span>问题</span>
            <input v-model="c.question" placeholder="例如：食品安全法第一百四十八条" :disabled="running" />
          </label>
          <label class="ev-field">
            <span>期望命中的文件</span>
            <select v-model="c.expected_file_id" :disabled="running">
              <option value="" disabled>选择文件</option>
              <option v-for="f in files" :key="f.id" :value="f.id">{{ f.filename }}</option>
            </select>
          </label>
          <label class="ev-field">
            <span>参考答案（可选）</span>
            <input v-model="c.reference_answer" placeholder="用于 RAGAs LLM 指标，可留空" :disabled="running" />
          </label>
          <label class="ev-field ev-check">
            <span>负样本</span>
            <input type="checkbox" v-model="c.expect_miss" :disabled="running" />
          </label>
          <button class="ev-btn-ghost" @click="loadCandidates(idx)" :disabled="running || !c.question.trim() || !c.expected_file_id">
            <Search :size="14" /> 候选
          </button>
          <button class="ev-btn-ghost ev-btn-danger" @click="removeCase(idx)" :disabled="running">
            <Trash2 :size="14" />
          </button>
        </div>
        <div v-if="c.candidates !== null" class="ev-candidates">
          <div class="ev-candidates-head">
            <span>勾选真正回答该问题所需的 chunk 作为相关集（已选 {{ (c.expected_chunk_ids || []).length }} 条）</span>
            <button class="ev-btn-ghost" @click="closeCandidates(idx)"><X :size="12" /> 关闭</button>
          </div>
          <div class="ev-candidates-list">
            <label v-for="(cd, j) in c.candidates" :key="j" class="ev-cand-item">
              <input type="checkbox" :value="cd.chunk_id" v-model="c.expected_chunk_ids" />
              <span class="ev-cand-score">{{ cd.score.toFixed(3) }}</span>
              <span class="ev-cand-snippet">{{ cd.snippet }}</span>
            </label>
          </div>
          <p v-if="c.candidates.length === 0" class="ev-hint">该文件暂无候选 chunk，可尝试换问题或调大 top_k</p>
        </div>
      </div>

      <div v-if="form.kbId && cases.length" class="ev-run-bar">
        <button class="ev-btn-primary" @click="run" :disabled="running || !validCases">
          <LoaderCircle v-if="running" :size="15" class="spin" />
          <Play v-else :size="15" />
          {{ running ? '评估运行中…' : '运行评估' }}
        </button>
        <span v-if="runError" class="ev-error">{{ runError }}</span>
      </div>
    </section>

    <!-- ── 结果展示 ── -->
    <section v-if="result" class="ev-card">
      <div class="ev-card-head">
        <h2>本次结果</h2>
        <div class="ev-head-actions">
          <button v-if="result.id" class="ev-btn-ghost" @click="downloadReport(result.id)">
            <Download :size="14" /> 下载报告
          </button>
          <span v-if="result.created_at" class="ev-hint">{{ formatTime(result.created_at) }}</span>
        </div>
      </div>

      <div class="ev-metric-grid">
        <div class="ev-metric">
          <span>HitRate@K</span>
          <strong>{{ fmt(result.hit_rate_at_k) }}</strong>
          <small>文件级 {{ fmt(result.file_hit_rate_at_k) }}</small>
        </div>
        <div class="ev-metric">
          <span>MRR@K</span>
          <strong>{{ fmt(result.mrr_at_k) }}</strong>
          <small>文件级 {{ fmt(result.file_mrr_at_k) }}</small>
        </div>
        <div class="ev-metric">
          <span>Recall@K</span>
          <strong>{{ fmt(result.recall_at_k) }}</strong>
          <small>文件级 {{ fmt(result.file_recall_at_k) }}</small>
        </div>
        <div class="ev-metric">
          <span>Precision@K</span>
          <strong>{{ fmt(result.precision_at_k) }}</strong>
          <small>文件级 {{ fmt(result.file_precision_at_k) }}</small>
        </div>
        <div class="ev-metric">
          <span>nDCG@K</span>
          <strong>{{ fmt(result.ndcg_at_k) }}</strong>
          <small>文件级 {{ fmt(result.file_ndcg_at_k) }}</small>
        </div>
        <div class="ev-metric">
          <span>平均得分</span>
          <strong>{{ fmt(result.avg_score) }}</strong>
          <small>{{ result.k }} 条 / 查询</small>
        </div>
      </div>

      <!-- RAGAs -->
      <div class="ev-ragas">
        <div class="ev-ragas-head">
          <span class="ev-ragas-title">RAGAs 指标</span>
          <span v-if="ragasStatus" class="ev-badge" :class="'is-' + ragasStatus">
            {{ ragasStatusLabel }}
          </span>
          <span v-if="ragas.ragas_version" class="ev-hint">ragas {{ ragas.ragas_version }}</span>
        </div>

        <div v-if="ragasStatus === 'completed' || ragasStatus === 'partial'" class="ev-metric-grid ev-metric-grid-sm">
          <div v-for="(value, key) in ragas.metrics" :key="key" class="ev-metric">
            <span>{{ metricLabel(key) }}</span>
            <strong>{{ value === null || value === undefined ? '—' : fmt(value) }}</strong>
          </div>
        </div>

        <div v-else-if="ragas.error" class="ev-ragas-error">
          <CircleAlert :size="15" /> {{ ragas.error }}
        </div>
        <div v-else class="ev-ragas-error">
          <CircleAlert :size="15" /> RAGAs 未启用或不可用（{{ ragasStatusLabel }}）
        </div>
      </div>

      <!-- 逐条明细 -->
      <h3 class="ev-subhead">逐条明细</h3>
      <div class="ev-table-wrap">
        <table class="ev-table">
          <thead>
            <tr>
              <th>#</th>
              <th>问题</th>
              <th>参考类型</th>
              <th>期望文件</th>
              <th>命中排名</th>
              <th>Top 得分</th>
              <th>返回数</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(d, i) in result.details" :key="i">
              <td>{{ i + 1 }}</td>
              <td class="ev-cell-q">{{ d.question }}</td>
              <td>{{ referenceModeLabel(d.reference_mode) }}</td>
              <td>{{ fileName(d.expected_file_id) }}</td>
              <td>
                <span v-if="d.chunk_hit_rank" class="ev-hit">#{{ d.chunk_hit_rank }}</span>
                <span v-else-if="d.file_hit_rank" class="ev-hit">文件#{{ d.file_hit_rank }}</span>
                <span v-else class="ev-miss">未命中</span>
              </td>
              <td>{{ fmt(d.top_score) }}</td>
              <td>{{ d.returned }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ── 历史运行 ── -->
    <section class="ev-card">
      <div class="ev-card-head">
        <h2>历史运行</h2>
        <button class="ev-btn-ghost" @click="loadHistory" :disabled="loadingHistory">
          <RefreshCw :size="14" :class="{ spin: loadingHistory }" /> 刷新
        </button>
      </div>

      <div v-if="history.length === 0" class="ev-empty">暂无评估记录</div>

      <div v-else class="ev-table-wrap">
        <table class="ev-table">
          <thead>
            <tr>
              <th>名称</th>
              <th v-if="!embedded">知识库</th>
              <th>K</th>
              <th>查询数</th>
              <th>HitRate</th>
              <th>MRR</th>
              <th>RAGAs 状态</th>
              <th>时间</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in history" :key="r.id">
              <td class="ev-cell-name">{{ r.name }}</td>
              <td v-if="!embedded">{{ kbName(r.knowledge_base_id) }}</td>
              <td>{{ r.top_k }}</td>
              <td>{{ r.query_count }}</td>
              <td>{{ fmt(r.hit_rate) }}</td>
              <td>{{ fmt(r.mrr) }}</td>
              <td><span class="ev-badge" :class="'is-' + historyRagasStatus(r)">{{ historyRagasStatus(r) }}</span></td>
              <td>{{ formatTime(r.created_at) }}</td>
              <td><button class="ev-btn-ghost" @click="viewRun(r.id)"><Eye :size="14" /> 查看</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ── 历史详情弹层 ── -->
    <Teleport to="body">
      <div v-if="detail" class="ev-modal-overlay" @click.self="detail = null">
        <div class="ev-modal">
          <div class="ev-modal-head">
            <h3>{{ detail.name }}</h3>
            <div class="ev-head-actions">
              <button class="ev-btn-ghost" @click="downloadReport(detail.id)"><Download :size="14" /> 报告</button>
              <button class="ev-btn-ghost" @click="detail = null"><X :size="16" /></button>
            </div>
          </div>
          <div v-if="detail.metrics" class="ev-metric-grid ev-metric-grid-sm">
            <div class="ev-metric">
              <span>HitRate@K</span>
              <strong>{{ fmt(detail.metrics.hit_rate_at_k) }}</strong>
            </div>
            <div class="ev-metric">
              <span>MRR@K</span>
              <strong>{{ fmt(detail.metrics.mrr_at_k) }}</strong>
            </div>
            <div v-for="(value, key) in (detail.metrics.ragas && detail.metrics.ragas.metrics || {})" :key="key" class="ev-metric">
              <span>{{ metricLabel(key) }}</span>
              <strong>{{ fmt(value) }}</strong>
            </div>
          </div>
          <div v-if="detail.details && detail.details.length" class="ev-table-wrap ev-modal-table">
            <table class="ev-table">
              <thead>
                <tr><th>#</th><th>问题</th><th>命中排名</th><th>Top 得分</th></tr>
              </thead>
              <tbody>
                <tr v-for="(d, i) in detail.details" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td class="ev-cell-q">{{ d.question }}</td>
                  <td>
                    <span v-if="d.chunk_hit_rank" class="ev-hit">#{{ d.chunk_hit_rank }}</span>
                    <span v-else-if="d.file_hit_rank" class="ev-hit">文件#{{ d.file_hit_rank }}</span>
                    <span v-else class="ev-miss">未命中</span>
                  </td>
                  <td>{{ fmt(d.top_score) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { Plus, Trash2, Play, LoaderCircle, RefreshCw, Eye, X, CircleAlert, Save, Search, Download } from 'lucide-vue-next'
import api from '../api'

const props = defineProps({
  kbId: { type: String, default: '' },
  kbName: { type: String, default: '' },
})

const embedded = computed(() => !!props.kbId)
const kbNameLabel = computed(() => props.kbName || props.kbId)

const kbList = ref([])
const files = ref([])
const cases = ref([newCase()])
const running = ref(false)
const runError = ref('')
const result = ref(null)
const history = ref([])
const loadingHistory = ref(false)
const detail = ref(null)
const datasets = ref([])
const datasetSel = ref('')
const datasetHint = ref('')

const form = ref({
  kbId: '',
  topK: 4,
  name: `run-${new Date().toISOString().slice(0, 16).replace(/[-T:]/g, '')}`,
})

function newCase() {
  return {
    question: '',
    expected_file_id: '',
    expected_chunk_ids: [],
    reference_answer: '',
    expect_miss: false,
    candidates: null,
  }
}

const validCases = computed(() =>
  cases.value.length > 0 && cases.value.every((c) => c.question.trim() && c.expected_file_id),
)

const ragas = computed(() => (result.value?.metrics?.ragas) || {})
const ragasStatus = computed(() => ragas.value.status || 'disabled')

const ragasStatusLabel = computed(() => {
  const map = { completed: '已完成', partial: '部分完成', disabled: '未启用', unavailable: '不可用', failed: '失败' }
  return map[ragasStatus.value] || ragasStatus.value
})

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(4)
}

function metricLabel(key) {
  const map = {
    id_context_precision: 'ID 上下文精确率',
    id_context_recall: 'ID 上下文召回率',
    context_precision: '上下文精确率',
    context_recall: '上下文召回率',
    faithfulness: '忠实度',
    answer_relevancy: '答案相关性',
  }
  return map[key] || key
}

function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { hour12: false })
}

function fileName(id) {
  return files.value.find((f) => f.id === id)?.filename || id
}

function kbName(id) {
  if (embedded.value) return kbNameLabel.value
  return kbList.value.find((kb) => kb.id === id)?.name || id
}

function addCase() {
  cases.value.push(newCase())
}

function removeCase(idx) {
  cases.value.splice(idx, 1)
}

async function loadKbs() {
  kbList.value = await api.get('/knowledge/bases')
}

async function onKbChange() {
  await loadFiles()
  await loadDatasets()
  datasetSel.value = ''
}

async function loadFiles() {
  files.value = []
  result.value = null
  cases.value = [newCase()]
  if (!form.value.kbId) return
  files.value = await api.get(`/knowledge/bases/${form.value.kbId}/files`)
}

async function run() {
  if (!form.value.kbId || !validCases.value || running.value) return
  running.value = true
  runError.value = ''
  try {
    const payload = {
      name: form.value.name.trim() || `run-${Date.now()}`,
      kb_id: form.value.kbId,
      top_k: form.value.topK,
      cases: cases.value.map((c) => ({
        question: c.question.trim(),
        expected_file_id: c.expected_file_id,
        expected_chunk_ids: c.expected_chunk_ids || [],
        reference_answer: c.reference_answer.trim(),
        expect_miss: !!c.expect_miss,
      })),
    }
    const res = await api.post('/evaluation/runs', payload)
    result.value = res
    await loadHistory()
  } catch (err) {
    runError.value = err.response?.data?.detail
      ? (typeof err.response.data.detail === 'string' ? err.response.data.detail : JSON.stringify(err.response.data.detail))
      : (err.message || '评估失败')
  } finally {
    running.value = false
  }
}

async function loadHistory() {
  loadingHistory.value = true
  try {
    let runs = await api.get('/evaluation/runs')
    if (embedded.value) runs = runs.filter((r) => r.knowledge_base_id === props.kbId)
    history.value = runs
  } finally {
    loadingHistory.value = false
  }
}

function historyRagasStatus(r) {
  const m = r.ragas_status
  return m || '—'
}

async function viewRun(id) {
  detail.value = await api.get(`/evaluation/runs/${id}`)
}

async function loadDatasets() {
  if (!form.value.kbId) return
  try {
    const list = await api.get('/evaluation/datasets')
    datasets.value = list.filter((d) => d.knowledge_base_id === form.value.kbId)
  } catch {
    datasets.value = []
  }
}

async function onDatasetSelect() {
  if (!datasetSel.value) return
  try {
    const ds = await api.get(`/evaluation/datasets/${datasetSel.value}`)
    cases.value = (ds.cases || []).map((c) => ({
      question: c.question,
      expected_file_id: c.expected_file_id,
      expected_chunk_ids: c.expected_chunk_ids || [],
      reference_answer: c.reference_answer || '',
      expect_miss: !!c.expect_miss,
      candidates: null,
    }))
    datasetHint.value = `已加载评测集「${ds.name}」v${ds.version}，共 ${ds.case_count} 条用例`
    result.value = null
  } catch {
    datasetHint.value = '评测集加载失败'
  }
}

async function saveDataset() {
  if (!form.value.kbId || !validCases.value) return
  try {
    const name = form.value.name.trim() || 'golden-set'
    const res = await api.post('/evaluation/datasets', {
      name,
      kb_id: form.value.kbId,
      description: '',
      cases: cases.value.map((c) => ({
        question: c.question.trim(),
        expected_file_id: c.expected_file_id,
        expected_chunk_ids: c.expected_chunk_ids || [],
        reference_answer: c.reference_answer.trim(),
        expect_miss: !!c.expect_miss,
      })),
    })
    datasetHint.value = `评测集「${res.name}」已保存（v${res.version}，${res.case_count} 条）`
    await loadDatasets()
    datasetSel.value = res.id
  } catch (err) {
    datasetHint.value = err.message || '保存失败'
  }
}

async function loadCandidates(idx) {
  const c = cases.value[idx]
  if (!form.value.kbId || !c.question.trim() || !c.expected_file_id) return
  c.candidates = null
  try {
    c.candidates = await api.post('/evaluation/chunk-candidates', {
      kb_id: form.value.kbId,
      file_id: c.expected_file_id,
      question: c.question.trim(),
      top_k: 10,
    })
  } catch {
    c.candidates = []
  }
}

function closeCandidates(idx) {
  cases.value[idx].candidates = null
}

function referenceModeLabel(mode) {
  const map = {
    chunk_ids: 'chunk 标注',
    chunk: '单 chunk',
    file: '整文件兜底',
    negative: '负样本',
  }
  return map[mode] || mode || '-'
}

async function downloadReport(id) {
  try {
    const blob = await api.getBlob(`/evaluation/runs/${id}/report`)
    const url = URL.createObjectURL(blob.data)
    const a = document.createElement('a')
    a.href = url
    a.download = `eval-report-${id}.md`
    a.click()
    URL.revokeObjectURL(url)
  } catch (err) {
    datasetHint.value = err.message || '报告下载失败'
  }
}

onMounted(async () => {
  if (embedded.value) {
    form.value.kbId = props.kbId
    await loadFiles()
  } else {
    await loadKbs()
  }
  await loadDatasets()
  await loadHistory()
})
</script>

<style scoped>
.ev-shell {
  max-width: 1060px;
  margin: 0 auto;
  padding: 28px 32px 48px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.ev-shell.ev-embedded { max-width: none; padding: 0; }
.ev-header h1 { font-size: 22px; font-weight: 700; }
.ev-eyebrow { font-size: 11px; letter-spacing: 0.12em; color: var(--gray-500); font-weight: 600; }
.ev-header p { margin-top: 4px; color: var(--gray-600); font-size: 13px; }
.ev-card {
  background: var(--gray-0);
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
}
.ev-card-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px;
}
.ev-card-head h2 { font-size: 15px; font-weight: 700; }
.ev-hint { font-size: 12px; color: var(--gray-500); }
.ev-config-grid {
  display: grid; grid-template-columns: 220px 140px 1fr; gap: 12px;
}
.ev-field { display: flex; flex-direction: column; gap: 5px; font-size: 12px; color: var(--gray-600); }
.ev-field-wide { grid-column: span 1; }
.ev-field input, .ev-field select {
  padding: 8px 10px; border: 1px solid var(--gray-200); border-radius: var(--radius-md);
  font-size: 13px; background: var(--gray-0); color: var(--gray-900); outline: none;
}
.ev-field input:focus, .ev-field select:focus { border-color: var(--gray-800); }
.ev-case-list { display: flex; flex-direction: column; gap: 10px; }
.ev-case-row {
  display: flex; align-items: flex-end; gap: 10px;
  padding: 12px; border: 1px solid var(--gray-100); border-radius: var(--radius-md);
  background: var(--gray-25);
}
.ev-case-index {
  flex: none; width: 24px; height: 24px; border-radius: var(--radius-full);
  background: var(--gray-100); color: var(--gray-700);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 600; margin-bottom: 20px;
}
.ev-field-grow { flex: 1; min-width: 0; }
.ev-case-row .ev-field { min-width: 150px; }
.ev-case-row input { min-width: 120px; }
.ev-run-bar { display: flex; align-items: center; gap: 12px; margin-top: 16px; }
.ev-btn-primary {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--gray-900); color: var(--gray-0);
  border: none; border-radius: var(--radius-md);
  padding: 9px 18px; font-size: 13px; font-weight: 600;
}
.ev-btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.ev-btn-ghost {
  display: inline-flex; align-items: center; gap: 5px;
  background: transparent; border: 1px solid var(--gray-200); border-radius: var(--radius-md);
  padding: 6px 10px; font-size: 12px; color: var(--gray-700); cursor: pointer;
}
.ev-btn-ghost:hover { border-color: var(--gray-400); }
.ev-btn-danger { color: var(--gray-700); border-color: transparent; }
.ev-error { color: var(--color-error-700); font-size: 12px; }
.ev-empty {
  padding: 22px; text-align: center; color: var(--gray-500);
  font-size: 13px; border: 1px dashed var(--gray-200); border-radius: var(--radius-md);
}
.ev-metric-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;
}
.ev-metric-grid-sm { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
.ev-metric {
  border: 1px solid var(--gray-100); border-radius: var(--radius-md);
  padding: 12px 14px; background: var(--gray-25);
}
.ev-metric span { display: block; font-size: 11px; color: var(--gray-500); }
.ev-metric strong { font-size: 20px; font-weight: 700; }
.ev-metric small { display: block; font-size: 11px; color: var(--gray-400); margin-top: 2px; }
.ev-ragas {
  margin-top: 16px; padding: 14px; border: 1px solid var(--gray-150);
  border-radius: var(--radius-md); background: var(--gray-0);
}
.ev-ragas-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.ev-ragas-title { font-size: 13px; font-weight: 700; }
.ev-badge {
  font-size: 11px; padding: 2px 8px; border-radius: var(--radius-full);
  background: var(--gray-100); color: var(--gray-700);
}
.ev-badge.is-completed { background: var(--gray-900); color: var(--gray-0); }
.ev-badge.is-partial { background: var(--gray-200); }
.ev-badge.is-failed, .ev-badge.is-unavailable { background: var(--gray-100); color: var(--gray-600); }
.ev-ragas-error {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: var(--gray-600); padding: 8px 0;
}
.ev-subhead { font-size: 13px; font-weight: 700; margin: 18px 0 8px; }
.ev-table-wrap { overflow-x: auto; border: 1px solid var(--gray-100); border-radius: var(--radius-md); }
.ev-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.ev-table th, .ev-table td {
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--gray-50);
  white-space: nowrap;
}
.ev-table th { background: var(--gray-25); color: var(--gray-600); font-weight: 600; }
.ev-table tr:last-child td { border-bottom: none; }
.ev-cell-q { max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ev-cell-name { font-weight: 600; }
.ev-hit { color: var(--gray-900); font-weight: 600; }
.ev-miss { color: var(--gray-400); }
.ev-modal-overlay {
  position: fixed; inset: 0; background: rgba(0, 0, 0, 0.28);
  display: flex; align-items: center; justify-content: center; z-index: 100;
}
.ev-modal {
  width: min(760px, 92vw); max-height: 82vh; overflow: auto;
  background: var(--gray-0); border-radius: var(--radius-lg);
  padding: 20px 22px; box-shadow: var(--shadow-deep);
}
.ev-modal-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.ev-modal-table { margin-top: 14px; }
.ev-dataset-bar { display: flex; align-items: flex-end; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
.ev-check { flex-direction: row; align-items: center; gap: 6px; padding-bottom: 8px; }
.ev-check input { width: auto; }
.ev-head-actions { display: flex; align-items: center; gap: 10px; }
.ev-candidates {
  margin-top: 10px; padding: 10px 12px;
  border: 1px solid var(--gray-150); border-radius: var(--radius-md); background: var(--gray-25);
}
.ev-candidates-head {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 12px; color: var(--gray-600); margin-bottom: 8px;
}
.ev-candidates-list { display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow: auto; }
.ev-cand-item { display: flex; align-items: flex-start; gap: 8px; font-size: 12px; color: var(--gray-700); }
.ev-cand-score { flex: none; min-width: 44px; font-variant-numeric: tabular-nums; color: var(--gray-500); }
.ev-cand-snippet { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 520px; }
.spin { animation: ev-spin 0.9s linear infinite; }
@keyframes ev-spin { to { transform: rotate(360deg); } }

/* Claude-inspired warm editorial refresh */
.ev-shell { max-width: 1180px; padding: 34px 34px 56px; gap: 16px; }
.ev-header { padding: 4px 2px 10px; }
.ev-header h1 { margin-top: 5px; color: var(--gray-950); font-family: var(--font-display); font-size: 32px; font-weight: var(--font-display-weight); letter-spacing: -.025em; }
.ev-eyebrow { color: var(--main-600); font-size: 10px; font-weight: 750; letter-spacing: .16em; }
.ev-header p { max-width: 760px; color: var(--gray-500); line-height: 1.7; }
.ev-card { border-color: var(--gray-150); border-radius: 16px; padding: 20px 22px; box-shadow: 0 1px 2px rgba(58,48,40,.025); }
.ev-card-head h2 { color: var(--gray-950); font-family: var(--font-display); font-size: 17px; font-weight: var(--font-display-weight); letter-spacing: -.01em; }
.ev-field > span { color: var(--gray-700); font-weight: 650; }
.ev-field input,
.ev-field select { min-height: 40px; border-color: var(--gray-150); border-radius: 9px; background: var(--gray-10); }
.ev-field input:hover,
.ev-field select:hover { border-color: var(--gray-300); }
.ev-field input:focus,
.ev-field select:focus { border-color: var(--main-400); background: #fff; box-shadow: 0 0 0 3px var(--main-50); }
.ev-case-row { border-color: var(--gray-150); border-radius: 12px; background: var(--gray-10); }
.ev-case-index { background: var(--main-50); color: var(--main-700); }
.ev-btn-primary { min-height: 38px; padding-inline: 17px; border-radius: 9px; background: var(--gray-1000); box-shadow: 0 3px 10px rgba(20,20,19,.12); }
.ev-btn-primary:hover:not(:disabled) { background: var(--gray-800); }
.ev-btn-ghost { min-height: 34px; border-color: var(--gray-150); border-radius: 9px; background: #fff; }
.ev-btn-ghost:hover { border-color: var(--main-200); background: var(--main-30); color: var(--main-700); }
.ev-metric { border-color: var(--gray-150); border-radius: 11px; background: linear-gradient(145deg, #fff, var(--gray-25)); }
.ev-metric strong { color: var(--gray-950); font-size: 22px; letter-spacing: -.03em; }
.ev-badge.is-completed { background: var(--gray-1000); }
.ev-table-wrap { border-color: var(--gray-150); border-radius: 11px; }
.ev-table th { background: var(--gray-50); color: var(--gray-600); }
.ev-table tbody tr:hover { background: var(--main-20); }
.ev-modal { border: 1px solid var(--gray-150); border-radius: 18px; }

@media (max-width: 980px) {
  .ev-config-grid { grid-template-columns: minmax(180px, 1fr) 130px; }
  .ev-field-wide { grid-column: 1 / -1; }
  .ev-case-row { display: grid; grid-template-columns: 28px minmax(180px, 1fr) minmax(160px, .7fr); align-items: end; }
  .ev-case-index { grid-row: 1 / 3; align-self: start; margin: 23px 0 0; }
}

@media (max-width: 680px) {
  .ev-shell { padding: 24px 14px 42px; }
  .ev-header h1 { font-size: 26px; }
  .ev-card { padding: 16px; }
  .ev-card-head { align-items: flex-start; gap: 10px; }
  .ev-card-head .ev-hint { display: none; }
  .ev-config-grid,
  .ev-case-row { grid-template-columns: 1fr; }
  .ev-field-wide { grid-column: auto; }
  .ev-case-index { grid-row: auto; margin: 0; }
  .ev-case-row .ev-field { min-width: 0; }
  .ev-dataset-bar { align-items: stretch; flex-direction: column; }
  .ev-dataset-bar .ev-btn-ghost { justify-content: center; }
}
</style>
