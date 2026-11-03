<template>
  <section class="ob-detail">
    <div class="ob-detail-toolbar">
      <button class="ob-button" @click="$emit('back')">
        <ArrowLeft :size="15" />
        返回记录
      </button>
      <button class="ob-button" :disabled="loading || judging" @click="load">
        <RefreshCw :size="15" :class="{ spin: loading }" />
        刷新
      </button>
    </div>
    <p v-if="error" class="ob-error" role="alert">{{ error }}</p>
    <div v-if="loading && !detail" class="ob-empty">
      <LoaderCircle :size="25" class="spin" />
      <strong>正在读取执行记录</strong>
    </div>
    <template v-if="detail">
      <header class="ob-detail-header">
        <div>
          <span class="ob-kicker">TRACE / {{ detail.id }}</span>
          <h1>
            {{ detail.question || detail.conversation_title || '执行详情' }}
          </h1>
          <p>
            {{ detail.agent_name }} · {{ modelName(detail.model_id) }} ·
            {{ formatTime(detail.started_at) }}
          </p>
        </div>
        <router-link
          class="ob-button"
          :to="{ path: '/', query: { session: detail.conversation_id } }"
        >
          返回原会话
          <ArrowUpRight :size="15" />
        </router-link>
      </header>
      <div class="ob-detail-metrics">
        <span class="ob-status" :class="detail.status">
          <i></i>
          {{ statusLabel(detail.status) }}
        </span>
        <span>
          总耗时
          <b>{{ formatMs(detail.duration_ms) }}</b>
        </span>
        <span>
          输入
          <b>{{ formatNumber(detail.token_usage?.input_tokens) }}</b>
        </span>
        <span>
          输出
          <b>{{ formatNumber(detail.token_usage?.output_tokens) }}</b>
        </span>
        <span>
          Token 合计
          <b>{{ formatNumber(detail.token_usage?.total_tokens) }}</b>
        </span>
      </div>
      <article class="ob-card ob-answer">
        <div class="ob-section-head">
          <h2>最终回答</h2>
          <span class="ob-muted">内容已脱敏 · 长内容可返回会话查看</span>
        </div>
        <pre>{{ detail.answer || '此执行没有保存最终回答。' }}</pre>
        <MessageFeedback
          v-if="detail.answer_message_id"
          :message-id="detail.answer_message_id"
          @saved="refreshLocal"
        />
      </article>
      <ObservationWaterfall :remote="remote" :events="detail.events" />
      <p v-if="detail.events_truncated" class="ob-notice">
        本地事件仅显示前 500 条。
      </p>
      <div class="ob-bottom-grid">
        <article class="ob-card">
          <div class="ob-section-head">
            <div>
              <h2>
                回答评分
                <span>AI Judge</span>
              </h2>
              <p>手动调用所选模型，会产生模型用量。评分仅供质量参考。</p>
            </div>
            <Sparkles :size="19" class="ob-accent" />
          </div>
          <div class="ob-judge-controls">
            <label class="ob-sr-only" for="ob-judge-model">评分模型</label>
            <select
              id="ob-judge-model"
              v-model="judgeModel"
              :disabled="judging"
            >
              <option value="" disabled>选择评分模型</option>
              <option
                v-for="m in models.filter((m) => m.available)"
                :key="m.id"
                :value="m.id"
              >
                {{ m.name }}
              </option>
            </select>
            <button
              class="ob-button primary"
              :disabled="
                judging ||
                !judgeModel ||
                detail.status !== 'completed' ||
                !detail.answer ||
                detail.judgement?.state === 'running'
              "
              @click="judge"
            >
              <LoaderCircle v-if="judging" :size="15" class="spin" />
              <Sparkles v-else :size="15" />
              {{
                judging
                  ? '评分中…'
                  : detail.judgement?.result
                    ? '重新评分'
                    : '评测此回答'
              }}
            </button>
          </div>
          <p v-if="judgeError" class="ob-error" role="alert">
            {{ judgeError }}
          </p>
          <p v-if="detail.judgement?.state === 'running'" class="ob-notice">
            此回答正在评分，可稍后刷新。重复点击不会创建并发评分。
          </p>
          <p
            v-if="['failed', 'stale'].includes(detail.judgement?.state)"
            class="ob-notice"
          >
            上次评分{{
              detail.judgement.state === 'stale' ? '执行已过期' : '失败'
            }}，可稍后重试。
          </p>
          <template v-if="detail.judgement?.result">
            <p v-if="detail.judgement.state !== 'completed'" class="ob-muted">
              下方保留最近一次成功评分的结果。
            </p>
            <div class="ob-score-values">
              <span>
                切题程度
                <b>
                  {{ formatMetric(detail.judgement.result.answer_relevance) }}
                </b>
              </span>
              <span>
                证据支持
                <b>
                  {{ formatMetric(detail.judgement.result.evidence_support) }}
                </b>
              </span>
            </div>
            <p class="ob-judge-explanation">
              {{ detail.judgement.result.explanation }}
            </p>
            <p class="ob-muted">
              {{ detail.judgement.result.model_name }} ·
              {{ detail.judgement.result.rubric_version }} ·
              {{
                formatTime(
                  detail.judgement.result.judged_at ||
                    (detail.judgement.state === 'completed'
                      ? detail.judgement.finished_at
                      : null),
                )
              }}
            </p>
            <p
              v-if="!detail.judgement.result.evidence_available"
              class="ob-muted"
            >
              未记录到工具证据，证据支持度未评分。
            </p>
          </template>
        </article>
        <article class="ob-card">
          <div class="ob-section-head">
            <div>
              <h2>
                反馈与评分
                <span>Scores</span>
              </h2>
              <p>本地结果与 Langfuse 投递状态。</p>
            </div>
            <ListChecks :size="19" class="ob-accent" />
          </div>
          <div v-if="detail.scores.length" class="ob-score-list">
            <div v-for="score in detail.scores" :key="score.id">
              <span>
                <strong>{{ score.name }}</strong>
                <small>{{ score.comment || '无附加说明' }}</small>
              </span>
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
          <div v-else class="ob-empty compact">
            <p>尚无关联评分。用户反馈或 AI 评分后会显示在这里。</p>
          </div>
          <p class="ob-muted">
            “已接收”表示 API 已确认接收；步骤索引状态见上方观测区。
          </p>
        </article>
      </div>
    </template>
  </section>
</template>
<script setup>
import { onBeforeUnmount, ref, watch } from 'vue'
import {
  ArrowLeft,
  ArrowUpRight,
  ListChecks,
  LoaderCircle,
  RefreshCw,
  Sparkles,
} from 'lucide-vue-next'
import api from '../api'
import MessageFeedback from './MessageFeedback.vue'
import ObservationWaterfall from './ObservationWaterfall.vue'
import {
  formatMetric,
  formatMs,
  formatNumber,
  formatTime,
  statusLabel,
} from '../utils/observability.js'
const props = defineProps({
  traceId: { type: String, required: true },
  models: { type: Array, default: () => [] },
})
defineEmits(['back'])
const detail = ref(null),
  remote = ref(null),
  loading = ref(false),
  error = ref(''),
  judgeModel = ref(''),
  judging = ref(false),
  judgeError = ref('')
let generation = 0
const modelName = (id) =>
  props.models.find((m) => m.id === id)?.name || id || '未记录模型'
function message(err, fallback) {
  return typeof err.response?.data?.detail === 'string'
    ? err.response.data.detail
    : fallback
}
async function load() {
  const id = props.traceId,
    current = ++generation
  loading.value = true
  error.value = ''
  try {
    const value = await api.get(
      `/observability/traces/${encodeURIComponent(id)}`,
    )
    if (current !== generation) return
    detail.value = value
    if (!judgeModel.value)
      judgeModel.value =
        props.models.find((m) => m.id === value.model_id && m.available)?.id ||
        props.models.find((m) => m.is_default && m.available)?.id ||
        ''
    remote.value = { state: 'loading', trace_id: value.observability_trace_id }
    api
      .get(`/observability/traces/${encodeURIComponent(id)}/observations`)
      .then((value) => {
        if (current === generation) remote.value = value
      })
      .catch(() => {
        if (current === generation)
          remote.value = { state: 'unavailable', observations: [] }
      })
  } catch (err) {
    if (current === generation)
      error.value = message(err, '执行记录读取失败，请重试')
  } finally {
    if (current === generation) loading.value = false
  }
}
async function refreshLocal() {
  const current = generation
  try {
    const value = await api.get(
      `/observability/traces/${encodeURIComponent(props.traceId)}`,
    )
    if (current === generation) detail.value = value
  } catch {}
}
async function judge() {
  if (judging.value) return
  const id = props.traceId,
    current = generation
  judging.value = true
  judgeError.value = ''
  try {
    const value = await api.post(
      `/observability/traces/${encodeURIComponent(id)}/judge`,
      { model_id: judgeModel.value, request_id: crypto.randomUUID() },
    )
    if (current !== generation) return
    detail.value.judgement = value
    await refreshLocal()
  } catch (err) {
    if (current === generation) {
      judgeError.value = message(err, '评分失败，请稍后重试')
      await refreshLocal()
    }
  } finally {
    if (current === generation) judging.value = false
  }
}
watch(
  () => props.traceId,
  () => {
    detail.value = null
    judgeModel.value = ''
    judging.value = false
    judgeError.value = ''
    load()
  },
  { immediate: true },
)
watch(
  () => props.models,
  () => {
    if (!judgeModel.value && detail.value)
      judgeModel.value =
        props.models.find((m) => m.id === detail.value.model_id && m.available)
          ?.id || ''
  },
)
onBeforeUnmount(() => {
  generation++
})
</script>
