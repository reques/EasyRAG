<template>
  <section class="ob-waterfall" aria-label="步骤详情">
    <div class="ob-section-head">
      <div>
        <h2>
          执行步骤
          <span>Observations</span>
        </h2>
        <p>{{ remoteStateLabel(remote?.state) }}</p>
      </div>
      <div class="ob-segment">
        <button
          :class="{ active: source === 'remote' }"
          :disabled="!remote?.observations?.length"
          @click="chosenSource = 'remote'"
        >
          Langfuse {{ remote?.observations?.length || 0 }}
        </button>
        <button
          :class="{ active: source === 'local' }"
          :disabled="!events.length"
          @click="chosenSource = 'local'"
        >
          本地事件 {{ events.length }}
        </button>
      </div>
    </div>
    <p v-if="remote?.truncated" class="ob-notice">仅显示前 500 个远端步骤。</p>
    <div v-if="rows.length" class="ob-waterfall-layout">
      <div class="ob-step-list">
        <div class="ob-step-axis">
          <span>{{ source === 'remote' ? '步骤 / 层级' : '事件 / 顺序' }}</span>
          <span>{{ source === 'remote' ? '耗时分布' : '事件时间' }}</span>
        </div>
        <button
          v-for="row in rows"
          :key="row.id"
          class="ob-step-row"
          :class="{
            selected: selected?.id === row.id,
            failed: ['error', 'failed', 'timed_out'].includes(row.status),
          }"
          :style="{ '--step-depth': row.depth }"
          @click="selectedId = row.id"
        >
          <span class="ob-step-name">
            <span class="ob-step-dot" :class="row.type?.toLowerCase()"></span>
            <span>
              <strong>{{ row.name }}</strong>
              <small>{{ row.type }} · {{ statusLabel(row.status) }}</small>
            </span>
          </span>
          <span class="ob-step-timing">
            <span
              v-if="row.offset != null && source === 'remote'"
              class="ob-step-track"
            >
              <i
                :style="{
                  left: `${row.offset}%`,
                  width: `${Math.min(row.width || 0.8, 100 - row.offset)}%`,
                }"
              ></i>
            </span>
            <small>
              {{
                source === 'remote'
                  ? formatMs(row.duration)
                  : formatTime(row.started_at).split(' ')[1] || '—'
              }}
            </small>
          </span>
        </button>
      </div>
      <article v-if="selected" class="ob-step-details">
        <header>
          <span class="ob-kicker">
            {{ source === 'remote' ? 'OBSERVATION' : 'LOCAL EVENT' }}
          </span>
          <h3>{{ selected.name }}</h3>
          <code>{{ selected.id }}</code>
        </header>
        <dl>
          <div>
            <dt>状态</dt>
            <dd>{{ statusLabel(selected.status) }}</dd>
          </div>
          <div>
            <dt>耗时</dt>
            <dd>{{ formatMs(selected.duration) }}</dd>
          </div>
          <div v-if="selected.model">
            <dt>模型</dt>
            <dd>{{ selected.model }}</dd>
          </div>
          <div v-if="selected.token_usage?.input != null">
            <dt>输入 / 输出 Token</dt>
            <dd>
              {{ formatNumber(selected.token_usage.input) }} /
              {{ formatNumber(selected.token_usage.output) }}
            </dd>
          </div>
          <div v-if="selected.cost_usd != null">
            <dt>Langfuse 费用估算</dt>
            <dd>${{ selected.cost_usd.toFixed(6) }}</dd>
          </div>
        </dl>
        <p v-if="selected.error" class="ob-error">{{ selected.error }}</p>
        <div
          v-for="field in ['input', 'output', 'metadata']"
          :key="field"
          class="ob-io"
        >
          <h4>
            {{ { input: '输入', output: '输出', metadata: '步骤信息' }[field] }}
          </h4>
          <pre>{{ pretty(selected[field]) }}</pre>
        </div>
      </article>
    </div>
    <div v-else class="ob-empty compact">
      <Activity :size="25" />
      <strong>
        {{
          remote?.state === 'loading' ? '正在读取执行步骤' : '没有可显示的步骤'
        }}
      </strong>
      <p>本地事件和远端观测会按各自的数据来源展示。</p>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { Activity } from 'lucide-vue-next'
import {
  formatMs,
  formatNumber,
  formatTime,
  observationRows,
  pretty,
  remoteStateLabel,
  statusLabel,
} from '../utils/observability.js'
const props = defineProps({
  remote: { type: Object, default: null },
  events: { type: Array, default: () => [] },
})
const chosenSource = ref(''),
  selectedId = ref('')
const source = computed(
  () =>
    chosenSource.value ||
    (props.remote?.observations?.length ? 'remote' : 'local'),
)
const local = computed(() =>
  props.events.map((e) => ({
    id: e.id,
    parent_id: e.parent_id,
    name: e.metadata?.title || e.metadata?.tool || e.type,
    type: e.type,
    status: e.status,
    started_at: e.timestamp,
    input: e.input,
    output: e.output,
    metadata: e.metadata,
  })),
)
const rows = computed(() =>
  observationRows(
    source.value === 'remote' ? props.remote?.observations || [] : local.value,
  ),
)
const selected = computed(
  () => rows.value.find((r) => r.id === selectedId.value) || rows.value[0],
)
watch(
  () => props.remote?.trace_id,
  () => {
    chosenSource.value = ''
    selectedId.value = ''
  },
)
</script>
