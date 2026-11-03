<template>
  <div v-if="value" class="evaluation-observability" role="status">
    <span v-if="value.linked_items">
      已关联 {{ value.linked_items }}/{{ value.total_items }} 条实验样例
    </span>
    <span v-else>
      {{
        value.state === 'disabled'
          ? '本次评测未启用 Langfuse 观测'
          : '本次评测未记录到 Langfuse 实验观测'
      }}
    </span>
    <router-link
      v-if="showNativeLink && nativeRunId"
      :to="{
        path: '/observability',
        query: { tab: 'evaluations', run: nativeRunId },
      }"
    >
      查看逐题观测与评分 →
    </router-link>
    <a v-if="safeUrl" :href="safeUrl" target="_blank" rel="noopener noreferrer">
      在 Langfuse 比较实验 ↗
    </a>
    <small v-if="value.linked_items">远端接收与索引状态未确认</small>
  </div>
</template>

<script setup>
import { computed } from 'vue'
const props = defineProps({
  value: { type: Object, default: null },
  showNativeLink: { type: Boolean, default: true },
})
const nativeRunId = computed(() =>
  /^[a-f0-9-]{36}$/.test(props.value?.experiment_id || '')
    ? props.value.experiment_id
    : null,
)
const safeUrl = computed(() => {
  try {
    const url = new URL(props.value?.url)
    return ['http:', 'https:'].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : null
  } catch {
    return null
  }
})
</script>

<style scoped>
.evaluation-observability {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin: 12px 0;
  padding: 10px 12px;
  border: 1px solid var(--gray-150, #e4e4e4);
  border-radius: 8px;
  font-size: 12px;
}
.evaluation-observability a {
  color: var(--text-primary, #333);
  text-decoration: underline;
  text-underline-offset: 3px;
}
.evaluation-observability small {
  color: var(--text-secondary, #777);
}
</style>
