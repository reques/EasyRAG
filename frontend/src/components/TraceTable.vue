<template>
  <div v-if="items.length" class="ob-table-wrap">
    <table class="ob-table">
      <thead>
        <tr>
          <th>问题 / 执行</th>
          <th>智能体 / 模型</th>
          <th>状态</th>
          <th>耗时</th>
          <th>Token</th>
          <th>时间</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="item in items" :key="item.id">
          <td>
            <button class="ob-trace-link" @click="$emit('open', item.id)">
              <strong>
                {{ item.question || item.conversation_title || '未记录问题' }}
              </strong>
              <code>{{ item.id }}</code>
            </button>
          </td>
          <td>
            <span>{{ item.agent_name }}</span>
            <small>{{ modelName(item.model_id) }}</small>
          </td>
          <td>
            <span class="ob-status" :class="item.status">
              <i></i>
              {{ statusLabel(item.status) }}
            </span>
          </td>
          <td>{{ formatMs(item.duration_ms) }}</td>
          <td>{{ formatNumber(item.token_usage?.total_tokens) }}</td>
          <td class="ob-muted">{{ formatTime(item.started_at) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div v-else class="ob-empty">
    <Workflow :size="29" />
    <strong>没有匹配的执行记录</strong>
    <p>发送一条消息后，可以在这里查看执行过程；也可以调整筛选条件。</p>
  </div>
</template>
<script setup>
import { Workflow } from 'lucide-vue-next'
import {
  formatMs,
  formatNumber,
  formatTime,
  statusLabel,
} from '../utils/observability.js'
defineProps({
  items: { type: Array, default: () => [] },
  modelName: { type: Function, default: (id) => id || '未记录模型' },
})
defineEmits(['open'])
</script>
