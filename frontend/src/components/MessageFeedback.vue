<template>
  <div class="answer-feedback">
    <div class="feedback-actions" aria-label="评价这条回答">
      <button type="button" :aria-pressed="rating === 'like'" :disabled="saving" title="回答有帮助" @click="submit('like')">
        <ThumbsUp :size="13" /> 有帮助
      </button>
      <button type="button" :aria-pressed="rating === 'dislike'" :disabled="saving" title="回答需要改进" @click="openReason">
        <ThumbsDown :size="13" /> 需改进
      </button>
      <span v-if="saved && !editing" class="feedback-saved" role="status">已记录</span>
    </div>
    <form v-if="editing" class="feedback-form" @submit.prevent="submit('dislike')">
      <label :for="`feedback-${messageId}`">哪里需要改进？<span>（选填）</span></label>
      <textarea :id="`feedback-${messageId}`" v-model="reason" maxlength="2000" rows="2" placeholder="例如：引用不准确、没有回答问题…" :disabled="saving" />
      <div class="feedback-actions">
        <button type="submit" :disabled="saving">{{ saving ? '保存中…' : '提交反馈' }}</button>
        <button type="button" :disabled="saving" @click="editing = false">取消</button>
      </div>
    </form>
    <span v-if="error" class="feedback-error" role="alert">{{ error }}</span>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { ThumbsUp, ThumbsDown } from 'lucide-vue-next'
import api from '../api'

const props = defineProps({ messageId: { type: Number, required: true } })
const emit = defineEmits(['saved'])
const rating = ref(null)
const reason = ref('')
const saving = ref(false)
const saved = ref(false)
const editing = ref(false)
const error = ref('')
let requestVersion = 0
watch(() => props.messageId, async id => {
  const version = ++requestVersion
  rating.value = null
  reason.value = ''
  saved.value = editing.value = false
  error.value = ''
  try {
    const result = await api.get(`/chat/messages/${id}/feedback`)
    if (version !== requestVersion) return
    rating.value = result.rating
    reason.value = result.reason || ''
    saved.value = !!result.rating
  } catch { /* Feedback can still be submitted when an initial read fails. */ }
}, { immediate: true })

function openReason() { editing.value = true; error.value = '' }
async function submit(value) {
  if (saving.value) return
  saving.value = true
  error.value = ''
  const version = ++requestVersion
  try {
    const result = await api.put(`/chat/messages/${props.messageId}/feedback`, { rating: value, reason: value === 'dislike' ? reason.value : '' })
    if (version !== requestVersion) return
    rating.value = result.rating
    reason.value = result.reason || ''
    saved.value = true
    editing.value = false
    emit('saved')
  } catch {
    if (version === requestVersion) error.value = '反馈保存失败，请重试'
  } finally { saving.value = false }
}
</script>

<style scoped>
.answer-feedback { display: inline-flex; flex-direction: column; gap: 7px; align-items: flex-start; font-size: 12px; }
.feedback-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.feedback-actions button { display: inline-flex; align-items: center; gap: 5px; border: 0; border-radius: 5px; padding: 4px 7px; background: transparent; color: var(--text-secondary, #777); font: inherit; cursor: pointer; }
.feedback-actions button:hover, .feedback-actions button[aria-pressed="true"] { background: var(--bg-secondary, #f2f2f2); color: var(--text-primary, #222); }
.feedback-actions button:focus-visible { outline: 2px solid var(--accent, #52796f); outline-offset: 2px; }
.feedback-actions button:disabled { opacity: .5; cursor: wait; }
.feedback-saved, .feedback-form label span { color: var(--text-secondary, #777); }
.feedback-form { display: grid; gap: 7px; width: min(360px, 70vw); padding: 10px; border: 1px solid var(--gray-150, #e4e4e4); border-radius: 8px; }
.feedback-form textarea { width: 100%; box-sizing: border-box; resize: vertical; padding: 7px; border: 1px solid var(--gray-150, #e4e4e4); border-radius: 5px; color: inherit; background: transparent; font: inherit; }
.feedback-error { color: var(--danger, #b23a3a); }
</style>
