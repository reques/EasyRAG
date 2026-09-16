<template>
  <Teleport to="body">
    <div class="artifact-preview-overlay" @click.self="$emit('close')">
      <section ref="dialog" class="artifact-preview" role="dialog" aria-modal="true" aria-labelledby="artifact-preview-title" tabindex="-1" @keydown="onKeydown">
        <header>
          <div><h2 id="artifact-preview-title">{{ artifact.filename }}</h2><small>{{ format }} · 预览</small></div>
          <button type="button" @click="$emit('download', artifact)">下载</button>
          <button type="button" aria-label="关闭预览" @click="$emit('close')">×</button>
        </header>
        <div v-if="loading" class="preview-placeholder" role="status">正在加载预览…</div>
        <div v-else-if="error" class="preview-placeholder" role="alert">{{ error }}<button @click="loadPreview">重试</button></div>
        <template v-else>
          <p v-if="preview.note" class="preview-note">{{ preview.note }}</p>
          <iframe v-if="preview.kind === 'html'" title="文档内容预览" sandbox="" :srcdoc="htmlDocument"></iframe>
          <iframe v-else-if="preview.kind === 'pdf'" title="PDF 页面预览" :src="pdfUrl"></iframe>
          <template v-else-if="preview.kind === 'sheets'">
            <nav class="sheet-tabs" aria-label="工作表">
              <button v-for="(sheet, i) in preview.sheets" :key="sheet.name" :aria-pressed="sheetIndex === i" @click="sheetIndex = i">{{ sheet.name }}</button>
            </nav>
            <p v-if="activeSheet?.truncated" class="preview-note">预览最多显示每个工作表的前 200 行、50 列，完整数据请下载查看。</p>
            <div class="sheet-scroll">
              <table v-if="activeSheet"><tbody>
                <tr v-for="(row, i) in activeSheet.rows" :key="i"><th scope="row">{{ i + 1 }}</th><td v-for="(cell, j) in row" :key="j">{{ cell ?? '' }}</td></tr>
              </tbody></table>
            </div>
          </template>
          <pre v-else class="preview-text">{{ preview.content }}</pre>
        </template>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import api from '../api'
import { artifactRoute, previewDocument } from '../utils/artifact-preview.js'

const props = defineProps({ artifact: { type: Object, required: true } })
const emit = defineEmits(['close', 'download'])
const dialog = ref(null)
const loading = ref(true)
const error = ref('')
const preview = ref({})
const sheetIndex = ref(0)
const pdfUrl = ref('')
const format = computed(() => props.artifact.filename?.split('.').pop()?.toUpperCase())
const activeSheet = computed(() => preview.value.sheets?.[sheetIndex.value])
const htmlDocument = computed(() => preview.value.kind === 'html' ? previewDocument(preview.value.content || '') : '')
let request = 0
let returnFocus
function clearPdf() {
  if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value)
  pdfUrl.value = ''
}
async function loadPreview() {
  const current = ++request
  loading.value = true
  error.value = ''
  preview.value = {}
  sheetIndex.value = 0
  clearPdf()
  try {
    const path = artifactRoute(props.artifact) + '/preview'
    if (format.value === 'PDF') {
      const { data } = await api.getBlob(path)
      if (current !== request) return
      pdfUrl.value = URL.createObjectURL(data)
      preview.value = { kind: 'pdf' }
    } else {
      const result = await api.get(path)
      if (current !== request) return
      preview.value = result
    }
  } catch {
    if (current === request) error.value = '预览加载失败，可重试或下载文件查看。'
  } finally {
    if (current === request) loading.value = false
  }
}
function onKeydown(event) {
  if (event.key === 'Escape') { event.stopPropagation(); emit('close') }
  if (event.key !== 'Tab') return
  const targets = [...dialog.value.querySelectorAll('button:not(:disabled), iframe')]
  const first = targets[0], last = targets.at(-1)
  if (event.shiftKey && [first, dialog.value].includes(document.activeElement)) { event.preventDefault(); last?.focus() }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
}
watch(() => props.artifact, loadPreview, { immediate: true })
onMounted(async () => { returnFocus = document.activeElement; await nextTick(); dialog.value?.focus() })
onUnmounted(() => { request++; clearPdf(); returnFocus?.focus() })
</script>

<style scoped>
.artifact-preview-overlay{position:fixed;inset:0;z-index:2000;display:flex;align-items:center;justify-content:center;padding:24px;background:rgba(20,25,30,.4)}
.artifact-preview{display:flex;flex-direction:column;width:min(1100px,100%);height:90vh;max-height:100%;background:#fff;color:#262626;border-radius:16px;box-shadow:0 24px 80px #0003;overflow:hidden;outline:none}
header{display:flex;align-items:center;gap:12px;padding:16px 22px;border-bottom:1px solid #e8e8e8}header>div{flex:1;min-width:0}h2{font-size:16px;margin:0;overflow-wrap:anywhere}small{color:#888}
button{border:1px solid #ddd;background:#fff;color:#333;border-radius:8px;padding:6px 12px;cursor:pointer}button:focus-visible{outline:2px solid #3597ae;outline-offset:2px}
iframe{flex:1;width:100%;border:0;min-height:0;background:white}.preview-placeholder{flex:1;display:flex;gap:12px;align-items:center;justify-content:center;color:#777}
.preview-note{margin:0;padding:8px 20px;background:#f7f7f7;color:#777;font-size:12px}.preview-text{margin:0;padding:24px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;flex:1}
.sheet-tabs{display:flex;gap:8px;padding:12px 20px;overflow-x:auto;border-bottom:1px solid #eee;flex-shrink:0}.sheet-tabs button{white-space:nowrap}.sheet-tabs button[aria-pressed=true]{background:#e8f5f7;border-color:#5aa4b4;color:#216779}
.sheet-scroll{flex:1;overflow:auto}table{border-collapse:separate;border-spacing:0;min-width:100%;font-size:13px}td,th{border-right:1px solid #e4e7e9;border-bottom:1px solid #e4e7e9;padding:8px 12px;white-space:pre-wrap;min-width:100px;max-width:350px;overflow-wrap:anywhere}th{position:sticky;left:0;min-width:40px;background:#f4f6f7;color:#777;font-weight:normal}tr:first-child td{background:#edf5f6;font-weight:600}
@media(max-width:760px){.artifact-preview-overlay{padding:0}.artifact-preview{width:100%;height:100dvh;border-radius:0}header{padding:12px;gap:8px}}
</style>
