<template>
  <div class="knowledge-view">
    <header class="chat-header">
      <h2><LibraryBig :size="16" /> 知识库管理</h2>
      <div class="header-actions">
        <button @click="showCreate = true" class="btn-primary-sm">
          <Plus :size="14" /> 新建知识库
        </button>
      </div>
    </header>

    <!-- 创建知识库弹窗 -->
    <div v-if="showCreate" class="modal-overlay" @click.self="showCreate = false">
      <div class="modal">
        <h3>新建知识库</h3>
        <label>
          <span>名称</span>
          <input v-model="newKb.name" placeholder="我的知识库" />
        </label>
        <label>
          <span>描述（选填）</span>
          <input v-model="newKb.description" placeholder="简要描述…" />
        </label>
        <p v-if="createError" class="auth-error">{{ createError }}</p>
        <div class="modal-actions">
          <button @click="showCreate = false" class="btn-secondary">取消</button>
          <button @click="createKb" :disabled="creating" class="btn-primary-sm">
            {{ creating ? '创建中…' : '创建' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 文件上传弹窗 -->
    <div v-if="showUpload" class="modal-overlay" @click.self="closeUpload">
      <div class="modal">
        <h3>上传文档到「{{ activeKb?.name }}」</h3>
        <label class="file-label">
          <input type="file" @change="onFileSelect" :disabled="uploading" accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg,.bmp,.webp" />
          <span v-if="!uploadFile">点击选择文件 (.txt .md .pdf .docx 图片)</span>
          <span v-else>📄 {{ uploadFile.name }}<span class="file-size-hint">（{{ formatSize(uploadFile.size) }}）</span></span>
        </label>

        <!-- 进度条：传输阶段 + 索引阶段 -->
        <div v-if="uploading || uploadPhase !== 'idle'" class="upload-progress">
          <div class="upload-progress-label">
            <span>{{ progressLabel }}</span>
            <span>{{ displayProgress }}%</span>
          </div>
          <div class="progress-track">
            <div
              class="progress-fill"
              :class="{ indeterminate: uploadPhase === 'indexing' && indexProgress === 0, failed: uploadPhase === 'failed' }"
              :style="{ width: displayProgress + '%' }"
            ></div>
          </div>
        </div>

        <p v-if="uploadMsg" :class="uploadOk ? 'auth-success' : 'auth-error'">{{ uploadMsg }}</p>
        <div class="modal-actions">
          <button @click="closeUpload" :disabled="uploading && uploadPhase === 'transferring'" class="btn-secondary">
            {{ uploading ? '后台继续' : '关闭' }}
          </button>
          <button @click="doUpload" :disabled="!uploadFile || uploading" class="btn-primary-sm">
            {{ uploading ? '处理中…' : '上传并索引' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 文件预览弹窗 -->
    <div v-if="showPreview" class="modal-overlay" @click.self="closePreview">
      <div class="modal preview-modal">
        <div class="preview-header">
          <h3>
            <FileText :size="16" />
            {{ previewFile?.filename }}
          </h3>
          <button @click="closePreview" class="btn-secondary">&times;</button>
        </div>
        <div v-if="previewLoading" class="preview-loading">加载中…</div>
        <div v-else-if="previewError" class="preview-error">{{ previewError }}</div>
        <!-- PDF / 图片：浏览器原生渲染 -->
        <div v-else-if="previewContentType === 'binary'" class="preview-binary-wrap">
          <iframe v-if="previewFile?.file_type === 'pdf'" :src="rawUrl" class="preview-frame" />
          <img v-else :src="rawUrl" :alt="previewFile?.filename" class="preview-image" />
        </div>
        <!-- 文本预览 (txt/md/docx) -->
        <div v-else class="preview-text-wrap">
          <pre class="preview-text">{{ previewText }}</pre>
        </div>
      </div>
    </div>

    <!-- 知识库列表 -->
    <div v-if="loading" class="chat-empty"><p>加载中…</p></div>

    <div v-else-if="kbList.length === 0" class="chat-empty">
      <h3>暂无知识库</h3>
      <p>点击上方按钮创建你的第一个知识库</p>
    </div>

    <div v-else class="kb-grid">
      <div
        v-for="kb in kbList"
        :key="kb.id"
        class="kb-card"
        :class="{ active: activeKb?.id === kb.id }"
        @click="selectKb(kb)"
      >
        <div class="kb-card-header">
          <span class="kb-icon"><FolderOpen :size="17" /></span>
          <strong>{{ kb.name }}</strong>
        </div>
        <p v-if="kb.description" class="kb-desc">{{ kb.description }}</p>
        <div class="kb-meta">
          <span>集合: {{ kb.collection_name }}</span>
          <span>创建: {{ kb.created_at?.slice(0, 10) }}</span>
        </div>
      </div>
    </div>

    <!-- 文件列表 -->
    <div v-if="activeKb" class="kb-files-section">
      <div class="kb-files-header">
        <h3>「{{ activeKb.name }}」中的文件</h3>
        <button @click="showUpload = true" class="btn-primary-sm">
          <Upload :size="14" /> 上传文件
        </button>
      </div>
      <!-- 删除成功提示 -->
      <div v-if="deleteSuccess" class="delete-success">
        <span>✓</span> {{ deleteSuccess }}
      </div>
      <div v-if="filesLoading" style="color:#888;padding:20px">加载中…</div>
      <div v-else-if="fileList.length === 0" style="color:#888;padding:20px">暂无文件</div>
      <table v-else class="file-table">
        <thead>
          <tr>
            <th>文件名</th>
            <th>类型</th>
            <th>分块数</th>
            <th>字符数</th>
            <th>状态</th>
            <th>上传时间</th>
            <th style="width:60px"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="f in fileList"
            :key="f.id"
            class="file-row"
          >
            <td class="file-name-cell" @click.stop="openPreview(f)">
              <FileText :size="14" class="file-icon" />
              {{ f.filename }}
            </td>
            <td><span class="file-type-badge">{{ f.file_type }}</span></td>
            <td>{{ f.chunk_count }}</td>
            <td>{{ f.char_count.toLocaleString() }}</td>
            <td><span :class="['status-badge', f.status]">{{ f.status }}</span></td>
            <td>{{ f.created_at?.slice(0, 10) }}</td>
            <td>
              <button @click.stop="confirmDelete(f)" class="btn-icon-danger" title="删除文件">
                <Trash2 :size="14" />
              </button>
            </td>
          </tr>
        </tbody>
      </table>

    <!-- 删除确认弹窗 -->
    <div v-if="showDeleteConfirm" class="modal-overlay" @click.self="showDeleteConfirm = false">
      <div class="modal">
        <h3>确认删除</h3>
        <p class="delete-warning">
          确定要删除文件「<strong>{{ deleteTarget?.filename }}</strong>」吗？<br/>
          此操作将同时删除向量索引和存储的源文件，不可恢复。
        </p>
        <div class="modal-actions">
          <button @click="showDeleteConfirm = false" class="btn-secondary">取消</button>
          <button @click="doDelete" :disabled="deleting" class="btn-danger-sm">
            {{ deleting ? '删除中…' : '确认删除' }}
          </button>
        </div>
      </div>
    </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { LibraryBig, Plus, FolderOpen, Upload, FileText, Trash2 } from 'lucide-vue-next'
import api from '../api'

const route = useRoute()

const kbList = ref([])
const activeKb = ref(null)
const fileList = ref([])
const loading = ref(true)
const filesLoading = ref(false)

// 创建
const showCreate = ref(false)
const newKb = reactive({ name: '', description: '' })
const creating = ref(false)
const createError = ref('')

// 上传
const showUpload = ref(false)
const uploadFile = ref(null)
const uploading = ref(false)
const uploadMsg = ref('')
const uploadOk = ref(false)
// 进度条状态: idle | transferring | indexing | done | failed
const uploadPhase = ref('idle')
const transferProgress = ref(0)   // HTTP 传输进度 0-100（真实，axios 回调）
const indexProgress = ref(0)      // 后端索引进度 0-100（轮询 /files 接口）
let pollTimer = null

const displayProgress = computed(() => {
  if (uploadPhase.value === 'transferring') return transferProgress.value
  if (uploadPhase.value === 'indexing') return Math.max(indexProgress.value, 5)
  if (uploadPhase.value === 'done') return 100
  if (uploadPhase.value === 'failed') return 100
  return 0
})

const progressLabel = computed(() => {
  switch (uploadPhase.value) {
    case 'transferring': return '上传文件中…'
    case 'indexing': return indexProgress.value > 0 ? '索引中（解析/向量化/图谱）…' : '排队等待索引…'
    case 'done': return '完成'
    case 'failed': return '索引失败'
    default: return ''
  }
})

function formatSize(bytes) {
  if (!bytes && bytes !== 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function closeUpload() {
  // 传输阶段不允许关（关了请求也断了）；索引阶段可关，后台继续，文件列表里仍能看到进度
  if (uploading.value && uploadPhase.value === 'transferring') return
  stopPolling()
  showUpload.value = false
  uploadMsg.value = ''
  if (!uploading.value) {
    uploadPhase.value = 'idle'
    transferProgress.value = 0
    indexProgress.value = 0
  }
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

// 预览
const showPreview = ref(false)
const previewFile = ref(null)
const previewLoading = ref(false)
const previewError = ref('')
const previewText = ref('')
const previewContentType = ref('')
const rawUrl = ref('')

// 删除
const showDeleteConfirm = ref(false)
const deleteTarget = ref(null)
const deleting = ref(false)
const deleteSuccess = ref('')

async function loadKbs() {
  loading.value = true
  try {
    kbList.value = await api.get('/knowledge/bases')
  } catch { /* 忽略 */ }
  finally { loading.value = false }
}

async function createKb() {
  createError.value = ''
  if (!newKb.name.trim()) { createError.value = '请输入名称'; return }
  creating.value = true
  try {
    await api.post('/knowledge/bases', { name: newKb.name, description: newKb.description })
    showCreate.value = false
    newKb.name = ''
    newKb.description = ''
    await loadKbs()
  } catch (e) {
    createError.value = e.response?.data?.detail || '创建失败'
  } finally {
    creating.value = false
  }
}

async function selectKb(kb) {
  activeKb.value = kb
  filesLoading.value = true
  try {
    fileList.value = await api.get(`/knowledge/bases/${kb.id}/files`)
  } catch { fileList.value = [] }
  finally { filesLoading.value = false }
}

function onFileSelect(e) {
  uploadFile.value = e.target.files[0] || null
  uploadMsg.value = ''
}

async function doUpload() {
  if (!uploadFile.value || !activeKb.value) return
  uploading.value = true
  uploadMsg.value = ''
  uploadOk.value = false
  uploadPhase.value = 'transferring'
  transferProgress.value = 0
  indexProgress.value = 0

  const kbId = activeKb.value.id
  try {
    const fd = new FormData()
    fd.append('file', uploadFile.value)
    // 202 Accepted：服务器已收到文件，索引进后台
    await api.upload(`/knowledge/bases/${kbId}/upload`, fd, (e) => {
      if (e.total) transferProgress.value = Math.round((e.loaded / e.total) * 100)
    })

    // 传输完成 → 进入索引阶段，开始轮询
    uploadPhase.value = 'indexing'
    await pollIndexing(kbId)
  } catch (e) {
    uploadPhase.value = 'failed'
    uploadMsg.value = `❌ ${e.response?.data?.detail || '上传失败'}`
    uploadOk.value = false
    uploading.value = false
  }
}

// 轮询文件列表，跟踪最新那个 processing 文件的 progress，直到 completed/failed
async function pollIndexing(kbId) {
  stopPolling()
  return new Promise((resolve) => {
    pollTimer = setInterval(async () => {
      try {
        const files = await api.get(`/knowledge/bases/${kbId}/files`)
        fileList.value = files  // 同步刷新表格，关掉弹窗也能看到进度
        // 找最新一个非终态文件（本次上传的那个）
        const target = files.find((f) => f.status === 'processing' || f.status === 'pending')
        if (target) {
          indexProgress.value = target.progress ?? 0
          return  // 继续轮询
        }
        // 没有 processing 文件了 → 找最新文件看终态
        stopPolling()
        const latest = files[0]
        if (latest?.status === 'failed') {
          uploadPhase.value = 'failed'
          uploadMsg.value = `❌ 索引失败：${latest.error_message || '未知错误'}`
          uploadOk.value = false
        } else {
          uploadPhase.value = 'done'
          uploadMsg.value = `✅ 上传成功，索引了 ${latest?.chunk_count ?? 0} 个块`
          uploadOk.value = true
          uploadFile.value = null
        }
        uploading.value = false
        resolve()
      } catch {
        // 轮询失败（网络抖动）不打断，下一轮再试
      }
    }, 1500)
  })
}

async function openPreview(f) {
  showPreview.value = true
  previewFile.value = f
  previewLoading.value = true
  previewError.value = ''
  previewText.value = ''
  previewContentType.value = ''
  rawUrl.value = ''

  try {
    const kbId = activeKb.value.id

    // PDF / 图片：直接拉原始文件用浏览器原生渲染
    if (['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'webp'].includes(f.file_type)) {
      const { data: blob } = await api.getBlob(
        `/knowledge/bases/${kbId}/files/${f.id}/raw`
      )
      rawUrl.value = URL.createObjectURL(blob)
      previewContentType.value = 'binary'
    } else {
      // 文本文件：走 preview 端点提取文本
      const data = await api.get(`/knowledge/bases/${kbId}/files/${f.id}/preview`)
      previewContentType.value = data.content_type
      previewText.value = data.text_content || '(空文件)'
    }
  } catch (e) {
    previewError.value = e.response?.data?.detail || e.message || '预览失败'
  } finally {
    previewLoading.value = false
  }
}

function closePreview() {
  showPreview.value = false
  previewFile.value = null
  previewText.value = ''
  rawUrl.value = ''
}

function confirmDelete(f) {
  deleteTarget.value = f
  showDeleteConfirm.value = true
}

async function doDelete() {
  if (!deleteTarget.value || !activeKb.value) return
  deleting.value = true
  deleteSuccess.value = ''
  try {
    await api.delete(`/knowledge/bases/${activeKb.value.id}/files/${deleteTarget.value.id}`)
    showDeleteConfirm.value = false
    deleteSuccess.value = `「${deleteTarget.value.filename}」删除成功`
    deleteTarget.value = null
    await selectKb(activeKb.value)
    // 3 秒后自动隐藏成功提示
    setTimeout(() => { deleteSuccess.value = '' }, 3000)
  } catch (e) {
    const detail = e.response?.data?.detail
    const msg = detail || `HTTP ${e.response?.status || 'error'}: ${e.response?.statusText || e.message}`
    alert(`删除失败：${msg}`)
  } finally {
    deleting.value = false
  }
}

// 根据路由 query (?kb=..&file=..) 选中知识库并打开文件预览 — 对话页引用跳转入口
async function applyRouteQuery() {
  const kbId = route.query.kb
  const fileId = route.query.file
  if (!kbId) return
  if (!kbList.value.length) {
    await loadKbs()
  }
  const kb = kbList.value.find((k) => k.id === kbId)
  if (!kb) return
  // 若已是当前 kb 且文件已加载, 直接复用; 否则重新选中加载文件列表
  if (activeKb.value?.id !== kb.id) {
    await selectKb(kb)
  }
  if (fileId) {
    const f = fileList.value.find((x) => x.id === fileId)
    if (f) openPreview(f)
  }
}

onMounted(async () => {
  await loadKbs()
  await applyRouteQuery()
})

// 已在知识库页时, 从对话页再次点击引用(query 变化)也要响应
watch(() => route.query, () => {
  if (route.path === '/knowledge') applyRouteQuery()
})
</script>
