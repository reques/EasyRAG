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
    <div v-if="showUpload" class="modal-overlay" @click.self="showUpload = false">
      <div class="modal">
        <h3>上传文档到「{{ activeKb?.name }}」</h3>
        <label class="file-label">
          <input type="file" @change="onFileSelect" accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg,.bmp,.webp" />
          <span v-if="!uploadFile">点击选择文件 (.txt .md .pdf .docx 图片)</span>
          <span v-else>📄 {{ uploadFile.name }}</span>
        </label>
        <p v-if="uploadMsg" :class="uploadOk ? 'auth-success' : 'auth-error'">{{ uploadMsg }}</p>
        <div class="modal-actions">
          <button @click="showUpload = false; uploadMsg = ''" class="btn-secondary">关闭</button>
          <button @click="doUpload" :disabled="!uploadFile || uploading" class="btn-primary-sm">
            {{ uploading ? '上传中…' : '上传并索引' }}
          </button>
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
          </tr>
        </thead>
        <tbody>
          <tr v-for="f in fileList" :key="f.id">
            <td>{{ f.filename }}</td>
            <td><span class="file-type-badge">{{ f.file_type }}</span></td>
            <td>{{ f.chunk_count }}</td>
            <td>{{ f.char_count.toLocaleString() }}</td>
            <td><span :class="['status-badge', f.status]">{{ f.status }}</span></td>
            <td>{{ f.created_at?.slice(0, 10) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { LibraryBig, Plus, FolderOpen, Upload } from 'lucide-vue-next'
import api from '../api'

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
  try {
    const fd = new FormData()
    fd.append('file', uploadFile.value)
    const res = await api.upload(`/knowledge/bases/${activeKb.value.id}/upload`, fd)
    uploadMsg.value = `✅ 上传成功，索引了 ${res.indexed} 个块`
    uploadOk.value = true
    uploadFile.value = null
    // 刷新文件列表
    await selectKb(activeKb.value)
  } catch (e) {
    uploadMsg.value = `❌ ${e.response?.data?.detail || '上传失败'}`
    uploadOk.value = false
  } finally {
    uploading.value = false
  }
}

onMounted(loadKbs)
</script>
