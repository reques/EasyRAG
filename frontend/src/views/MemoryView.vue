<template>
  <div class="memory-page">
    <header class="memory-header">
      <div>
        <span class="memory-eyebrow">PERSONAL MEMORY</span>
        <h1>记忆管理</h1>
        <p>查看和修正 Agent 跨对话使用的个人事实。修改会从下一次对话开始生效。</p>
      </div>
      <button class="memory-button secondary" :disabled="loading" @click="loadFacts">
        <RefreshCw :size="15" :class="{ spin: loading }" /> 刷新
      </button>
    </header>

    <section class="memory-overview" aria-label="记忆概览">
      <article>
        <span class="memory-metric-icon"><BrainCircuit :size="18" /></span>
        <div><small>长期事实</small><strong>{{ facts.length }}</strong></div>
      </article>
      <article>
        <span class="memory-metric-icon"><MessagesSquare :size="18" /></span>
        <div><small>有来源会话</small><strong>{{ sourcedCount }}</strong></div>
      </article>
      <article>
        <span class="memory-metric-icon"><ShieldCheck :size="18" /></span>
        <div><small>情景经历</small><strong>{{ episodes.length }}</strong></div>
      </article>
    </section>

    <section class="memory-section episode-section">
      <div class="memory-toolbar">
        <div>
          <h2>任务经历</h2>
          <p>记录做过什么、结果、经验和未完成事项；相关经历会在后续任务中被召回。</p>
        </div>
      </div>
      <div v-if="!loading && episodes.length === 0" class="memory-state compact">
        <History :size="24" />
        <strong>还没有任务经历</strong>
        <span>完成包含工具操作、重要决定或失败经验的任务后，会自动形成情景记忆。</span>
      </div>
      <div v-else class="memory-list">
        <article v-for="episode in episodes" :key="episode.id" class="memory-card episode-card">
          <span class="memory-card-icon"><History :size="16" /></span>
          <div class="memory-card-body">
            <div class="episode-heading">
              <strong>{{ episode.title }}</strong>
              <span :class="['outcome', episode.outcome]">{{ outcomeText(episode.outcome) }}</span>
            </div>
            <p>{{ episode.summary }}</p>
            <small class="episode-goal">目标：{{ episode.goal }}</small>
            <small v-if="episode.lessons" class="episode-detail">经验：{{ episode.lessons }}</small>
            <small v-if="episode.unfinished" class="episode-detail">未完成：{{ episode.unfinished }}</small>
            <div class="memory-card-meta"><span><Clock3 :size="12" /> {{ formatDate(episode.created_at) }}</span></div>
          </div>
          <div class="memory-card-actions">
            <button class="danger" title="删除经历" @click="episodeDeleteTarget = episode">
              <Trash2 :size="15" />
            </button>
          </div>
        </article>
      </div>
    </section>

    <section class="memory-section">
      <div class="memory-toolbar">
        <div>
          <h2>Agent 记住的事实</h2>
          <p>每轮对话完成后，Fast LLM 会判断是否需要新增、修正或删除长期事实。</p>
        </div>
        <label class="memory-search">
          <Search :size="15" />
          <input v-model="search" type="search" placeholder="搜索记忆内容" />
        </label>
      </div>

      <div v-if="error" class="memory-state error">
        <CircleAlert :size="22" />
        <strong>记忆加载失败</strong>
        <span>{{ error }}</span>
        <button class="memory-button secondary" @click="loadFacts">重新加载</button>
      </div>
      <div v-else-if="loading" class="memory-state">
        <LoaderCircle :size="22" class="spin" />
        <strong>正在读取记忆</strong>
        <span>正在加载当前账号的长期事实。</span>
      </div>
      <div v-else-if="facts.length === 0" class="memory-state">
        <BrainCircuit :size="25" />
        <strong>还没有长期记忆</strong>
        <span>你可以在对话中说明长期偏好或明确说“请记住……”。</span>
      </div>
      <div v-else-if="filteredFacts.length === 0" class="memory-state">
        <SearchX :size="24" />
        <strong>没有匹配的记忆</strong>
        <span>换一个关键词，或清空搜索条件。</span>
        <button class="memory-button secondary" @click="search = ''">清空搜索</button>
      </div>
      <div v-else class="memory-list">
        <article v-for="fact in filteredFacts" :key="fact.id" class="memory-card">
          <span class="memory-card-icon"><Sparkles :size="16" /></span>
          <div class="memory-card-body">
            <template v-if="editingId === fact.id">
              <textarea
                v-model="editText"
                maxlength="500"
                rows="3"
                aria-label="编辑记忆"
                @keydown.ctrl.enter="saveEdit(fact)"
              />
              <div class="memory-edit-footer">
                <small>{{ editText.length }}/500 · Ctrl + Enter 保存</small>
                <div>
                  <button class="memory-button ghost" :disabled="saving" @click="cancelEdit">
                    <X :size="14" /> 取消
                  </button>
                  <button
                    class="memory-button primary"
                    :disabled="saving || !editText.trim()"
                    @click="saveEdit(fact)"
                  >
                    <Save :size="14" /> {{ saving ? '保存中…' : '保存' }}
                  </button>
                </div>
              </div>
            </template>
            <template v-else>
              <p>{{ fact.fact }}</p>
              <div class="memory-card-meta">
                <span><Clock3 :size="12" /> {{ formatDate(fact.created_at) }}</span>
                <span v-if="fact.source_conversation_id">
                  <MessagesSquare :size="12" /> 来自对话
                </span>
              </div>
            </template>
          </div>
          <div v-if="editingId !== fact.id" class="memory-card-actions">
            <button title="编辑记忆" @click="startEdit(fact)"><Pencil :size="15" /></button>
            <button class="danger" title="删除记忆" @click="deleteTarget = fact">
              <Trash2 :size="15" />
            </button>
          </div>
        </article>
      </div>
    </section>

    <Teleport to="body">
      <div v-if="deleteTarget" class="memory-modal-overlay" @click.self="deleteTarget = null">
        <div class="memory-modal">
          <h3>删除这条记忆？</h3>
          <p>{{ deleteTarget.fact }}</p>
          <small>删除后，Agent 将不再在后续对话中使用这条事实。</small>
          <div>
            <button class="memory-button secondary" @click="deleteTarget = null">取消</button>
            <button class="memory-button danger-fill" :disabled="deleting" @click="deleteFact">
              {{ deleting ? '删除中…' : '确认删除' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
    <Teleport to="body">
      <div v-if="episodeDeleteTarget" class="memory-modal-overlay" @click.self="episodeDeleteTarget = null">
        <div class="memory-modal">
          <h3>删除这段任务经历？</h3>
          <p>{{ episodeDeleteTarget.title }}</p>
          <small>删除后，这段经验不会再被后续任务召回。</small>
          <div>
            <button class="memory-button secondary" @click="episodeDeleteTarget = null">取消</button>
            <button class="memory-button danger-fill" :disabled="deletingEpisode" @click="deleteEpisode">
              {{ deletingEpisode ? '删除中…' : '确认删除' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import {
  BrainCircuit, CircleAlert, Clock3, History, LoaderCircle, MessagesSquare, Pencil,
  RefreshCw, Save, Search, SearchX, ShieldCheck, Sparkles, Trash2, X,
} from 'lucide-vue-next'
import api from '../api'

const facts = ref([])
const episodes = ref([])
const loading = ref(false)
const error = ref('')
const search = ref('')
const editingId = ref(null)
const editText = ref('')
const saving = ref(false)
const deleteTarget = ref(null)
const deleting = ref(false)
const episodeDeleteTarget = ref(null)
const deletingEpisode = ref(false)

const sourcedCount = computed(() => facts.value.filter(item => item.source_conversation_id).length)
const filteredFacts = computed(() => {
  const needle = search.value.trim().toLowerCase()
  return needle
    ? facts.value.filter(item => item.fact.toLowerCase().includes(needle))
    : facts.value
})

function errorDetail(err) {
  return err.response?.data?.detail || err.message || '未知错误'
}

async function loadFacts() {
  loading.value = true
  error.value = ''
  try {
    const [loadedFacts, loadedEpisodes] = await Promise.all([
      api.get('/memory/facts', { limit: 200 }),
      api.get('/memory/episodes', { limit: 100 }),
    ])
    facts.value = loadedFacts
    episodes.value = loadedEpisodes
  } catch (err) {
    error.value = errorDetail(err)
  } finally {
    loading.value = false
  }
}

function startEdit(fact) {
  editingId.value = fact.id
  editText.value = fact.fact
}

function cancelEdit() {
  editingId.value = null
  editText.value = ''
}

async function saveEdit(fact) {
  const value = editText.value.trim()
  if (!value || saving.value) return
  saving.value = true
  try {
    const updated = await api.patch(`/memory/facts/${fact.id}`, { fact: value })
    facts.value = facts.value.map(item => item.id === fact.id ? updated : item)
    cancelEdit()
  } catch (err) {
    alert(`保存失败：${errorDetail(err)}`)
  } finally {
    saving.value = false
  }
}

async function deleteFact() {
  if (!deleteTarget.value || deleting.value) return
  deleting.value = true
  const id = deleteTarget.value.id
  try {
    await api.delete(`/memory/facts/${id}`)
    facts.value = facts.value.filter(item => item.id !== id)
    if (editingId.value === id) cancelEdit()
    deleteTarget.value = null
  } catch (err) {
    alert(`删除失败：${errorDetail(err)}`)
  } finally {
    deleting.value = false
  }
}

async function deleteEpisode() {
  if (!episodeDeleteTarget.value || deletingEpisode.value) return
  deletingEpisode.value = true
  const id = episodeDeleteTarget.value.id
  try {
    await api.delete(`/memory/episodes/${id}`)
    episodes.value = episodes.value.filter(item => item.id !== id)
    episodeDeleteTarget.value = null
  } catch (err) {
    alert(`删除失败：${errorDetail(err)}`)
  } finally {
    deletingEpisode.value = false
  }
}

function outcomeText(value) {
  return ({ success: '成功', partial: '部分完成', failed: '失败' })[value] || value
}

function formatDate(value) {
  if (!value) return '未知时间'
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric', month: 'short', day: 'numeric',
  }).format(new Date(value))
}

onMounted(loadFacts)
</script>

<style scoped>
.memory-page { height: 100%; overflow: auto; padding: 34px 42px 60px; background: var(--gray-25); }
.memory-header { max-width: 980px; margin: 0 auto 24px; display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }
.memory-eyebrow { display: block; color: var(--gray-500); font-size: 11px; font-weight: 650; letter-spacing: .13em; margin-bottom: 7px; }
.memory-header h1 { font-size: 27px; line-height: 1.25; color: var(--gray-1000); margin-bottom: 7px; }
.memory-header p, .memory-toolbar p { color: var(--gray-600); font-size: 13px; }
.memory-button { border-radius: var(--radius-md); padding: 7px 13px; display: inline-flex; align-items: center; justify-content: center; gap: 6px; border: 1px solid var(--gray-150); font-size: 13px; }
.memory-button.secondary, .memory-button.ghost { color: var(--gray-700); background: #fff; }
.memory-button.primary { color: #fff; background: var(--main-700); border-color: var(--main-700); }
.memory-button.danger-fill { color: #fff; background: var(--gray-900); border-color: var(--gray-900); }
.memory-button:disabled { opacity: .45; cursor: not-allowed; }
.memory-overview { max-width: 980px; margin: 0 auto 24px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.memory-overview article { background: #fff; border: 1px solid var(--gray-150); border-radius: var(--radius-lg); padding: 16px; display: flex; gap: 12px; align-items: center; }
.memory-metric-icon { width: 36px; height: 36px; border-radius: 8px; background: var(--gray-50); display: grid; place-items: center; }
.memory-overview small { color: var(--gray-500); display: block; font-size: 11px; }
.memory-overview strong { font-size: 21px; line-height: 1.1; }
.memory-section { max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid var(--gray-150); border-radius: var(--radius-lg); overflow: hidden; }
.episode-section { margin-top: 18px; }
.memory-toolbar { padding: 18px 20px; border-bottom: 1px solid var(--gray-100); display: flex; justify-content: space-between; gap: 20px; align-items: center; }
.memory-toolbar h2 { font-size: 16px; margin-bottom: 2px; }
.memory-search { width: 250px; height: 36px; border: 1px solid var(--gray-150); border-radius: var(--radius-md); display: flex; align-items: center; gap: 8px; padding: 0 10px; color: var(--gray-500); }
.memory-search:focus-within { border-color: var(--gray-400); }
.memory-search input { border: 0; outline: 0; min-width: 0; width: 100%; color: var(--gray-900); }
.memory-list { padding: 8px; }
.memory-card { display: flex; gap: 12px; padding: 14px 12px; border-radius: var(--radius-md); border-bottom: 1px solid var(--gray-100); }
.memory-card:last-child { border-bottom: 0; }
.memory-card:hover { background: var(--gray-25); }
.memory-card-icon { width: 30px; height: 30px; border-radius: 7px; background: var(--gray-50); display: grid; place-items: center; flex: 0 0 auto; }
.memory-card-body { flex: 1; min-width: 0; }
.memory-card-body p { font-size: 14px; color: var(--gray-900); overflow-wrap: anywhere; }
.memory-card-meta { display: flex; gap: 14px; margin-top: 7px; color: var(--gray-500); font-size: 11px; }
.memory-card-meta span { display: inline-flex; align-items: center; gap: 4px; }
.memory-card-actions { display: flex; gap: 3px; opacity: 0; transition: opacity .15s; }
.memory-card:hover .memory-card-actions { opacity: 1; }
.memory-card-actions button { width: 30px; height: 30px; border: 0; border-radius: 6px; background: transparent; color: var(--gray-500); display: grid; place-items: center; }
.memory-card-actions button:hover { background: var(--gray-100); color: var(--gray-900); }
.memory-card-actions button.danger:hover { color: var(--color-error-700); }
.memory-card textarea { width: 100%; resize: vertical; border: 1px solid var(--gray-200); border-radius: var(--radius-md); padding: 9px 10px; outline: 0; color: var(--gray-900); }
.memory-card textarea:focus { border-color: var(--gray-500); }
.memory-edit-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; }
.memory-edit-footer small { color: var(--gray-500); }
.memory-edit-footer > div { display: flex; gap: 7px; }
.memory-state { min-height: 270px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; color: var(--gray-500); text-align: center; padding: 30px; }
.memory-state strong { color: var(--gray-800); font-size: 14px; }
.memory-state span { font-size: 12px; max-width: 380px; }
.memory-state .memory-button { margin-top: 6px; }
.memory-state.compact { min-height: 180px; }
.episode-heading { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
.episode-heading strong { color: var(--gray-900); font-size: 14px; }
.outcome { padding: 2px 7px; border-radius: 999px; font-size: 10px; background: var(--gray-100); color: var(--gray-600); }
.outcome.success { background: #ecfdf3; color: #067647; }
.outcome.partial { background: #fffaeb; color: #b54708; }
.outcome.failed { background: #fef3f2; color: #b42318; }
.episode-goal, .episode-detail { display: block; margin-top: 6px; color: var(--gray-600); line-height: 1.5; }
.memory-modal-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.22); display: grid; place-items: center; padding: 20px; }
.memory-modal { width: min(430px, 92vw); background: #fff; border-radius: 12px; padding: 22px; box-shadow: var(--shadow-deep); }
.memory-modal h3 { font-size: 17px; margin-bottom: 10px; }
.memory-modal p { background: var(--gray-50); border-radius: 7px; padding: 10px; color: var(--gray-800); font-size: 13px; }
.memory-modal small { display: block; color: var(--gray-500); margin: 9px 0 18px; }
.memory-modal > div { display: flex; justify-content: flex-end; gap: 8px; }
.spin { animation: memory-spin .9s linear infinite; }
@keyframes memory-spin { to { transform: rotate(360deg); } }
@media (max-width: 760px) {
  .memory-page { padding: 24px 16px 48px; }
  .memory-header, .memory-toolbar { flex-direction: column; align-items: stretch; }
  .memory-overview { grid-template-columns: 1fr; }
  .memory-search { width: 100%; }
  .memory-card-actions { opacity: 1; }
}
</style>
