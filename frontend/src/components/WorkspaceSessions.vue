<template>
  <section class="workspace-sessions" aria-label="工作区与会话">
    <div class="workspace-section-heading">
      <span>项目 / 工作区</span>
      <button type="button" class="workspace-icon-button" title="新建工作区" aria-label="新建工作区" @click="openDialog('create-workspace')"><FolderPlus :size="16" /></button>
    </div>
    <label class="workspace-search"><Search :size="13" /><input v-model="search" aria-label="搜索会话或工作区" placeholder="搜索会话…" /></label>
    <div v-if="store.loadError || actionError" class="workspace-error" role="alert">
      {{ actionError || store.loadError }}
      <button type="button" @click="actionError = ''; store.loadConversations()">重试</button>
    </div>
    <div v-if="store.loading && !store.conversations.length && !store.workspaces.length" class="workspace-empty">正在加载…</div>
    <div class="workspace-tree">
      <section v-for="bucket in visibleBuckets" :key="bucket.id" class="workspace-bucket">
        <div v-if="bucket.groups.length || bucket.custom" class="workspace-bucket-heading"><span>{{ bucket.name }}</span><button v-if="bucket.custom" class="workspace-icon-button" :aria-label="`${bucket.name}分区菜单`" @click="openMenu('section', bucket, $event)"><MoreHorizontal :size="13" /></button></div>
      <section v-for="group in bucket.groups" :key="group.id || 'ungrouped'" class="workspace-group">
        <div class="workspace-group-heading" :class="{ current: store.activeWorkspaceId === group.id && store.activeConversationId }">
          <button type="button" class="workspace-folder" :aria-expanded="isOpen(group)" @click="toggleGroup(group)">
            <ChevronDown :size="12" :class="{ collapsed: !isOpen(group) }" />
            <component :is="group.id ? Folder : MessagesSquare" :size="16" />
            <span>{{ group.name }}</span><Pin v-if="group.is_pinned" :size="11" /><small>{{ group.conversations.length }}</small>
          </button>
          <button v-if="group.id" type="button" class="workspace-icon-button workspace-project-menu-button" :class="{ opened: menu?.kind === 'workspace' && menu.item.id === group.id }" :aria-label="`${group.name}工作区菜单`" :aria-expanded="menu?.kind === 'workspace' && menu.item.id === group.id" :aria-controls="menu?.kind === 'workspace' && menu.item.id === group.id ? 'workspace-project-menu' : undefined" title="工作区菜单" @click="openMenu('workspace', group, $event)"><MoreHorizontal :size="17" :stroke-width="2" /></button>
          <button type="button" class="workspace-icon-button group-action" :aria-label="`在${group.name}中新建会话`" title="新建会话" :disabled="creating" @click="newChat(group.id)"><Plus :size="14" /></button>
        </div>
        <div v-if="isOpen(group)" class="workspace-children">
          <div v-for="conv in displayedConversations(group)" :key="conv.id" class="workspace-conversation" :class="{ selected: isSelected(conv) }">
            <button type="button" class="workspace-conversation-link" :title="conv.is_draft ? '未发送的临时会话，切换后自动消失' : `${conv.title || '新对话'}\nSession: ${conv.id}`" :aria-current="isSelected(conv) ? 'page' : undefined" @click="select(conv.id)">
              <span class="workspace-session-dot"></span><span>{{ conv.title || '新对话' }}</span>
              <small v-if="conv.is_draft" class="workspace-draft-label">草稿</small>
            </button>
            <button v-if="!conv.is_draft" type="button" class="workspace-icon-button conversation-action" :aria-label="`${conv.title || '新对话'}会话菜单`" @click="openMenu('conversation', conv, $event)"><MoreHorizontal :size="15" /></button>
          </div>
          <button v-if="!search && group.conversations.length > 5" type="button" class="workspace-show-more" @click="expanded[key(group)] = !expanded[key(group)]">{{ expanded[key(group)] ? '收起列表' : `展开全部 ${group.conversations.length} 个会话` }}</button>
          <button v-if="!group.conversations.length && !search" type="button" class="workspace-empty-create" :disabled="creating" @click="newChat(group.id)"><Plus :size="12" /> 创建第一个会话</button>
        </div>
      </section>
      <div v-if="bucket.custom && !bucket.groups.length" class="workspace-empty-section">从项目菜单将项目移到此分区</div>
      </section>
      <div v-if="!visibleGroups.length" class="workspace-empty">没有匹配的工作区或会话</div>
      <button v-if="!store.workspaces.length && !search" type="button" class="workspace-onboard" @click="openDialog('create-workspace')"><FolderPlus :size="17" /><span>按项目整理会话<small>为不同工作创建独立空间</small></span></button>
    </div>
    <button type="button" class="workspace-archives-button" @click="showArchives"><Archive :size="14" /> 已归档聊天</button>

    <Teleport to="body">
      <div v-if="menu" class="conv-menu-overlay" @click="menu = null" @contextmenu.prevent="menu = null" @keydown.esc="menu = null">
        <div :id="menu.kind === 'workspace' ? 'workspace-project-menu' : undefined" class="conv-menu workspace-menu" :style="menuStyle" @click.stop>
          <template v-if="menu.kind === 'workspace'">
            <button class="conv-menu-item" @click="togglePin(menu.item)"><Pin :size="14" /> {{ menu.item.is_pinned ? '取消置顶' : '置顶' }}</button>
            <button class="conv-menu-item" @click="openDialog('edit-workspace', menu.item)"><Settings2 :size="14" /> 编辑</button>
            <div class="workspace-menu-divider"></div>
            <button class="conv-menu-item" @click="openDialog('section-workspace', menu.item)"><List :size="14" /> 分区 <ChevronRight :size="13" class="workspace-menu-arrow" /></button>
            <button class="conv-menu-item" @click="openDirectory(menu.item)"><FolderOpen :size="14" /> 在资源管理器中打开</button>
            <div class="workspace-menu-divider"></div>
            <button class="conv-menu-item" @click="openDialog('archive-workspace', menu.item)"><Archive :size="14" /> 归档聊天</button>
            <div class="workspace-menu-divider"></div>
            <button class="conv-menu-item danger" @click="openDialog('remove-workspace', menu.item)"><X :size="14" /> 移除项目</button>
          </template>
          <template v-else-if="menu.kind === 'section'">
            <button class="conv-menu-item" @click="openDialog('rename-section', menu.item)"><Pencil :size="14" /> 重命名分区</button>
            <button class="conv-menu-item danger" @click="openDialog('remove-section', menu.item)"><X :size="14" /> 移除分区</button>
          </template>
          <template v-else>
            <button class="conv-menu-item" @click="openDialog('rename-conversation', menu.item)"><Pencil :size="14" /> 重命名</button>
            <button class="conv-menu-item" @click="openDialog('move-conversation', menu.item)"><FolderInput :size="14" /> 移动到工作区</button>
            <button class="conv-menu-item" @click="copySessionId(menu.item.id)"><Copy :size="14" /> 复制 Session ID</button>
            <button class="conv-menu-item" :disabled="summarizing === menu.item.id" @click="summarize(menu.item.id)"><Sparkles :size="14" /> {{ summarizing === menu.item.id ? '生成中…' : '生成摘要标题' }}</button>
            <button class="conv-menu-item" @click="openDialog('archive-conversation', menu.item)"><Archive :size="14" /> 归档聊天</button>
            <button class="conv-menu-item danger" @click="openDialog('delete-conversation', menu.item)"><Trash2 :size="14" /> 删除会话</button>
          </template>
        </div>
      </div>
      <div v-if="dialog" class="modal-overlay" @click.self="closeDialog" @keydown.esc="closeDialog" @keydown.tab="trapFocus">
        <form ref="dialogEl" class="modal workspace-dialog" role="dialog" aria-modal="true" aria-labelledby="workspace-dialog-title" @submit.prevent="submitDialog">
          <div class="workspace-dialog-heading"><component :is="dialog.type.includes('workspace') ? Folder : MessagesSquare" :size="19" /><h3 id="workspace-dialog-title">{{ dialogTitle }}</h3></div>
          <template v-if="dialog.type === 'archives'">
            <p class="workspace-dialog-description">归档保留全部历史。恢复后可用原 Session ID 继续会话。</p>
            <div v-if="archiveLoading" class="workspace-empty">正在加载…</div>
            <div v-else-if="!store.archivedConversations.length" class="workspace-empty">暂无归档聊天</div>
            <div v-else class="workspace-archive-list"><div v-for="conv in store.archivedConversations" :key="conv.id"><span><strong>{{ conv.title || '新对话' }}</strong><small>{{ store.workspaces.find(w => w.id === conv.workspace_id)?.name || '最近' }} · {{ conv.id.slice(0, 8) }}</small></span><button type="button" class="btn-secondary" :disabled="busy" @click="restore(conv)">恢复并打开</button></div></div>
          </template>
          <p v-else-if="dialog.type.startsWith('archive-')" class="workspace-dialog-description">归档「{{ dialog.item.name || dialog.item.title }}」{{ dialog.type === 'archive-workspace' ? '中的所有聊天' : '' }}？会从侧栏隐藏，可在“已归档聊天”中恢复。正在运行的任务需要先停止。</p>
          <p v-else-if="dialog.type === 'remove-section'" class="workspace-dialog-description">移除「{{ dialog.item.name }}」分区？其中的项目会回到默认项目列表，聊天记录全部保留。</p>
          <p v-else-if="dialog.type === 'remove-workspace'" class="workspace-dialog-description">移除「{{ dialog.item.name }}」后，里面的会话会移到“最近”。消息、任务和 Session ID 都会保留。</p>
          <p v-else-if="dialog.type === 'delete-conversation'" class="delete-warning">永久删除「{{ dialog.item.title || '新对话' }}」及其消息？此操作无法恢复。运行中的会话需要先停止任务。</p>
          <template v-else-if="dialog.type === 'move-conversation'">
            <label for="workspace-destination">目标工作区</label>
            <select id="workspace-destination" ref="inputEl" v-model="destination" :disabled="busy"><option value="">最近</option><option v-for="workspace in store.workspaces" :key="workspace.id" :value="workspace.id">{{ workspace.name }}</option></select>
            <p class="workspace-dialog-description">保留当前会话的 Session ID、历史和运行状态。</p>
          </template>
          <template v-else-if="dialog.type === 'section-workspace'">
            <label for="workspace-section">项目分区</label><select id="workspace-section" ref="inputEl" v-model="sectionDestination" :disabled="busy"><option value="">默认项目列表</option><option v-for="section in store.sections" :key="section.id" :value="section.id">{{ section.name }}</option><option value="__new__">＋ 新建分区</option></select>
            <template v-if="sectionDestination === '__new__'"><label for="workspace-new-section">分区名称</label><input id="workspace-new-section" v-model="newSectionName" maxlength="80" required :disabled="busy" placeholder="例如：研究、工作、个人" /></template>
            <p class="workspace-dialog-description">分区用于整理项目；置顶项目会优先显示在顶部。</p>
          </template>
          <template v-else>
            <label for="workspace-name">{{ dialog.type.includes('workspace') ? '项目名称' : dialog.type.includes('section') ? '分区名称' : '会话名称' }}</label>
            <input id="workspace-name" ref="inputEl" v-model="name" :maxlength="dialog.type.includes('workspace') ? 80 : 256" required :disabled="busy" autocomplete="off" :placeholder="dialog.type.includes('workspace') ? '例如：论文研究、产品开发' : '为这次会话命名'" />
            <p v-if="dialog.type === 'create-workspace'" class="workspace-dialog-description">同一项目下管理多个会话，每个会话拥有独立的历史与 Session ID。</p>
            <template v-if="['create-workspace', 'edit-workspace'].includes(dialog.type)"><label for="workspace-local-path">本机项目目录（可选）</label><input id="workspace-local-path" v-model="localPath" maxlength="1024" :disabled="busy" placeholder="E:\LearnAgent\EasyRAG" autocomplete="off" /><p class="workspace-dialog-description">关联本机已有文件夹，用于在资源管理器中打开。不会上传文件或向 Agent 自动授权目录访问。</p></template>
          </template>
          <p v-if="dialogError" class="workspace-error" role="alert">{{ dialogError }}</p>
          <div class="modal-actions"><button type="button" class="btn-secondary" :disabled="busy" @click="closeDialog">{{ dialog.type === 'archives' ? '关闭' : '取消' }}</button><button v-if="dialog.type !== 'archives'" type="submit" :class="dialog.type.startsWith('delete') || dialog.type.startsWith('remove') ? 'btn-danger-sm' : 'btn-primary'" :disabled="busy || (needsName && !name.trim()) || (dialog.type === 'section-workspace' && sectionDestination === '__new__' && !newSectionName.trim())">{{ busy ? '保存中…' : dialog.type === 'create-workspace' ? '创建工作区' : '确认' }}</button></div>
        </form>
      </div>
    </Teleport>
    <span class="workspace-sr-only" role="status">{{ notice }}</span>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Archive, ChevronDown, ChevronRight, Copy, Folder, FolderInput, FolderOpen, FolderPlus, List, MessagesSquare, MoreHorizontal, Pencil, Pin, Plus, Search, Settings2, Sparkles, Trash2, X } from 'lucide-vue-next'
import { useChatStore } from '../stores/chat'
import api from '../api'
import { sectionWorkspaces } from '../utils/workspace-sessions.js'
import { openWorkspaceDirectory } from '../api/workspace-opener.js'

const store = useChatStore()
const emit = defineEmits(['select'])
const search = ref('')
const collapsed = ref({})
const expanded = ref({})
const menu = ref(null)
const menuStyle = ref({})
const dialog = ref(null)
const dialogEl = ref(null)
const inputEl = ref(null)
const name = ref('')
const destination = ref('')
const localPath = ref('')
const sectionDestination = ref('')
const newSectionName = ref('')
const archiveLoading = ref(false)
const busy = ref(false)
const creating = ref(false)
const summarizing = ref(null)
const actionError = ref('')
const dialogError = ref('')
const notice = ref('')
let restoreFocus = null
const key = group => group.id || 'ungrouped'
const allGroups = computed(() => [...store.grouped.groups, store.grouped.ungrouped])
const visibleGroups = computed(() => {
  const query = search.value.trim().toLocaleLowerCase()
  if (!query) return allGroups.value
  return allGroups.value.map(group => ({ ...group, conversations: group.name.toLocaleLowerCase().includes(query) ? group.conversations : group.conversations.filter(c => `${c.title || ''} ${c.id}`.toLocaleLowerCase().includes(query)) })).filter(group => group.conversations.length || group.name.toLocaleLowerCase().includes(query))
})
const visibleBuckets = computed(() => [
  ...sectionWorkspaces(visibleGroups.value.filter(g => g.id), store.sections),
  { id: 'ungrouped', name: '', groups: visibleGroups.value.filter(g => !g.id) },
])
const isSelected = conv => conv.is_draft ? store.draft?.id === conv.id : store.activeConversationId === conv.id
const isOpen = group => !!search.value || !collapsed.value[key(group)]
const displayedConversations = group => search.value || expanded.value[key(group)] ? group.conversations : group.conversations.slice(0, 5)
const needsName = computed(() => ['create-workspace', 'edit-workspace', 'rename-conversation', 'rename-section'].includes(dialog.value?.type))
const dialogTitle = computed(() => ({ 'create-workspace': '新建工作区', 'edit-workspace': '编辑项目', 'remove-workspace': '移除项目', 'rename-conversation': '重命名会话', 'move-conversation': '移动会话', 'delete-conversation': '删除会话', 'section-workspace': '项目分区', 'rename-section': '重命名分区', 'remove-section': '移除分区', 'archive-workspace': '归档项目中的聊天', 'archive-conversation': '归档聊天', archives: '已归档聊天' }[dialog.value?.type]))
const detail = error => error.response?.data?.detail || error.message || '操作失败，请重试'

watch(() => store.owner, owner => {
  try { collapsed.value = JSON.parse(localStorage.getItem(`easyrag_workspace_collapse:${owner}`) || '{}') || {} } catch { collapsed.value = {} }
  expanded.value = {}
  menu.value = null
  dialog.value = null
}, { immediate: true })
watch(collapsed, value => localStorage.setItem(`easyrag_workspace_collapse:${store.owner}`, JSON.stringify(value)), { deep: true })
watch(() => [store.activeConversationId, store.conversations, store.draft], () => {
  const group = allGroups.value.find(g => g.conversations.some(isSelected))
  if (!group) return
  collapsed.value[key(group)] = false
  if (group.conversations.findIndex(isSelected) >= 5) expanded.value[key(group)] = true
}, { immediate: true })

function toggleGroup(group) { collapsed.value[key(group)] = isOpen(group) }
function select(id) { store.selectConversation(id); emit('select') }
async function newChat(workspaceId = store.activeWorkspaceId) {
  if (creating.value) return
  creating.value = true
  actionError.value = ''
  try {
    const conv = await store.createConversation(workspaceId)
    if (conv && (store.activeConversationId === conv.id || store.draft?.id === conv.id)) emit('select')
  } catch (error) { actionError.value = detail(error) }
  finally { creating.value = false }
}
defineExpose({ newChat })

function openMenu(kind, item, event) {
  if (menu.value?.kind === kind && menu.value.item.id === item.id) { menu.value = null; return }
  const rect = event.currentTarget.getBoundingClientRect()
  menuStyle.value = { top: `${Math.max(8, Math.min(rect.bottom + 4, window.innerHeight - 280))}px`, left: `${Math.max(8, Math.min(rect.left, window.innerWidth - 230))}px` }
  menu.value = { kind, item, opener: event.currentTarget }
  nextTick(() => document.querySelector('.workspace-menu button')?.focus())
}
function openDialog(type, item = null) {
  restoreFocus = menu.value?.opener || document.activeElement
  menu.value = null
  dialog.value = { type, item }
  name.value = item?.name || item?.title || ''
  destination.value = item?.workspace_id || ''
  localPath.value = item?.local_path || ''
  sectionDestination.value = item?.section_id || ''
  newSectionName.value = ''
  dialogError.value = ''
  nextTick(() => { (inputEl.value || dialogEl.value?.querySelector('button'))?.focus(); inputEl.value?.select?.() })
}
function closeDialog() { if (busy.value) return; dialog.value = null; restoreFocus?.focus?.() }
function trapFocus(event) {
  const items = [...dialogEl.value.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled)')]
  const target = event.shiftKey ? items.at(-1) : items[0]
  if (document.activeElement === (event.shiftKey ? items[0] : items.at(-1))) { event.preventDefault(); target?.focus() }
}
async function submitDialog() {
  if (busy.value) return
  busy.value = true
  dialogError.value = ''
  const { type, item } = dialog.value
  const operationOwner = store.owner
  try {
    if (type === 'create-workspace') {
      const workspace = await store.createWorkspace(name.value.trim(), localPath.value.trim() || null)
      if (workspace) await newChat(workspace.id)
    } else if (type === 'edit-workspace') {
      await store.updateWorkspace(item.id, { name: name.value.trim(), local_path: localPath.value.trim() || null })
    } else if (type === 'section-workspace') {
      let id = sectionDestination.value || null
      if (id === '__new__') {
        const section = await api.post('/chat/workspace-sections', { name: newSectionName.value.trim() })
        if (operationOwner !== store.owner) return
        id = section.id
        store.sections.push(section)
        sectionDestination.value = id
      }
      await store.updateWorkspace(item.id, { section_id: id })
    } else if (type === 'rename-section') {
      await api.patch(`/chat/workspace-sections/${item.id}`, { name: name.value.trim() })
      await store.loadConversations()
    } else if (type === 'remove-section') {
      await api.delete(`/chat/workspace-sections/${item.id}`)
      await store.loadConversations()
    } else if (type.startsWith('archive-')) {
      const result = await api.post(type === 'archive-workspace' ? `/chat/workspaces/${item.id}/archive` : `/chat/conversations/${item.id}/archive`)
      const ids = result.conversation_ids || [item.id]
      if (ids.includes(store.activeConversationId)) store.startNewConversation(type === 'archive-workspace' ? item.id : item.workspace_id)
      await store.loadConversations()
    } else if (type === 'remove-workspace') {
      await api.delete(`/chat/workspaces/${item.id}`)
      await store.loadConversations()
    } else if (type === 'delete-conversation') {
      await api.delete(`/chat/conversations/${item.id}`)
      if (store.activeConversationId === item.id) store.startNewConversation()
      await store.loadConversations()
    } else {
      await store.updateConversation(item.id, type === 'rename-conversation' ? { title: name.value.trim() } : { workspace_id: destination.value || null })
    }
    dialog.value = null
    restoreFocus?.focus?.()
  } catch (error) { dialogError.value = detail(error) }
  finally { busy.value = false }
}
async function togglePin(item) {
  menu.value = null
  try { await store.updateWorkspace(item.id, { is_pinned: !item.is_pinned }) }
  catch (error) { actionError.value = detail(error) }
}
async function openDirectory(item) {
  menu.value = null
  if (!item.local_path) { openDialog('edit-workspace', item); dialogError.value = '请先关联本机项目目录，保存后再打开。'; return }
  try { await openWorkspaceDirectory(item.id); notice.value = '已在资源管理器中打开项目目录' }
  catch (error) { actionError.value = detail(error) }
}
async function showArchives() {
  openDialog('archives')
  archiveLoading.value = true
  try { await store.loadArchivedConversations() }
  catch (error) { dialogError.value = detail(error) }
  finally { archiveLoading.value = false }
}
async function restore(conv) {
  if (busy.value) return
  busy.value = true
  try {
    const restored = await store.restoreConversation(conv.id)
    if (restored) { store.selectConversation(conv.id); dialog.value = null; emit('select') }
  } catch (error) { dialogError.value = detail(error) }
  finally { busy.value = false }
}
async function copySessionId(id) {
  menu.value = null
  try { await navigator.clipboard.writeText(id); notice.value = `已复制 Session ID：${id}` }
  catch { actionError.value = '复制失败，请从顶部 Session 按钮查看完整 ID' }
}
async function summarize(id) {
  menu.value = null
  summarizing.value = id
  try { await api.post(`/chat/conversations/${id}/summarize`); await store.loadConversations() }
  catch (error) { actionError.value = detail(error) }
  finally { summarizing.value = null }
}
function closeMenu(event) { if (event?.type !== 'keydown' || event.key === 'Escape') menu.value = null }
onMounted(() => { window.addEventListener('scroll', closeMenu, true); window.addEventListener('resize', closeMenu); window.addEventListener('keydown', closeMenu) })
onBeforeUnmount(() => { window.removeEventListener('scroll', closeMenu, true); window.removeEventListener('resize', closeMenu); window.removeEventListener('keydown', closeMenu) })
</script>

<style scoped>
.workspace-sessions { min-height: 0; flex: 1; display: flex; flex-direction: column; gap: 12px; }
.workspace-section-heading { display: flex; align-items: center; justify-content: space-between; padding: 0 8px; color: var(--gray-500); font-size: 11px; font-weight: 650; letter-spacing: .03em; }
.workspace-icon-button { width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; padding: 0; border: 0; border-radius: 6px; color: var(--gray-500); background: transparent; cursor: pointer; }
.workspace-icon-button:hover, .workspace-icon-button:focus-visible { background: var(--gray-150); color: var(--gray-950); }
.workspace-search { margin: 0 4px; padding: 6px 9px; display: flex; align-items: center; gap: 7px; border: 1px solid var(--gray-150); border-radius: 7px; color: var(--gray-400); background: rgba(255,255,255,.5); }
.workspace-search input { width: 100%; min-width: 0; padding: 0; border: 0; outline: none; background: transparent; font-size: 11px; }
.workspace-search:focus-within { border-color: var(--gray-500); }
.workspace-tree { min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 0 2px 18px; scrollbar-gutter: stable; }
.workspace-group + .workspace-group { margin-top: 18px; }
.workspace-group-heading { display: flex; align-items: center; gap: 1px; height: 33px; border-radius: 7px; }
.workspace-folder { display: flex; flex: 1; min-width: 0; height: 100%; align-items: center; gap: 7px; padding: 0 4px 0 0; text-align: left; border: 0; background: transparent; color: var(--gray-800); font-size: 12px; cursor: pointer; }
.workspace-folder > span { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.workspace-folder > svg { flex: 0 0 auto; }
.workspace-folder > svg:first-child { color: var(--gray-400); transition: transform .15s; }
.workspace-folder > svg.collapsed { transform: rotate(-90deg); }
.workspace-folder small { margin-left: auto; color: var(--gray-400); font-size: 10px; font-variant-numeric: tabular-nums; }
.workspace-group-heading:hover { background: rgba(255,255,255,.55); }
.workspace-group-heading.current .workspace-folder { color: var(--gray-1000); font-weight: 650; }
.group-action { width: 22px; opacity: .4; }
.workspace-project-menu-button { width: 28px; height: 28px; opacity: 1; color: var(--gray-700); }
.workspace-project-menu-button.opened { background: var(--gray-150); color: var(--gray-1000); }
.workspace-group-heading:hover .group-action, .group-action:focus-visible { opacity: 1; }
.workspace-children { padding-top: 3px; margin-left: 15px; }
.workspace-conversation { display: flex; align-items: center; min-width: 0; min-height: 35px; margin-bottom: 2px; border-radius: 7px; }
.workspace-conversation:hover { background: var(--gray-100); }
.workspace-conversation.selected { background: #fff; box-shadow: inset 0 0 0 1px var(--gray-150); }
.workspace-conversation-link { display: flex; flex: 1; align-items: center; min-width: 0; gap: 8px; padding: 9px 5px 9px 9px; border: 0; background: transparent; text-align: left; color: var(--gray-700); font-size: 12px; cursor: pointer; }
.workspace-conversation-link > span:last-child { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.selected .workspace-conversation-link { color: var(--gray-1000); }
.workspace-session-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--gray-300); flex: 0 0 auto; }
.selected .workspace-session-dot { background: var(--main-500); }
.conversation-action { margin-right: 3px; opacity: 0; }
.workspace-conversation:hover .conversation-action, .workspace-conversation:focus-within .conversation-action { opacity: 1; }
.workspace-show-more, .workspace-empty-create { display: flex; align-items: center; gap: 5px; padding: 8px 9px; background: transparent; border: 0; color: var(--gray-500); font-size: 11px; cursor: pointer; }
.workspace-show-more:hover, .workspace-empty-create:hover { color: var(--gray-950); }
.workspace-onboard { width: calc(100% - 8px); display: flex; gap: 10px; align-items: flex-start; margin: 22px 4px 0; padding: 14px 11px; border: 1px dashed var(--gray-200); border-radius: 9px; background: transparent; color: var(--gray-600); text-align: left; font-size: 11px; cursor: pointer; }
.workspace-onboard small { display: block; margin-top: 4px; font-size: 10px; color: var(--gray-400); }
.workspace-empty { padding: 12px 8px; font-size: 12px; color: var(--gray-500); }
.workspace-error { padding: 8px; color: var(--color-error-500); font-size: 12px; line-height: 1.6; }
.workspace-error button { margin-left: 6px; text-decoration: underline; background: transparent; border: 0; color: inherit; cursor: pointer; }
.workspace-menu { min-width: 200px; }
.workspace-dialog { width: min(440px, calc(100vw - 32px)); }
.workspace-dialog-heading { display: flex; align-items: center; gap: 9px; margin-bottom: 20px; }
.workspace-dialog-heading h3 { margin: 0; }
.workspace-dialog label { display: block; margin-bottom: 8px; font-size: 12px; color: var(--gray-700); }
.workspace-dialog input, .workspace-dialog select { width: 100%; padding: 10px 12px; border: 1px solid var(--gray-200); border-radius: 8px; background: #fff; color: var(--gray-950); font: inherit; font-size: 13px; }
.workspace-dialog-description { margin: 10px 0 16px; color: var(--gray-500); font-size: 12px; line-height: 1.8; }
.workspace-sr-only { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
.workspace-bucket + .workspace-bucket { margin-top: 16px; }
.workspace-bucket-heading { min-height: 22px; display: flex; align-items: center; justify-content: space-between; padding-left: 3px; color: var(--gray-400); font-size: 10px; }
.workspace-bucket-heading:has(> span:empty) { display: none; }
.workspace-draft-label { margin-left: auto; flex: 0 0 auto; color: var(--gray-400); font-size: 9px; }
.workspace-empty-section { padding: 8px 4px; font-size: 10px; color: var(--gray-400); }
.workspace-menu-divider { height: 1px; margin: 5px 8px; background: var(--gray-150); }
.workspace-menu-arrow { margin-left: auto; }
.workspace-archives-button { display: flex; align-items: center; gap: 8px; padding: 8px; flex-shrink: 0; border: 0; border-top: 1px solid var(--gray-150); background: transparent; color: var(--gray-500); text-align: left; font-size: 11px; cursor: pointer; }
.workspace-archives-button:hover { color: var(--gray-950); }
.workspace-archive-list { max-height: min(400px, 50vh); overflow-y: auto; }
.workspace-archive-list > div { display: flex; align-items: center; gap: 10px; justify-content: space-between; padding-block: 12px; border-bottom: 1px solid var(--gray-150); }
.workspace-archive-list > div > span { min-width: 0; }
.workspace-archive-list strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
.workspace-archive-list small { display: block; margin-top: 4px; color: var(--gray-400); font-size: 10px; }
.workspace-archive-list button { flex-shrink: 0; font-size: 11px; }
.workspace-dialog input + label, .workspace-dialog select + label { margin-top: 14px; }
@media (prefers-reduced-motion: reduce) { .workspace-folder > svg:first-child { transition: none; } }
@media (hover: none) { .group-action, .conversation-action { opacity: 1; } }
</style>
