import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api/index.js'
import { groupSessions, readSessionOwner, selectionKey } from '../utils/workspace-sessions.js'

export const useChatStore = defineStore('chat', () => {
  const owner = ref(readSessionOwner(localStorage))
  const conversations = ref([])
  const workspaces = ref([])
  const sections = ref([])
  const archivedConversations = ref([])
  const activeConversationId = ref(localStorage.getItem(selectionKey(owner.value)) || null)
  const draft = ref(activeConversationId.value ? null : {
    id: 'draft-0', title: '新对话', workspace_id: null, is_draft: true, has_messages: false, has_content: false,
  })
  const loading = ref(false)
  const loadError = ref('')
  const newConversationVersion = ref(0)
  const grouped = computed(() => groupSessions(workspaces.value, draft.value ? [draft.value, ...conversations.value] : conversations.value))
  const activeConversation = computed(() => draft.value || conversations.value.find(c => c.id === activeConversationId.value))
  const activeWorkspaceId = computed(() => activeConversation.value?.workspace_id || null)
  let loadVersion = 0
  let titleRefresh = null
  let draftSerial = 0

  function resetForUser() {
    const current = readSessionOwner(localStorage)
    if (current === owner.value) return
    owner.value = current
    ++loadVersion
    clearTimeout(titleRefresh)
    conversations.value = []
    workspaces.value = []
    sections.value = []
    draft.value = null
    archivedConversations.value = []
    activeConversationId.value = localStorage.getItem(selectionKey(current)) || null
    if (!activeConversationId.value) draft.value = {
      id: `draft-${++draftSerial}`, title: '新对话', workspace_id: null,
      is_draft: true, has_messages: false, has_content: false,
    }
    newConversationVersion.value++
    loadError.value = ''
    loading.value = false
  }

  async function loadConversations() {
    resetForUser()
    const version = ++loadVersion
    const requestedOwner = owner.value
    loading.value = true
    loadError.value = ''
    try {
      const [workspaceList, sectionList, firstPage] = await Promise.all([
        api.get('/chat/workspaces'), api.get('/chat/workspace-sections'), api.get('/chat/conversations', { limit: 200, offset: 0 }),
      ])
      const all = [...firstPage]
      let page = firstPage
      while (page.length === 200) {
        page = await api.get('/chat/conversations', { limit: 200, offset: all.length })
        all.push(...page)
      }
      if (version !== loadVersion || requestedOwner !== readSessionOwner(localStorage)) return
      conversations.value = [...new Map(all.map(c => [c.id, c])).values()]
      workspaces.value = workspaceList
      sections.value = sectionList
      if (draft.value?.workspace_id && !workspaceList.some(w => w.id === draft.value.workspace_id)) draft.value.workspace_id = null
      const selected = all.find(c => c.id === activeConversationId.value)
      if (selected?.has_messages === false) createConversation(selected.workspace_id)
      if (activeConversationId.value && !all.some(c => c.id === activeConversationId.value)) {
        const selectedId = activeConversationId.value
        try {
          const conv = await api.get(`/chat/conversations/${selectedId}`)
          if (version === loadVersion) conversations.value.push(conv)
        } catch (error) {
          if (version === loadVersion && selectedId === activeConversationId.value && error.response?.status === 404) startNewConversation()
          else throw error
        }
      }
    } catch (error) {
      if (version === loadVersion && requestedOwner === readSessionOwner(localStorage)) {
        loadError.value = error.response?.data?.detail || '会话列表加载失败，请重试'
      }
    } finally {
      if (version === loadVersion) loading.value = false
    }
  }

  function selectConversation(id) {
    if (id && id === draft.value?.id) return
    draft.value = null
    activeConversationId.value = id
    if (id) localStorage.setItem(selectionKey(owner.value), id)
    else localStorage.removeItem(selectionKey(owner.value))
  }

  function startNewConversation(workspaceId = null) {
    return createConversation(workspaceId)
  }

  function createConversation(workspaceId = activeWorkspaceId.value) {
    if (draft.value?.workspace_id === workspaceId && !draft.value.has_content) return draft.value
    const changed = !!activeConversationId.value || !!draft.value
    selectConversation(null)
    const conv = { id: `draft-${++draftSerial}`, title: '新对话', workspace_id: workspaceId,
      is_draft: true, has_messages: false, has_content: false }
    draft.value = conv
    if (changed) newConversationVersion.value++
    return conv
  }

  function registerConversation(id, workspaceId, requestedOwner = owner.value) {
    if (requestedOwner !== readSessionOwner(localStorage)) return
    if (!conversations.value.some(c => c.id === id)) {
      conversations.value.unshift({ id, session_id: id, title: '新对话', workspace_id: workspaceId, has_messages: true })
    }
  }

  async function createWorkspace(name, localPath = null) {
    const requestedOwner = owner.value
    const workspace = await api.post('/chat/workspaces', { name, local_path: localPath })
    if (requestedOwner !== readSessionOwner(localStorage)) return null
    ++loadVersion
    loading.value = false
    workspaces.value.push(workspace)
    return workspace
  }

  async function updateConversation(id, changes) {
    const requestedOwner = owner.value
    const conv = await api.patch(`/chat/conversations/${id}`, changes)
    if (requestedOwner !== readSessionOwner(localStorage)) return null
    ++loadVersion
    loading.value = false
    conversations.value = conversations.value.map(c => c.id === id ? conv : c)
    return conv
  }

  async function refreshAfterSend() {
    const requestedOwner = owner.value
    await loadConversations()
    clearTimeout(titleRefresh)
    titleRefresh = setTimeout(() => {
      if (requestedOwner === readSessionOwner(localStorage)) void loadConversations()
    }, 3000)
  }

  async function updateWorkspace(id, changes) {
    const requestedOwner = owner.value
    const workspace = await api.patch(`/chat/workspaces/${id}`, changes)
    if (requestedOwner !== readSessionOwner(localStorage)) return null
    ++loadVersion
    loading.value = false
    workspaces.value = workspaces.value.map(w => w.id === id ? { ...w, ...workspace } : w)
    return workspace
  }

  async function loadArchivedConversations() {
    const requestedOwner = owner.value
    const all = []
    let page
    do {
      page = await api.get('/chat/conversations', { archived: true, limit: 200, offset: all.length })
      all.push(...page)
    } while (page.length === 200)
    if (requestedOwner === readSessionOwner(localStorage)) archivedConversations.value = all.filter(c => c.has_messages !== false)
  }

  async function restoreConversation(id) {
    const requestedOwner = owner.value
    const conv = await api.post(`/chat/conversations/${id}/restore`)
    if (requestedOwner !== readSessionOwner(localStorage)) return null
    archivedConversations.value = archivedConversations.value.filter(c => c.id !== id)
    conversations.value = [conv, ...conversations.value.filter(c => c.id !== id)]
    return conv
  }

  return {
    conversations, workspaces, activeConversationId, activeConversation, activeWorkspaceId,
    grouped, loading, loadError, newConversationVersion, owner, draft, sections, archivedConversations,
    resetForUser, loadConversations, selectConversation, startNewConversation,
    createConversation, createWorkspace, updateConversation, refreshAfterSend, registerConversation,
    updateWorkspace, loadArchivedConversations, restoreConversation,
  }
})
