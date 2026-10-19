export function groupSessions(workspaces, conversations) {
  const groups = workspaces.map(workspace => ({ ...workspace, conversations: [] }))
  const byId = new Map(groups.map(group => [group.id, group]))
  const ungrouped = { id: null, name: '最近', conversations: [] }
  for (const conversation of conversations) {
    if (conversation.archived_at || (conversation.has_messages === false && !conversation.is_draft)) continue
    (byId.get(conversation.workspace_id) || ungrouped).conversations.push(conversation)
  }
  return { groups, ungrouped }
}

export function sectionWorkspaces(groups, sections) {
  const pinned = groups.filter(group => group.is_pinned)
  const remaining = groups.filter(group => !group.is_pinned)
  const known = new Set(sections.map(section => section.id))
  return [
    ...(pinned.length ? [{ id: 'pinned', name: '置顶', groups: pinned }] : []),
    { id: 'projects', name: '项目', groups: remaining.filter(group => !known.has(group.section_id)) },
    ...sections.map(section => ({ ...section, custom: true, groups: remaining.filter(group => group.section_id === section.id) })),
  ]
}

export function readSessionOwner(storage) {
  try { return JSON.parse(storage.getItem('user') || 'null')?.user_id || 'guest' }
  catch { return 'guest' }
}

export function selectionKey(owner) {
  return `easyrag_session_selection:${owner}`
}
