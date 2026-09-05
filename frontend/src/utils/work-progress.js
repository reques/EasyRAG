/** Apply one artifact to the live process without depending on adjacency. */
export function appendWorkArtifact(artifacts, workItems, event, wid) {
  const id = event.id || `art-${wid}`
  const update = item => ({
    ...item,
    content: (item.content || '') + (event.content || ''),
    streaming: event.streaming !== false,
    streamed: true,
  })
  if (event.id && 'streaming' in event && artifacts.some(item => item.id === id)) {
    return {
      artifacts: artifacts.map(item => item.id === id ? update(item) : item),
      workItems: workItems.map(item => item.t === 'artifact' && item.id === id ? update(item) : item),
    }
  }
  if (event.streaming === false && !event.content) return { artifacts, workItems }
  const artifact = {
    id, kind: event.kind || 'info', stage: event.stage || '', title: event.title || '',
    content: event.content || '', streaming: !!event.streaming,
    sequence: event.sequence, tool_call_id: event.tool_call_id || '', is_error: !!event.is_error,
  }
  return {
    artifacts: [...artifacts, artifact],
    workItems: [...workItems, { t: 'artifact', wid, _ts: Date.now(), ...artifact }],
  }
}

/** New messages carry shared event sequence numbers; old messages retain their fallback order. */
export function buildHistoryWorkItems(message) {
  const meta = message.meta || {}
  const artifacts = (meta.artifacts || []).filter(item => item?.kind && item.kind !== 'answer')
  const steps = (meta.steps || []).filter(item => item?.step && !(
    meta.intent === 'dynamic' && /动态\s*Agent/.test(item.detail || '')
  ))
  const workers = (meta.worker_outputs || []).filter(item => item?.kind)
  const ordered = [...steps, ...artifacts, ...workers]
  const hasSequence = ordered.length && ordered.every(item => Number.isFinite(item.sequence))
  const entries = hasSequence
    ? ordered.sort((a, b) => a.sequence - b.sequence)
    : meta.intent === 'deepagents' ? [...artifacts, ...workers, ...steps] : [...steps, ...artifacts, ...workers]
  return entries.map((item, index) => ({
    ...item,
    t: item.step ? 'step' : 'artifact',
    wid: `h${index + 1}`,
    id: item.id || `h-art-${index + 1}`,
    streaming: false,
  }))
}
