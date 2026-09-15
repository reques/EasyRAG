export function messageUsage(message) {
  return message?.meta?.tokenUsage || message?.meta?.token_usage || {}
}

export function sessionUsage(messages) {
  return messages.filter(m => m.role === 'assistant').reduce((total, m) => {
    const usage = messageUsage(m)
    for (const key of ['input_tokens', 'output_tokens', 'total_tokens']) total[key] += Number(usage[key] || 0)
    return total
  }, { input_tokens: 0, output_tokens: 0, total_tokens: 0 })
}

export function latestContextUsage(messages) {
  for (const message of [...messages].reverse()) {
    if (message.role !== 'assistant') continue
    const context = message.meta?.contextUsage || message.meta?.context_usage
    if (context && Number.isFinite(context.input_tokens)) return context
    // A legacy completed turn must not borrow an earlier turn's context size.
    if (messageUsage(message).total_tokens) return null
  }
  return null
}

export function mergeTasks(current, incoming) {
  const result = current.map(task => ({ ...task }))
  for (const task of incoming || []) {
    if (!task.task_id) continue
    const index = result.findIndex(t => t.task_id === task.task_id)
    if (index >= 0) result[index] = { ...result[index], goal: task.goal || result[index].goal, worker_hint: task.worker_hint || result[index].worker_hint }
    else result.push({ goal: '', worker_hint: '', status: 'pending', tools: [], output: '', error: '', expanded: false, ...task })
  }
  return result
}

export function shouldAutoOpen({ dismissed, narrow, pinned }, hasActivity) {
  return Boolean(hasActivity && !dismissed && (!narrow || pinned))
}

export function interruptTasks(tasks, stopped, reason = '') {
  return tasks.map(task => ['pending', 'running'].includes(task.status)
    ? { ...task, status: stopped ? 'cancelled' : 'error', error: reason }
    : task)
}

export function statusArtifacts(message, tasks = []) {
  const entries = (message?.artifacts || []).filter(a => ['file', 'report', 'document', 'worker'].includes(a.kind) && a.content)
  const result = entries.map((a, i) => ({ id: a.id || `artifact-${i}`, title: a.title || '任务产出', content: a.content }))
  for (const task of tasks) {
    if (task.output && !result.some(a => a.id === `worker-${task.task_id}`)) {
      result.push({ id: `worker-${task.task_id}`, title: task.goal || '子任务产出', content: task.output })
    }
  }
  return result
}
