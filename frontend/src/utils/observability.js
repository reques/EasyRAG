export const statusLabel = (value) =>
  ({
    completed: '已完成',
    running: '执行中',
    cancel_requested: '取消中',
    cancelled: '已取消',
    failed: '失败',
    error: '失败',
    interrupted: '已中断',
    timed_out: '超时',
    info: '记录',
    pending: '等待中',
  })[value] ||
  value ||
  '未记录'
export const remoteStateLabel = (value) =>
  ({
    available: 'Langfuse 已索引',
    loading: '正在读取 Langfuse',
    not_linked: '此次执行没有 Langfuse 关联',
    not_sampled: '此次执行未采样',
    disabled: 'Langfuse 未启用',
    misconfigured: 'Langfuse 连接配置不可用',
    pending_index: 'Langfuse 暂未索引，请稍后刷新',
    unavailable: 'Langfuse 暂不可用，本地记录仍可查看',
    response_too_large: '远端记录过大，请在 Langfuse 查看',
    invalid_link: '观测关联无效',
  })[value] || '尚未读取远端观测'
export const formatNumber = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? '—'
    : Number(value).toLocaleString('zh-CN')
export const formatMs = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? '—'
    : Number(value) >= 1000
      ? `${(Number(value) / 1000).toFixed(2)} s`
      : `${Math.round(Number(value))} ms`
export const formatMetric = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? '—'
    : Number(value).toFixed(3)
export const formatTime = (value) =>
  !value || !Number.isFinite(Date.parse(value))
    ? '—'
    : new Date(value).toLocaleString('zh-CN', { hour12: false })
export const pretty = (value) =>
  value == null
    ? '未采集内容'
    : typeof value === 'string'
      ? value
      : JSON.stringify(value, null, 2)
export function safeHttpUrl(value) {
  try {
    const u = new URL(value)
    return ['http:', 'https:'].includes(u.protocol) &&
      !u.username &&
      !u.password
      ? u.href
      : ''
  } catch {
    return ''
  }
}

const time = (raw) => {
  const value = Date.parse(raw)
  return Number.isFinite(value) ? value : null
}
// Parent-first ordering is stable for equal timestamps, orphans, and cycles.
export function observationRows(observations = []) {
  const nodes = observations
    .map((item, index) => {
      const start = time(item.started_at),
        end = time(item.completed_at)
      const measured = item.metadata?.durationMs
      const duration =
        start != null && end != null
          ? Math.max(0, end - start)
          : typeof measured === 'number' && Number.isFinite(measured)
            ? Math.max(0, measured)
            : null
      return { ...item, index, start, duration }
    })
    .sort(
      (a, b) =>
        (a.start ?? Infinity) - (b.start ?? Infinity) || a.index - b.index,
    )
  const byId = new Map(nodes.map((n) => [n.id, n])),
    children = new Map(),
    roots = []
  for (const node of nodes) {
    if (
      node.parent_id &&
      node.parent_id !== node.id &&
      byId.has(node.parent_id)
    ) {
      const siblings = children.get(node.parent_id) || []
      siblings.push(node)
      children.set(node.parent_id, siblings)
    } else roots.push(node)
  }
  const seen = new Set(),
    rows = []
  function visit(node, depth) {
    if (seen.has(node.id)) return
    seen.add(node.id)
    rows.push({ ...node, depth: Math.min(depth, 12) })
    for (const child of children.get(node.id) || []) visit(child, depth + 1)
  }
  roots.forEach((n) => visit(n, 0))
  nodes.forEach((n) => visit(n, 0))
  const starts = nodes.filter((n) => n.start != null).map((n) => n.start)
  const origin = starts.length ? Math.min(...starts) : null
  const total =
    origin == null
      ? 0
      : Math.max(
          1,
          ...nodes
            .filter((n) => n.start != null)
            .map((n) => n.start - origin + (n.duration || 0)),
        )
  return rows.map((n) => ({
    ...n,
    offset:
      n.start == null || origin == null
        ? null
        : Math.max(0, ((n.start - origin) / total) * 100),
    width:
      n.duration == null
        ? null
        : Math.min(100, Math.max(0.8, (n.duration / total) * 100)),
    total,
  }))
}

export function comparableRuns(runs) {
  if (runs.length < 2) return ''
  if (new Set(runs.map((r) => r.knowledge_base_id)).size > 1)
    return '所选运行来自不同知识库，指标仅供并列查看。'
  const snapshots = runs.map((r) => r.observability?.dataset_id)
  if (snapshots.some((x) => !x))
    return '部分历史运行没有数据集快照，无法确认评测样例完全一致。'
  if (new Set(snapshots).size > 1)
    return '所选运行的数据集快照不同，请结合样例变化解释指标差异。'
  return '评测数据集快照一致，可对照检索配置与指标变化。'
}
export function traceFilters(values) {
  return Object.fromEntries(
    Object.entries(values).filter(([, value]) => value != null && value !== ''),
  )
}
