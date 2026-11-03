export function groupProviders(models) {
  const groups = new Map()
  for (const model of models || []) {
    const name = String(model.provider || '其他供应商').trim() || '其他供应商'
    const key = name.toLocaleLowerCase()
    if (!groups.has(key)) groups.set(key, { id: key, name, models: [] })
    groups.get(key).models.push(model)
  }
  return [...groups.values()]
}

export function filterAgents(agents, query, kind) {
  const text = String(query || '').trim().toLocaleLowerCase()
  return (agents || []).filter(agent => agent.kind === kind && (!text ||
    [agent.name, agent.slug, agent.description].some(value => String(value || '').toLocaleLowerCase().includes(text))))
}
