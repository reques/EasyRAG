// The server capability is authoritative; localStorage role flags never grant access.
export async function loadMcpConsole(api) {
  const overview = await api.get('/mcp/overview')
  const readonly = { canManage: false, servers: overview?.servers || [], definitions: [] }
  if (overview?.can_manage !== true) return readonly
  try {
    const config = await api.get('/mcp/servers')
    return { canManage: true, servers: config?.servers || [], definitions: config?.definitions || [] }
  } catch (error) {
    // A role may be revoked between the capability query and configuration read.
    if (error.response?.status === 403) return readonly
    throw error
  }
}
