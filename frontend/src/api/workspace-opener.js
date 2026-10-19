export async function openWorkspaceDirectory(workspaceId) {
  let response
  try {
    response = await fetch('http://127.0.0.1:18765/open-workspace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${localStorage.getItem('token') || ''}` },
      body: JSON.stringify({ workspace_id: workspaceId }),
      signal: AbortSignal.timeout(8000),
    })
  } catch {
    throw new Error('无法连接本机目录打开服务。请先运行 scripts/start-workspace-opener.ps1，并允许浏览器访问本机网络。')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || '无法打开项目目录')
  }
  return response.json()
}
