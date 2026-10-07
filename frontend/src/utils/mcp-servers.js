/**
 * MCP 服务控制台纯逻辑层。
 *
 * 只处理数据、不含任何 Vue / DOM 依赖，便于 `node --test` 直接覆盖：
 *  - 后端 `GET /mcp/servers` 返回的 definitions(静态配置) 与 servers(实时状态) 合并
 *  - 连接状态归一化、运行时长、错误信息提取
 *  - 传输协议 / 工具白名单 / stdio 命令的展示与校验
 *  - 创建、编辑表单与后端 Pydantic 模型的互转
 *
 * 后端契约见 app/tools/mcp/models.py 与 backend/server/routers/mcp_router.py。
 */

/** 后端 MCPTransportType 枚举 → 界面文案。 */
export const TRANSPORT_LABELS = {
  sse: 'SSE',
  streamable_http: 'Streamable HTTP',
  stdio: 'Stdio',
}

/** 后端 MCPServerBase.validate_stdio_safety 的受信任可执行文件白名单。 */
export const TRUSTED_STDIO_BINARIES = ['python', 'python3', 'py', 'node', 'npx', 'uv']

export const TIMEOUT_RANGE = { min: 5, max: 300, default: 30 }

/** 状态归一化枚举（视图与徽标样式共用）。 */
export const SERVER_STATES = ['running', 'connecting', 'error', 'standby']

export const STATE_LABELS = {
  running: '运行中',
  connecting: '连接中',
  error: '连接异常',
  standby: '待命',
}

export const FILTERS = [
  { value: 'all', label: '全部' },
  { value: 'running', label: '运行中' },
  { value: 'standby', label: '待命' },
  { value: 'error', label: '异常' },
]

export function serverIdOf(server) {
  return String(server?.server_id || server?.name || '').trim()
}

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function normalizeServer(server, id) {
  const stdioCommand = asArray(server.stdio_command).map(String).filter(Boolean)
  return {
    ...server,
    server_id: id,
    name: server.name || id,
    description: server.description || '',
    transport: server.transport || 'sse',
    tools: asArray(server.tools),
    allowed_tools: asArray(server.allowed_tools),
    capabilities: asArray(server.capabilities).map(String).filter(Boolean),
    stdio_command: stdioCommand,
    url: server.url || '',
    timeout_s: Number(server.timeout_s || TIMEOUT_RANGE.default),
    enabled: server.enabled !== false,
    running: Boolean(server.running),
  }
}

/**
 * 合并 `{ servers, definitions }`：以实时状态为准，补全 definitions 里才有的
 * description / capabilities / timeout_s / stdio_command 等字段。
 */
export function mergeServers(liveServers = [], definitions = []) {
  const defs = new Map()
  for (const def of definitions) {
    const id = serverIdOf(def)
    if (id) defs.set(id, def)
  }

  const merged = []
  const seen = new Set()
  for (const live of liveServers) {
    const id = serverIdOf(live)
    if (!id) continue
    seen.add(id)
    merged.push(normalizeServer({ ...(defs.get(id) || {}), ...live }, id))
  }
  // 兜底：已配置但未被状态接口返回的服务，也按待命展示，避免配置"隐身"。
  for (const [id, def] of defs) {
    if (!seen.has(id)) merged.push(normalizeServer({ ...def, running: false }, id))
  }
  return merged
}

/** 归一化连接状态：running > connecting > error > standby。 */
export function serverState(server) {
  if (server?.running) return 'running'
  const raw = String(server?.status || '').toLowerCase()
  if (raw === 'connecting') return 'connecting'
  if (raw === 'error' || raw === 'failed') return 'error'
  if (serverError(server)) return 'error'
  return 'standby'
}

export function stateLabel(state) {
  return STATE_LABELS[state] || STATE_LABELS.standby
}

/** 运行/连接失败原因（status 接口给 error，definition 给 last_error）。 */
export function serverError(server) {
  const message = server?.error || server?.last_error || ''
  return String(message).trim()
}

/** 后端 started_at 为 time.time() 秒级时间戳；兼容毫秒输入。 */
export function uptimeText(startedAt, now = Date.now()) {
  const started = Number(startedAt)
  if (!Number.isFinite(started) || started <= 0) return ''
  const startedMs = started > 1e12 ? started : started * 1000
  const seconds = Math.max(0, Math.floor((now - startedMs) / 1000))
  if (seconds < 60) return `${seconds} 秒`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} 分钟`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} 小时 ${minutes % 60} 分钟`
  return `${Math.floor(hours / 24)} 天 ${hours % 24} 小时`
}

/** 工具白名单摘要：'*' 表示全量放行，[] 表示全部禁用。 */
export function allowedToolsSummary(allowed) {
  const list = asArray(allowed)
  if (list.includes('*')) return { mode: 'all', count: 0, label: '全部工具放行' }
  if (list.length === 0) return { mode: 'none', count: 0, label: '未放行任何工具' }
  return { mode: 'custom', count: list.length, label: `白名单 ${list.length} 个工具` }
}

export function isToolAllowed(allowed, toolName) {
  const list = asArray(allowed)
  return list.includes('*') || list.includes(toolName)
}

/** 后端注册工具时的命名空间：{server_id}__{tool_name}。 */
export function toolKey(serverId, rawName) {
  return `${serverId}__${rawName}`
}

export function toolNamespacedName(serverId, tool) {
  return tool?.namespaced_name || toolKey(serverId, tool?.name || '')
}

/** `GET /mcp/tools` 结果 → namespaced_name 索引，用于展开时补全工具描述。 */
export function buildToolCatalog(toolInfos = []) {
  const catalog = {}
  for (const info of toolInfos) {
    const key = info?.namespaced_name || toolKey(info?.server_id || '', info?.raw_name || '')
    if (key) catalog[key] = info
  }
  return catalog
}

/** 输入串（空格/换行/逗号分隔）→ 去重的字符串数组。 */
export function parseList(text) {
  return String(text ?? '')
    .split(/[\s,，;；]+/)
    .map(item => item.trim())
    .filter(Boolean)
}

/** 命令行文本 → argv 数组（与创建表单里 stdio_command 的录入方式一致）。 */
export function parseCommandLine(text) {
  return String(text ?? '').trim().split(/\s+/).filter(Boolean)
}

/** `KEY=VALUE` 多行文本 → 环境变量字典（stdio 子进程用）。 */
export function parseEnvText(text) {
  const env = {}
  for (const rawLine of String(text ?? '').split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue
    const index = line.indexOf('=')
    if (index < 0) continue
    const key = line.slice(0, index).trim()
    const value = line.slice(index + 1).trim()
    if (key) env[key] = value
  }
  return env
}

/** 环境变量字典 → `KEY=VALUE` 多行文本。 */
export function formatEnvText(env) {
  return Object.entries(env || {}).map(([key, value]) => `${key}=${value}`).join('\n')
}

/** 找出不符合 `KEY=VALUE` 格式的行（供表单校验提示）。 */
export function invalidEnvLines(text) {
  return String(text ?? '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#') && !/^[A-Za-z_][A-Za-z0-9_]*\s*=/.test(line))
}

export function formatCommand(command) {
  return asArray(command).join(' ')
}

/** stdio 首段可执行文件是否在后端白名单内（前后端同源，避免提交后才报错）。 */
export function isTrustedBinary(command) {
  const first = asArray(command)[0]
  if (!first) return false
  const base = String(first).toLowerCase().replace(/\.exe$/, '').split(/[\\/]/).pop()
  return TRUSTED_STDIO_BINARIES.includes(base)
}

/** 概览指标。 */
export function serverMetrics(servers = []) {
  const metrics = { total: servers.length, running: 0, connecting: 0, error: 0, standby: 0, tools: 0, whitelisted: 0 }
  for (const server of servers) {
    metrics[serverState(server)] += 1
    metrics.tools += server.tools.length
    const summary = allowedToolsSummary(server.allowed_tools)
    if (summary.mode !== 'all') metrics.whitelisted += 1
  }
  return metrics
}

/**
 * 状态筛选判定：连接中的服务也归入「运行中」，避免用户以为服务消失了。
 * 视图与筛选计数共用同一判定，保证 tab 数字与实际列表一致。
 */
export function serverMatchesFilter(server, filter = 'all') {
  if (filter === 'all') return true
  const state = serverState(server)
  if (filter === 'running') return state === 'running' || state === 'connecting'
  return state === filter
}

/** 各筛选 tab 的计数（all 为总数）。 */
export function filterCounts(servers = []) {
  const counts = { all: servers.length }
  for (const option of FILTERS) {
    if (option.value === 'all') continue
    counts[option.value] = servers.filter(server => serverMatchesFilter(server, option.value)).length
  }
  return counts
}

/** 关键词 + 状态筛选（关键词覆盖服务标识、名称、描述、能力标签与工具名）。 */
export function filterServers(servers = [], { query = '', filter = 'all' } = {}) {
  const needle = String(query).trim().toLowerCase()
  return servers.filter((server) => {
    if (!serverMatchesFilter(server, filter)) return false
    if (!needle) return true
    const haystack = [
      server.server_id,
      server.name,
      server.description,
      TRANSPORT_LABELS[server.transport] || server.transport,
      ...server.capabilities,
      ...server.tools.map(tool => tool.name),
      ...server.allowed_tools,
    ]
      .join(' ')
      .toLowerCase()
    return haystack.includes(needle)
  })
}

// ── 表单 ⇄ 后端模型 ──────────────────────────────────────────────────────

export function emptyForm() {
  return {
    server_id: '',
    name: '',
    description: '',
    transport: 'sse',
    url: '',
    auth_token: '',
    stdio_command: '',
    env_text: '',
    timeout_s: TIMEOUT_RANGE.default,
    enabled: true,
    scope: 'all', // all | custom —— 工具白名单模式
    allowed_tools: '',
    capabilities: '',
  }
}

/** 服务定义 → 编辑表单（server_id / transport / stdio_command 不可通过 PUT 变更）。 */
export function serverToForm(server) {
  const summary = allowedToolsSummary(server?.allowed_tools)
  return {
    ...emptyForm(),
    server_id: serverIdOf(server),
    name: server?.name || '',
    description: server?.description || '',
    transport: server?.transport || 'sse',
    url: server?.url || '',
    auth_token: server?.auth_token || '',
    stdio_command: formatCommand(server?.stdio_command),
    env_text: formatEnvText(server?.env),
    timeout_s: Number(server?.timeout_s || TIMEOUT_RANGE.default),
    enabled: server?.enabled !== false,
    scope: summary.mode === 'all' ? 'all' : 'custom',
    allowed_tools: summary.mode === 'custom' || summary.mode === 'none'
      ? asArray(server?.allowed_tools).join(', ')
      : '',
    capabilities: asArray(server?.capabilities).join(', '),
  }
}

/** 校验表单，返回字段级错误对象（空对象表示通过）。 */
export function validateForm(form, { mode = 'create' } = {}) {
  const errors = {}
  const serverId = String(form.server_id || '').trim()
  const name = String(form.name || '').trim()
  const timeout = Number(form.timeout_s)

  if (mode === 'create') {
    if (!serverId) errors.server_id = '请填写服务标识符'
    else if (!/^[a-zA-Z0-9_-]+$/.test(serverId)) errors.server_id = '仅允许英文、数字、下划线与连字符'
  }
  if (!name) errors.name = '请填写服务展示名称'

  if (form.transport === 'stdio') {
    const command = parseCommandLine(form.stdio_command)
    if (mode === 'create' && command.length === 0) errors.stdio_command = '请填写启动命令'
    else if (mode === 'create' && !isTrustedBinary(command)) {
      errors.stdio_command = `首段必须是受信任程序：${TRUSTED_STDIO_BINARIES.join(' / ')}`
    }
    const badEnvLines = invalidEnvLines(form.env_text)
    if (badEnvLines.length) errors.env_text = `环境变量需为 KEY=VALUE 格式：${badEnvLines[0]}`
  } else {
    const url = String(form.url || '').trim()
    if (!url) errors.url = '远程服务必须填写端点 URL'
    else if (!/^https?:\/\//i.test(url)) errors.url = 'URL 需以 http:// 或 https:// 开头'
  }

  if (!Number.isFinite(timeout) || timeout < TIMEOUT_RANGE.min || timeout > TIMEOUT_RANGE.max) {
    errors.timeout_s = `超时时间需在 ${TIMEOUT_RANGE.min} - ${TIMEOUT_RANGE.max} 秒之间`
  }

  if (form.scope === 'custom' && parseList(form.allowed_tools).length === 0) {
    errors.allowed_tools = '选择「白名单」后至少填写一个工具名'
  }
  return errors
}

export function formErrorList(errors) {
  return Object.values(errors || {})
}

/** 表单 → 后端 MCPServerCreate / MCPServerUpdate 载荷。 */
export function formToPayload(form, { mode = 'create' } = {}) {
  const payload = {
    name: String(form.name || '').trim(),
    description: String(form.description || '').trim(),
    timeout_s: Number(form.timeout_s),
    enabled: Boolean(form.enabled),
    allowed_tools: form.scope === 'all' ? ['*'] : parseList(form.allowed_tools),
    capabilities: parseList(form.capabilities),
  }
  if (mode === 'create') {
    payload.server_id = String(form.server_id || '').trim()
    payload.transport = form.transport
    if (form.transport === 'stdio') {
      payload.stdio_command = parseCommandLine(form.stdio_command)
    }
  }
  if (form.transport === 'stdio') {
    // stdio 服务的 API Key 等变量；编辑时始终上送，空值即删除该变量
    payload.env = parseEnvText(form.env_text)
  }
  if (form.transport !== 'stdio') {
    payload.url = String(form.url || '').trim()
    const token = String(form.auth_token || '').trim()
    if (token) payload.auth_token = token
  }
  return payload
}

// ── MCP 广场（ModelScope 目录）─────────────────────────────────────────────

/** 热度计数：12556 → '1.3万'，1234 → '1234'。 */
export function formatCount(value) {
  const count = Number(value || 0)
  if (!Number.isFinite(count) || count <= 0) return '0'
  if (count < 10000) return String(count)
  return `${(count / 10000).toFixed(1).replace(/\.0$/, '')}万`
}

/** 广场 id → 默认 server_id（与后端 slugify_catalog_id 保持一致）。 */
export function catalogServerId(catalogId) {
  return String(catalogId || '')
    .replace(/^@/, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'mcp-server'
}

/**
 * 安装计划 → 安装表单初值。
 * 计划由后端 `/mcp/catalog/{id}` 返回，已解析好传输协议、命令、必填环境变量。
 */
export function planToInstallForm(plan = {}) {
  const env = {}
  for (const key of plan.env_schema || []) env[key] = ''
  for (const [key, value] of Object.entries(plan.env_defaults || {})) env[key] = value
  return {
    server_id: plan.server_id_suggestion || catalogServerId(plan.catalog_id),
    name: plan.name || '',
    kind: plan.kind || 'deploy_required',
    transport: plan.transport || 'streamable_http',
    url: plan.url || '',
    stdio_command: plan.stdio_command || [],
    env,
    scope: 'all',
    allowed_tools: '',
    timeout_s: TIMEOUT_RANGE.default,
    enabled: true,
    overwrite: false,
  }
}

/** 集成的安装表单校验（在后端校验之前先给出可读提示）。 */
export function validateInstallForm(form, plan = {}) {
  const errors = {}
  const serverId = String(form.server_id || '').trim()
  if (!serverId) errors.server_id = '请填写服务标识符'
  else if (!/^[a-zA-Z0-9_-]+$/.test(serverId)) errors.server_id = '仅允许英文、数字、下划线与连字符'

  if (form.kind === 'deploy_required') {
    const url = String(form.url || '').trim()
    if (!url) errors.url = '该服务需要先在 ModelScope 部署，或手动填写端点 URL'
    else if (!/^https?:\/\//i.test(url)) errors.url = 'URL 需以 http:// 或 https:// 开头'
  }

  for (const key of plan.env_schema || []) {
    if (!String(form.env?.[key] || '').trim()) errors[`env.${key}`] = `请填写 ${key}`
  }

  if (form.kind === 'stdio') {
    const runtime = plan.runtime || {}
    // 运行环境缺失（如镜像未内置 uvx）只警告不拦截：用户可能自行补齐运行环境
    if (runtime.trusted === false) errors.runtime = `启动器 ${runtime.binary || ''} 不在受信任白名单内`
  }

  if (form.scope === 'custom' && parseList(form.allowed_tools).length === 0) {
    errors.allowed_tools = '选择「白名单」后至少填写一个工具名'
  }
  return errors
}

/** 安装表单 → 后端 `POST /mcp/catalog/{id}/install` 载荷。 */
export function installFormToPayload(form) {
  const payload = {
    server_id: String(form.server_id || '').trim(),
    name: String(form.name || '').trim(),
    env: Object.fromEntries(
      Object.entries(form.env || {}).map(([key, value]) => [key, String(value ?? '').trim()]),
    ),
    transport: form.kind === 'remote' || form.kind === 'deploy_required' ? form.transport : undefined,
    timeout_s: Number(form.timeout_s),
    enabled: Boolean(form.enabled),
    allowed_tools: form.scope === 'all' ? ['*'] : parseList(form.allowed_tools),
    overwrite: Boolean(form.overwrite),
  }
  if (form.kind === 'deploy_required') payload.url = String(form.url || '').trim()
  return payload
}

