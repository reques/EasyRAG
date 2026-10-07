import assert from 'node:assert/strict'
import test from 'node:test'
import {
  allowedToolsSummary,
  buildToolCatalog,
  catalogServerId,
  emptyForm,
  filterCounts,
  filterServers,
  formatCommand,
  formatCount,
  formatEnvText,
  formToPayload,
  installFormToPayload,
  invalidEnvLines,
  isTrustedBinary,
  isToolAllowed,
  mergeServers,
  parseCommandLine,
  parseEnvText,
  parseList,
  planToInstallForm,
  serverMetrics,
  serverMatchesFilter,
  serverState,
  serverToForm,
  stateLabel,
  toolNamespacedName,
  uptimeText,
  validateForm,
  validateInstallForm,
} from '../src/utils/mcp-servers.js'

// 与后端 GET /mcp/servers 真实返回结构对齐的样例。
const definitions = [
  {
    server_id: 'filesystem',
    name: 'filesystem',
    description: '只读文件访问',
    transport: 'stdio',
    stdio_command: ['npx', '-y', '@modelcontextprotocol/server-filesystem', './volumes'],
    capabilities: ['fs.read'],
    allowed_tools: ['read_file', 'list_directory'],
    enabled: true,
    timeout_s: 60,
  },
  {
    server_id: 'system-health',
    name: 'system-health',
    description: '依赖健康检查',
    transport: 'stdio',
    stdio_command: ['python', '-m', 'app.tools.mcp.health_server'],
    capabilities: ['ops.monitor'],
    allowed_tools: ['*'],
    enabled: false,
    timeout_s: 30,
  },
]

const live = [
  {
    server_id: 'filesystem',
    name: 'filesystem',
    transport: 'stdio',
    enabled: true,
    running: true,
    status: 'connected',
    error: null,
    started_at: 1_700_000_000,
    tools: [{ name: 'read_file', namespaced_name: 'filesystem__read_file', enabled: true }],
    allowed_tools: ['read_file', 'list_directory'],
  },
  {
    server_id: 'system-health',
    name: 'system-health',
    transport: 'stdio',
    enabled: false,
    running: false,
    status: 'error',
    error: 'spawn python ENOENT',
    started_at: null,
    tools: [],
    allowed_tools: ['*'],
  },
]

test('mergeServers keeps definition-only fields and lets live status win', () => {
  const merged = mergeServers(live, definitions)
  assert.equal(merged.length, 2)

  const fs = merged[0]
  assert.equal(fs.server_id, 'filesystem')
  assert.equal(fs.description, '只读文件访问')     // 仅 definition 提供
  assert.deepEqual(fs.capabilities, ['fs.read'])
  assert.equal(fs.timeout_s, 60)
  assert.equal(fs.running, true)                    // live 覆盖
  assert.equal(fs.tools.length, 1)

  const health = merged[1]
  assert.equal(health.running, false)
  assert.equal(serverState(health), 'error')
})

test('mergeServers normalizes shape and back-fills servers missing from live status', () => {
  const merged = mergeServers([], definitions)
  assert.equal(merged.length, 2)
  for (const server of merged) {
    assert.ok(Array.isArray(server.tools))
    assert.ok(Array.isArray(server.allowed_tools))
    assert.ok(Array.isArray(server.capabilities))
    assert.equal(server.timeout_s, Number(server.timeout_s))
  }
  // 未在 status 中出现 ⇒ 视为待命而不是消失
  assert.equal(serverState(merged[0]), 'standby')

  // 脏数据（无 server_id/name）直接丢弃
  assert.deepEqual(mergeServers([{ transport: 'sse' }], [{ description: 'x' }]), [])
})

test('serverState follows running > connecting > error > standby', () => {
  assert.equal(serverState({ running: true, status: 'error' }), 'running')
  assert.equal(serverState({ running: false, status: 'connecting' }), 'connecting')
  assert.equal(serverState({ running: false, status: 'disconnected', last_error: 'boom' }), 'error')
  assert.equal(serverState({ running: false, status: 'disconnected' }), 'standby')
  assert.equal(stateLabel('connecting'), '连接中')
  assert.equal(stateLabel('unknown'), '待命')
})

test('uptimeText renders seconds, minutes and hours from epoch seconds or millis', () => {
  const now = 1_700_003_600_000 // 与 started_at 相差 3600s
  assert.equal(uptimeText(1_700_000_000, now), '1 小时 0 分钟')
  assert.equal(uptimeText(1_700_003_540_000, now), '1 分钟')
  assert.equal(uptimeText(1_700_003_590_000, now), '10 秒')
  // 26 小时 ⇒ 1 天 2 小时（跨天只保留小时余数）
  assert.equal(uptimeText(1_700_000_000 - 93_600 + 3_600, now), '1 天 2 小时')
  assert.equal(uptimeText(null, now), '')
  assert.equal(uptimeText(0, now), '')
})

test('allowed tools summary and whitelist matching mirror backend semantics', () => {
  assert.deepEqual(allowedToolsSummary(['*']), { mode: 'all', count: 0, label: '全部工具放行' })
  assert.equal(allowedToolsSummary([]).mode, 'none')
  assert.equal(allowedToolsSummary(['a', 'b']).label, '白名单 2 个工具')
  assert.equal(isToolAllowed(['*'], 'anything'), true)
  assert.equal(isToolAllowed(['read_file'], 'read_file'), true)
  assert.equal(isToolAllowed([], 'read_file'), false)
})

test('list and command parsing tolerates the separators users actually type', () => {
  assert.deepEqual(parseList('a, b\nc；d，e'), ['a', 'b', 'c', 'd', 'e'])
  assert.deepEqual(parseList('   '), [])
  assert.deepEqual(parseCommandLine('  python -m app.tools.mcp.health_server '), ['python', '-m', 'app.tools.mcp.health_server'])
  assert.equal(formatCommand(['npx', '-y', 'server-filesystem']), 'npx -y server-filesystem')
})

test('env text round-trips KEY=VALUE lines and flags malformed input', () => {
  assert.deepEqual(parseEnvText('AMAP_MAPS_API_KEY=k-1\n# 注释\n\nDB_URL = postgres://x'), {
    AMAP_MAPS_API_KEY: 'k-1',
    DB_URL: 'postgres://x',
  })
  assert.deepEqual(parseEnvText(''), {})
  assert.equal(formatEnvText({ A: '1', B: '2' }), 'A=1\nB=2')
  assert.deepEqual(invalidEnvLines('A=1\nBAD LINE\n1X=2'), ['BAD LINE', '1X=2'])
})

test('stdio services carry env in create and update payloads', () => {
  const form = {
    ...emptyForm(),
    server_id: 'health',
    name: '健康检查',
    transport: 'stdio',
    stdio_command: 'python -m app.tools.mcp.health_server',
    env_text: 'AMAP_MAPS_API_KEY=k-9',
  }
  assert.deepEqual(formToPayload(form, { mode: 'create' }).env, { AMAP_MAPS_API_KEY: 'k-9' })
  // 编辑时也始终上送 env（空值 = 删除该变量）
  form.env_text = ''
  assert.deepEqual(formToPayload(form, { mode: 'edit' }).env, {})
  assert.equal(validateForm({ ...form, env_text: 'NOT A LINE' }, { mode: 'edit' }).env_text !== undefined, true)
  // 远程服务不带 env
  const remote = formToPayload({ ...emptyForm(), name: 'X', transport: 'sse', url: 'https://x/sse' }, { mode: 'create' })
  assert.equal('env' in remote, false)
})

// ── MCP 广场 ──────────────────────────────────────────────────────────────

const AMAP_PLAN = {
  catalog_id: '@amap/amap-maps',
  name: '高德地图',
  description: 'LBS 服务',
  kind: 'stdio',
  transport: 'stdio',
  stdio_command: ['npx', '-y', '@amap/amap-maps-mcp-server'],
  url: null,
  env_schema: ['AMAP_MAPS_API_KEY'],
  env_defaults: {},
  runtime: { binary: 'npx', trusted: true, available: true, hint: '需要 Node.js 与 npx' },
  server_id_suggestion: 'amap-amap-maps',
}

test('catalog server id matches the backend slug rules', () => {
  assert.equal(catalogServerId('@amap/amap-maps'), 'amap-amap-maps')
  assert.equal(catalogServerId('@modelcontextprotocol/fetch'), 'modelcontextprotocol-fetch')
  assert.equal(catalogServerId(''), 'mcp-server')
  assert.ok(/^[a-zA-Z0-9_-]+$/.test(catalogServerId('@A/B.c_d')))
})

test('catalog heat formatting keeps small numbers readable', () => {
  assert.equal(formatCount(0), '0')
  assert.equal(formatCount(999), '999')
  assert.equal(formatCount(12345), '1.2万')
  assert.equal(formatCount(10000), '1万')
})

test('install form prefills from the plan and validates required env', () => {
  const form = planToInstallForm(AMAP_PLAN)
  assert.equal(form.server_id, 'amap-amap-maps')
  assert.equal(form.kind, 'stdio')
  assert.deepEqual(form.env, { AMAP_MAPS_API_KEY: '' })
  assert.deepEqual(form.stdio_command, ['npx', '-y', '@amap/amap-maps-mcp-server'])

  const errors = validateInstallForm(form, AMAP_PLAN)
  assert.ok(errors['env.AMAP_MAPS_API_KEY'])

  form.env.AMAP_MAPS_API_KEY = 'k-1'
  assert.deepEqual(validateInstallForm(form, AMAP_PLAN), {})

  const payload = installFormToPayload(form)
  assert.deepEqual(payload.env, { AMAP_MAPS_API_KEY: 'k-1' })
  assert.equal(payload.transport, undefined) // stdio 由计划的命令决定
  assert.deepEqual(payload.allowed_tools, ['*'])
  assert.equal(payload.overwrite, false)
})

test('install form blocks deploy-required services without a url and warns on missing runtime', () => {
  const emptyPlan = { catalog_id: 'a/b', name: '无配置服务', kind: 'deploy_required', env_schema: [] }
  const form = planToInstallForm(emptyPlan)
  assert.ok(validateInstallForm(form, emptyPlan).url)
  form.url = 'https://mcp.example.com/mcp'
  assert.deepEqual(validateInstallForm(form, emptyPlan), {})
  assert.equal(installFormToPayload(form).url, 'https://mcp.example.com/mcp')
  assert.equal(installFormToPayload(form).transport, 'streamable_http')

  const uvxPlan = {
    catalog_id: '@modelcontextprotocol/fetch',
    name: 'fetch',
    kind: 'stdio',
    transport: 'stdio',
    stdio_command: ['uvx', 'mcp-server-fetch'],
    env_schema: [],
    runtime: { binary: 'uvx', trusted: true, available: false, hint: '镜像未内置 uv' },
  }
  const uvxForm = planToInstallForm(uvxPlan)
  // 运行环境缺失只提示不拦截（用户可能自行补齐 uv），受信任校验才拦截
  assert.equal(validateInstallForm(uvxForm, uvxPlan).runtime, undefined)
  assert.match(uvxPlan.runtime.hint, /uv/)
  assert.match(validateInstallForm(uvxForm, { ...uvxPlan, runtime: { binary: 'bash', trusted: false } }).runtime, /bash/)
  uvxForm.scope = 'custom'
  assert.ok(validateInstallForm(uvxForm, uvxPlan).allowed_tools)
})

test('trusted binary check matches the backend stdio whitelist', () => {
  assert.equal(isTrustedBinary(['python', '-m', 'x']), true)
  assert.equal(isTrustedBinary(['C:\\Python311\\python.exe', '-m', 'x']), true)
  assert.equal(isTrustedBinary(['/usr/bin/node', 'x.js']), true)
  assert.equal(isTrustedBinary(['bash', '-c', 'rm -rf /']), false)
  assert.equal(isTrustedBinary([]), false)
})

test('validateForm reports field errors for create mode', () => {
  const form = { ...emptyForm(), server_id: 'bad id!', name: '', transport: 'sse', url: 'ftp://x', timeout_s: 999 }
  const errors = validateForm(form, { mode: 'create' })
  assert.ok(errors.server_id)
  assert.ok(errors.name)
  assert.ok(errors.url)
  assert.ok(errors.timeout_s)

  const stdio = { ...emptyForm(), server_id: 'ok_id', name: 'OK', transport: 'stdio', stdio_command: 'bash -c hi' }
  assert.ok(validateForm(stdio, { mode: 'create' }).stdio_command)

  const good = {
    ...emptyForm(),
    server_id: 'github',
    name: 'GitHub',
    transport: 'stdio',
    stdio_command: 'python -m app.tools.mcp.health_server',
    scope: 'custom',
    allowed_tools: 'check_redis',
  }
  assert.deepEqual(validateForm(good, { mode: 'create' }), {})
})

test('validateForm enforces an explicit whitelist and skips immutable fields on edit', () => {
  const custom = { ...emptyForm(), name: 'X', transport: 'sse', url: 'https://mcp.demo/sse', scope: 'custom', allowed_tools: ' , ' }
  assert.ok(validateForm(custom, { mode: 'edit' }).allowed_tools)

  const edit = { ...emptyForm(), server_id: '', name: 'X', transport: 'sse', url: 'https://mcp.demo/sse' }
  assert.deepEqual(validateForm(edit, { mode: 'edit' }), {})
})

test('formToPayload builds create payloads for both transports', () => {
  const stdioForm = {
    ...emptyForm(),
    server_id: 'health',
    name: '健康检查',
    description: '  依赖探活  ',
    transport: 'stdio',
    stdio_command: 'python -m app.tools.mcp.health_server',
    timeout_s: '45',
    scope: 'custom',
    allowed_tools: 'check_redis, check_postgres',
    capabilities: 'ops.monitor',
  }
  const payload = formToPayload(stdioForm, { mode: 'create' })
  assert.deepEqual(payload, {
    server_id: 'health',
    name: '健康检查',
    description: '依赖探活',
    transport: 'stdio',
    stdio_command: ['python', '-m', 'app.tools.mcp.health_server'],
    env: {},
    timeout_s: 45,
    enabled: true,
    allowed_tools: ['check_redis', 'check_postgres'],
    capabilities: ['ops.monitor'],
  })

  const remoteForm = { ...emptyForm(), server_id: 'crm', name: 'CRM', transport: 'sse', url: ' https://mcp.demo/sse ', auth_token: ' token ' }
  const remote = formToPayload(remoteForm, { mode: 'create' })
  assert.equal(remote.url, 'https://mcp.demo/sse')
  assert.equal(remote.auth_token, 'token')
  assert.deepEqual(remote.allowed_tools, ['*'])
  assert.equal('stdio_command' in remote, false)
})

test('formToPayload only sends PUT-mutable fields when editing', () => {
  const form = {
    ...emptyForm(),
    server_id: 'filesystem',
    name: '文件系统',
    transport: 'stdio',
    stdio_command: 'npx -y server-filesystem ./volumes',
    enabled: false,
    scope: 'custom',
    allowed_tools: 'read_file',
  }
  const payload = formToPayload(form, { mode: 'edit' })
  assert.equal('server_id' in payload, false)
  assert.equal('transport' in payload, false)
  assert.equal('stdio_command' in payload, false)
  assert.equal('url' in payload, false)
  assert.equal(payload.enabled, false)
  assert.deepEqual(payload.allowed_tools, ['read_file'])
})

test('serverToForm round-trips a definition back into the editor', () => {
  const form = serverToForm(definitions[0])
  assert.equal(form.server_id, 'filesystem')
  assert.equal(form.scope, 'custom')
  assert.equal(form.allowed_tools, 'read_file, list_directory')
  assert.equal(form.capabilities, 'fs.read')
  assert.equal(form.stdio_command, 'npx -y @modelcontextprotocol/server-filesystem ./volumes')
  assert.deepEqual(formToPayload(form, { mode: 'edit' }).allowed_tools, ['read_file', 'list_directory'])

  const wildcard = serverToForm(definitions[1])
  assert.equal(wildcard.scope, 'all')
  assert.equal(wildcard.allowed_tools, '')
})

test('filterServers narrows by state and by keyword', () => {
  const merged = mergeServers(live, definitions)
  assert.equal(filterServers(merged, { filter: 'running' }).length, 1)
  assert.equal(filterServers(merged, { filter: 'error' }).length, 1)
  assert.equal(filterServers(merged, { filter: 'standby' }).length, 0)
  assert.equal(filterServers(merged, { query: 'FS.READ' }).length, 1)
  assert.equal(filterServers(merged, { query: 'read_file' }).length, 1)
  assert.equal(filterServers(merged, { query: '不存在的服务' }).length, 0)
  assert.equal(filterServers(merged, {}).length, 2)
})

test('connecting services count as running in both tabs and counters', () => {
  const connecting = { server_id: 'a', name: 'a', status: 'connecting', running: false }
  assert.equal(serverMatchesFilter(connecting, 'running'), true)
  assert.equal(serverMatchesFilter(connecting, 'standby'), false)
  // 待命 ≠ 连接中，避免 tab 数字与列表不一致
  const counts = filterCounts([connecting, { server_id: 'b', status: 'disconnected' }, { server_id: 'c', status: 'error' }])
  assert.deepEqual(counts, { all: 3, running: 1, standby: 1, error: 1 })
})

test('serverMetrics aggregates states, tools and whitelist usage', () => {
  const metrics = serverMetrics(mergeServers(live, definitions))
  assert.deepEqual(metrics, { total: 2, running: 1, connecting: 0, error: 1, standby: 0, tools: 1, whitelisted: 1 })
})

test('tool catalog indexes namespaced tools for the expandable detail view', () => {
  const catalog = buildToolCatalog([
    { namespaced_name: 'filesystem__read_file', raw_name: 'read_file', server_id: 'filesystem', description: '读取文件' },
  ])
  assert.equal(catalog.filesystem__read_file.description, '读取文件')
  assert.equal(toolNamespacedName('filesystem', { name: 'read_file' }), 'filesystem__read_file')
  assert.equal(toolNamespacedName('filesystem', { name: 'read_file', namespaced_name: 'custom__read_file' }), 'custom__read_file')
})
