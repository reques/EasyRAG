<template>
  <div class="mcp-page">
    <header class="mcp-header">
      <div>
        <span class="mcp-eyebrow">MCP SERVICE CONSOLE</span>
        <h1>MCP 服务管理</h1>
        <p>挂载远程 SSE / Streamable HTTP 服务或本地受信任 Stdio 能力，工具会以 <code>{服务}__{工具}</code> 命名空间注入 Agent，并受白名单与沙箱策略约束。</p>
      </div>
      <div class="mcp-header-actions">
        <button class="mcp-button" :disabled="loading" @click="fetchServers">
          <RefreshCw :size="15" :class="{ spin: loading }" /> 刷新
        </button>
        <button class="mcp-button" @click="catalogOpen = true">
          <Store :size="15" /> MCP 广场
        </button>
        <button class="mcp-button is-primary" @click="openCreate">
          <Plus :size="15" /> 添加服务
        </button>
      </div>
    </header>

    <section class="mcp-overview" aria-label="MCP 概览">
      <article>
        <i class="mcp-metric-icon"><Blocks :size="17" /></i>
        <div><small>已配置服务</small><strong>{{ metrics.total }}</strong></div>
      </article>
      <article>
        <i class="mcp-metric-icon"><PlugZap :size="17" /></i>
        <div><small>运行中</small><strong>{{ metrics.running }}</strong></div>
      </article>
      <article>
        <i class="mcp-metric-icon"><PowerOff :size="17" /></i>
        <div><small>待命 / 异常</small><strong>{{ metrics.standby + metrics.error }}</strong></div>
      </article>
      <article>
        <i class="mcp-metric-icon"><Wrench :size="17" /></i>
        <div><small>已注入工具</small><strong>{{ metrics.tools }}</strong></div>
      </article>
    </section>

    <p v-if="metrics.error" class="mcp-banner is-warn">
      <CircleAlert :size="15" />
      <span>{{ metrics.error }} 个服务连接异常，展开卡片可查看具体原因；修复后点击卡片上的「连接」重试。</span>
    </p>

    <section class="mcp-section">
      <div class="mcp-toolbar">
        <div class="mcp-toolbar-copy">
          <h2>服务列表</h2>
          <p v-if="!loading && servers.length">
            共 {{ metrics.total }} 个服务，其中 {{ metrics.whitelisted }} 个启用了工具白名单。
          </p>
          <p v-else>持久化在 config/mcp_servers.json，改动即时热挂载并从下一次工具调用生效。</p>
        </div>
        <div v-if="servers.length" class="mcp-toolbar-actions">
          <div class="mcp-filter" role="group" aria-label="按状态筛选">
            <button
              v-for="option in FILTERS"
              :key="option.value"
              type="button"
              :class="['mcp-filter-item', { 'is-active': filter === option.value }]"
              :aria-pressed="filter === option.value"
              @click="filter = option.value"
            >
              {{ option.label }}<span v-if="filterCounts[option.value]">{{ filterCounts[option.value] }}</span>
            </button>
          </div>
          <label class="mcp-search">
            <Search :size="14" />
            <input v-model="query" type="search" placeholder="搜索服务或工具" aria-label="搜索 MCP 服务" />
            <button v-if="query" type="button" aria-label="清空搜索" @click="query = ''"><X :size="13" /></button>
          </label>
        </div>
      </div>

      <div v-if="loading && !servers.length" class="mcp-state">
        <LoaderCircle :size="22" class="spin" />
        <strong>正在读取 MCP 服务配置</strong>
        <span>同时拉取每个服务的连接状态与已注册工具。</span>
      </div>

      <div v-else-if="error" class="mcp-state is-error">
        <CircleAlert :size="22" />
        <strong>配置加载失败</strong>
        <span>{{ error }}</span>
        <button class="mcp-button" @click="fetchServers">重新加载</button>
      </div>

      <div v-else-if="!servers.length" class="mcp-state">
        <Blocks :size="24" />
        <strong>还没有配置任何 MCP 服务</strong>
        <span>从 MCP 广场一键安装现成服务（地图、数据库、网页抓取…），或手动接入自己的 SSE / Stdio 服务。</span>
        <div class="mcp-state-actions">
          <button class="mcp-button is-primary" @click="catalogOpen = true"><Store :size="14" /> 浏览 MCP 广场</button>
          <button class="mcp-button" @click="openCreate"><Plus :size="14" /> 手动添加</button>
        </div>
      </div>

      <div v-else-if="!visibleServers.length" class="mcp-state">
        <SearchX :size="22" />
        <strong>没有匹配的服务</strong>
        <span>换一个关键词，或切换到「全部」查看所有服务。</span>
        <button class="mcp-button" @click="resetFilters">清空筛选</button>
      </div>

      <div v-else class="mcp-grid">
        <article v-for="srv in visibleServers" :key="srv.server_id" class="mcp-card">
          <header class="mcp-card-header">
            <i class="mcp-card-icon" :class="`is-${stateOf(srv)}`"><Blocks :size="17" /></i>
            <div class="mcp-card-title">
              <strong>{{ srv.name }}</strong>
              <small>{{ srv.server_id }}</small>
            </div>
            <div class="mcp-card-badges">
              <span :class="['mcp-state-badge', `is-${stateOf(srv)}`]">
                <i></i>{{ stateLabel(stateOf(srv)) }}
              </span>
              <span class="mcp-transport-badge">{{ TRANSPORT_LABELS[srv.transport] || srv.transport }}</span>
            </div>
          </header>

          <div class="mcp-card-body">
            <p class="mcp-card-desc">{{ srv.description || '未填写用途描述。' }}</p>

            <p v-if="serverError(srv)" class="mcp-banner is-error">
              <CircleAlert :size="14" /> <span>{{ serverError(srv) }}</span>
            </p>

            <dl class="mcp-card-meta">
              <div v-if="srv.transport === 'stdio'">
                <dt>启动命令</dt>
                <dd class="mono">{{ formatCommand(srv.stdio_command) || '—' }}</dd>
              </div>
              <div v-else>
                <dt>端点</dt>
                <dd class="mono">{{ srv.url || '—' }}</dd>
              </div>
              <div>
                <dt>调用超时</dt>
                <dd>{{ srv.timeout_s }} 秒</dd>
              </div>
              <div>
                <dt>启动策略</dt>
                <dd>{{ srv.enabled ? '随应用自动连接' : '仅手动连接' }}</dd>
              </div>
              <div v-if="stateOf(srv) === 'running' && uptimeText(srv.started_at)">
                <dt>已运行</dt>
                <dd>{{ uptimeText(srv.started_at) }}</dd>
              </div>
              <div v-if="srv.capabilities.length">
                <dt>能力标签</dt>
                <dd class="mcp-chips">
                  <span v-for="cap in srv.capabilities" :key="cap" class="mcp-chip">{{ cap }}</span>
                </dd>
              </div>
              <div v-if="srv.source">
                <dt>来源</dt>
                <dd>
                  <a
                    v-if="srv.source.startsWith('modelscope:')"
                    class="mcp-source-tag"
                    href="https://www.modelscope.cn/mcp"
                    target="_blank"
                    rel="noopener"
                  >
                    <Store :size="11" /> MCP 广场 · {{ srv.source.slice('modelscope:'.length) }}
                  </a>
                  <span v-else class="mcp-source-tag"><Store :size="11" /> {{ srv.source }}</span>
                </dd>
              </div>
            </dl>

            <div class="mcp-scope-row">
              <span :class="['mcp-scope-tag', `is-${allowedToolsSummary(srv.allowed_tools).mode}`]">
                <ShieldCheck v-if="allowedToolsSummary(srv.allowed_tools).mode !== 'all'" :size="12" />
                <ShieldAlert v-else :size="12" />
                {{ allowedToolsSummary(srv.allowed_tools).label }}
              </span>
              <span v-if="allowedToolsSummary(srv.allowed_tools).mode === 'custom'" class="mcp-scope-list">
                {{ srv.allowed_tools.join(' · ') }}
              </span>
            </div>

            <div class="mcp-tools">
              <button
                type="button"
                class="mcp-tools-toggle"
                :aria-expanded="Boolean(expanded[srv.server_id])"
                @click="toggleTools(srv)"
              >
                <Wrench :size="13" />
                <span>{{ srv.tools.length ? `${srv.tools.length} 个工具已注入` : '工具清单' }}</span>
                <ChevronDown :size="14" :class="{ 'is-open': expanded[srv.server_id] }" />
              </button>

              <div v-if="expanded[srv.server_id]" class="mcp-tools-drawer">
                <p v-if="toolLoading[srv.server_id]" class="mcp-tools-hint">
                  <LoaderCircle :size="12" class="spin" /> 正在读取工具定义…
                </p>
                <p v-else-if="!srv.tools.length" class="mcp-tools-hint">
                  {{ srv.running ? '该服务未暴露任何工具，或全部被白名单拦截。' : '服务待命，连接成功后会自动同步工具。' }}
                </p>
                <ul v-else class="mcp-tool-list">
                  <li v-for="tool in srv.tools" :key="toolNamespacedName(srv.server_id, tool)">
                    <div class="mcp-tool-head">
                      <code>{{ toolNamespacedName(srv.server_id, tool) }}</code>
                      <span :class="['mcp-tool-flag', { 'is-allowed': toolAllowed(srv, tool) }]">
                        {{ toolAllowed(srv, tool) ? '已放行' : '被白名单拦截' }}
                      </span>
                    </div>
                    <small v-if="toolDescription(srv.server_id, tool)">{{ toolDescription(srv.server_id, tool) }}</small>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <footer class="mcp-card-footer">
            <button
              v-if="!srv.running"
              type="button"
              class="mcp-action"
              :disabled="busy[srv.server_id]"
              @click="runAction(srv, 'start')"
            >
              <Play :size="13" /> 连接
            </button>
            <button
              v-else
              type="button"
              class="mcp-action"
              :disabled="busy[srv.server_id]"
              @click="runAction(srv, 'stop')"
            >
              <Square :size="13" /> 断开
            </button>
            <button
              v-if="srv.running"
              type="button"
              class="mcp-action"
              :disabled="busy[srv.server_id]"
              @click="runAction(srv, 'sync')"
            >
              <RefreshCw :size="13" :class="{ spin: busy[srv.server_id] }" /> 同步工具
            </button>
            <span class="mcp-card-footer-gap"></span>
            <button type="button" class="mcp-icon-action" title="编辑配置" aria-label="编辑配置" @click="openEdit(srv)">
              <Pencil :size="14" />
            </button>
            <button type="button" class="mcp-icon-action is-danger" title="删除服务" aria-label="删除服务" @click="deleteTarget = srv">
              <Trash2 :size="14" />
            </button>
          </footer>
        </article>
      </div>
    </section>
  </div>

  <McpServerDialog
    v-if="dialog.open"
    :server="dialog.server"
    :submitting="dialog.submitting"
    :error="dialog.error"
    @close="closeDialog"
    @submit="submitDialog"
  />

  <McpCatalogDialog v-if="catalogOpen" @close="catalogOpen = false" @installed="onInstalled" />

  <Teleport to="body">
    <div v-if="deleteTarget" class="mcp-dialog-overlay" @click.self="deleteTarget = null">
      <div class="mcp-confirm" role="dialog" aria-modal="true" aria-label="删除 MCP 服务">
        <h3>删除 MCP 服务</h3>
        <p>
          确定删除「<strong>{{ deleteTarget.name }}</strong>」（{{ deleteTarget.server_id }}）吗？
          {{ deleteTarget.running ? '当前连接会被断开，' : '' }}其注入 Agent 的工具将全部注销，配置文件同步更新。
        </p>
        <small>如需临时停用，建议改为「断开」而不是删除。</small>
        <div class="mcp-confirm-actions">
          <button type="button" class="mcp-button" @click="deleteTarget = null">取消</button>
          <button type="button" class="mcp-button is-danger" :disabled="deleting" @click="confirmDelete">
            <LoaderCircle v-if="deleting" :size="14" class="spin" />
            {{ deleting ? '删除中…' : '确认删除' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>

  <Teleport to="body">
    <div class="mcp-toasts" aria-live="polite">
      <p v-for="notice in notices" :key="notice.id" :class="['mcp-toast', `is-${notice.tone}`]">
        <CircleAlert v-if="notice.tone === 'error'" :size="14" />
        <CheckCircle2 v-else :size="14" />
        <span>{{ notice.text }}</span>
      </p>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, onActivated, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import {
  Blocks,
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  LoaderCircle,
  Pencil,
  Play,
  PlugZap,
  Plus,
  PowerOff,
  RefreshCw,
  Search,
  SearchX,
  ShieldAlert,
  ShieldCheck,
  Square,
  Store,
  Trash2,
  Wrench,
  X,
} from 'lucide-vue-next'
import api from '../api'
import McpCatalogDialog from '../components/McpCatalogDialog.vue'
import McpServerDialog from '../components/McpServerDialog.vue'
import {
  FILTERS,
  TRANSPORT_LABELS,
  allowedToolsSummary,
  buildToolCatalog,
  filterCounts as countByFilter,
  filterServers,
  formatCommand,
  isToolAllowed,
  mergeServers,
  serverError,
  serverMetrics,
  serverState,
  stateLabel,
  toolNamespacedName,
  uptimeText,
} from '../utils/mcp-servers'

const servers = ref([])
const loading = ref(false)
const error = ref('')
const query = ref('')
const filter = ref('all')
const expanded = reactive({})
const busy = reactive({})
const toolLoading = reactive({})
const toolCatalog = ref({})
const deleteTarget = ref(null)
const deleting = ref(false)
const notices = ref([])

const dialog = reactive({ open: false, server: null, submitting: false, error: '' })
const catalogOpen = ref(false)

const metrics = computed(() => serverMetrics(servers.value))
const visibleServers = computed(() => filterServers(servers.value, { query: query.value, filter: filter.value }))
const filterCounts = computed(() => countByFilter(servers.value))

const stateOf = serverState
const toolAllowed = (server, tool) => isToolAllowed(server.allowed_tools, tool.name)

function errorDetail(err) {
  const detail = err?.response?.data?.detail
  if (Array.isArray(detail)) return detail.map(item => item.msg || JSON.stringify(item)).join('；')
  return detail || err?.message || '未知错误'
}

let noticeSeed = 0
function notify(text, tone = 'success') {
  const id = ++noticeSeed
  notices.value = [...notices.value, { id, text, tone }]
  setTimeout(() => { notices.value = notices.value.filter(item => item.id !== id) }, tone === 'error' ? 6000 : 3200)
}

async function fetchServers() {
  loading.value = true
  error.value = ''
  try {
    const res = await api.get('/mcp/servers')
    servers.value = mergeServers(res?.servers || [], res?.definitions || [])
  } catch (err) {
    error.value = errorDetail(err)
  } finally {
    loading.value = false
  }
}

async function runAction(server, action) {
  if (busy[server.server_id]) return
  busy[server.server_id] = true
  try {
    await api.post(`/mcp/servers/${encodeURIComponent(server.server_id)}/${action}`)
    await fetchServers()
    const label = { start: '已建立连接', stop: '已断开连接', sync: '工具列表已同步' }[action]
    notify(`${server.name}：${label}`)
  } catch (err) {
    notify(`${server.name}：${errorDetail(err)}`, 'error')
  } finally {
    delete busy[server.server_id]
  }
}

async function toggleTools(server) {
  if (expanded[server.server_id]) {
    expanded[server.server_id] = false
    return
  }
  expanded[server.server_id] = true
  if (!server.running || toolLoading[server.server_id]) return
  const unknown = server.tools.some(tool => !toolCatalog.value[toolNamespacedName(server.server_id, tool)])
  if (!unknown) return
  // 状态接口只给工具名，描述与参数 schema 需要走 /mcp/tools 目录接口。
  toolLoading[server.server_id] = true
  try {
    const infos = await api.get('/mcp/tools', { server_ids: server.server_id })
    toolCatalog.value = { ...toolCatalog.value, ...buildToolCatalog(Array.isArray(infos) ? infos : []) }
  } catch { /* 目录接口失败时只展示工具名，不阻断抽屉 */ }
  finally {
    delete toolLoading[server.server_id]
  }
}

function toolDescription(serverId, tool) {
  return toolCatalog.value[toolNamespacedName(serverId, tool)]?.description || ''
}

function resetFilters() {
  query.value = ''
  filter.value = 'all'
}

function openCreate() {
  dialog.open = true
  dialog.server = null
  dialog.error = ''
}

function openEdit(server) {
  dialog.open = true
  dialog.server = server
  dialog.error = ''
}

function closeDialog() {
  dialog.open = false
  dialog.error = ''
}

async function submitDialog(payload) {
  dialog.submitting = true
  dialog.error = ''
  const editing = Boolean(dialog.server)
  try {
    if (editing) await api.put(`/mcp/servers/${encodeURIComponent(dialog.server.server_id)}`, payload)
    else await api.post('/mcp/servers', payload)
    dialog.open = false
    await fetchServers()
    notify(editing ? `已保存 ${payload.name}` : `已挂载 ${payload.name}`)
  } catch (err) {
    dialog.error = errorDetail(err)
  } finally {
    dialog.submitting = false
  }
}

async function confirmDelete() {
  if (!deleteTarget.value || deleting.value) return
  const target = deleteTarget.value
  deleting.value = true
  try {
    await api.delete(`/mcp/servers/${encodeURIComponent(target.server_id)}`)
    deleteTarget.value = null
    await fetchServers()
    notify(`已删除 ${target.name}`)
  } catch (err) {
    notify(`删除失败：${errorDetail(err)}`, 'error')
  } finally {
    deleting.value = false
  }
}

/** 广场安装成功：收起广场、刷新列表，让新服务带着连接状态出现在卡片里。 */
async function onInstalled(definition) {
  catalogOpen.value = false
  await fetchServers()
  const label = definition?.name || definition?.server_id || '服务'
  notify(`已从广场安装 ${label}，正在建立连接…`)
}

// 页面被 LayoutView 的 keep-alive 缓存：首次挂载拉取一次，之后每次切回 MCP 页再同步连接状态。
let firstActivation = true
onMounted(fetchServers)
onActivated(() => {
  if (firstActivation) {
    firstActivation = false
    return // 首次进入由 onMounted 负责，避免重复请求
  }
  fetchServers()
})
onBeforeUnmount(() => { notices.value = [] })
</script>

<style scoped>
.mcp-page {
  height: 100%;
  overflow: auto;
  scrollbar-gutter: stable;
  padding: 34px 42px 60px;
  background: var(--gray-25);
}

.mcp-header {
  max-width: 1240px;
  margin: 0 auto 22px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
}
.mcp-eyebrow {
  display: block;
  color: var(--gray-500);
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .13em;
  margin-bottom: 7px;
}
.mcp-header h1 { font-size: 27px; line-height: 1.25; color: var(--gray-1000); margin-bottom: 7px; }
.mcp-header p { color: var(--gray-600); font-size: 13px; max-width: 720px; }
.mcp-header p code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--gray-100);
  border-radius: 4px;
  padding: 1px 5px;
}
.mcp-header-actions { display: flex; gap: 9px; flex: 0 0 auto; }

.mcp-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 7px 13px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: var(--gray-700);
  font-size: 13px;
  font-weight: 550;
}
.mcp-button:hover:not(:disabled) { background: var(--gray-50); border-color: var(--gray-200); color: var(--gray-900); }
.mcp-button.is-primary { background: var(--main-700); border-color: var(--main-700); color: var(--gray-0); }
.mcp-button.is-primary:hover:not(:disabled) { background: var(--main-600); border-color: var(--main-600); color: var(--gray-0); }
.mcp-button.is-danger { background: var(--color-error-700); border-color: var(--color-error-700); color: var(--gray-0); }
.mcp-button.is-danger:hover:not(:disabled) { background: var(--color-error-600); border-color: var(--color-error-600); color: var(--gray-0); }
.mcp-button:disabled { opacity: .5; cursor: not-allowed; }
.mcp-page button:focus-visible,
.mcp-page input:focus-visible { outline: 2px solid var(--main-500); outline-offset: 2px; }

.mcp-overview {
  max-width: 1240px;
  margin: 0 auto 18px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.mcp-overview article {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 15px 16px;
  background: var(--gray-0);
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-lg);
}
.mcp-metric-icon {
  width: 36px;
  height: 36px;
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  border-radius: var(--radius-md);
  background: var(--gray-50);
  color: var(--gray-700);
}
.mcp-overview small { display: block; color: var(--gray-500); font-size: 11px; }
.mcp-overview strong { font-size: 21px; line-height: 1.15; color: var(--gray-1000); }

.mcp-banner {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  padding: 9px 12px;
  border-radius: var(--radius-md);
  font-size: 12px;
  line-height: 1.5;
}
.mcp-banner svg { flex: 0 0 auto; margin-top: 2px; }
.mcp-banner.is-warn { max-width: 1240px; margin: 0 auto 16px; background: var(--color-warning-50); color: var(--color-warning-900); }
.mcp-banner.is-error { background: var(--color-error-50); color: var(--color-error-700); }

.mcp-section {
  max-width: 1240px;
  margin: 0 auto;
}
.mcp-toolbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.mcp-toolbar-copy h2 { font-size: 16px; color: var(--gray-1000); }
.mcp-toolbar-copy p { color: var(--gray-600); font-size: 12px; margin-top: 2px; }
.mcp-toolbar-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

.mcp-filter {
  display: inline-flex;
  padding: 2px;
  gap: 2px;
  background: var(--gray-100);
  border-radius: var(--radius-md);
}
.mcp-filter-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 11px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--gray-600);
  font-size: 12px;
}
.mcp-filter-item span {
  min-width: 16px;
  padding: 0 4px;
  border-radius: var(--radius-full);
  background: var(--gray-200);
  color: var(--gray-700);
  font-size: 10px;
  line-height: 15px;
  text-align: center;
}
.mcp-filter-item.is-active { background: var(--gray-0); color: var(--gray-1000); font-weight: 600; }
.mcp-filter-item.is-active span { background: var(--main-100); color: var(--main-900); }

.mcp-search {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 240px;
  height: 34px;
  padding: 0 9px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: var(--gray-500);
}
.mcp-search:focus-within { border-color: var(--gray-400); }
.mcp-search input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; color: var(--gray-900); font-size: 13px; }
.mcp-search button { display: grid; place-items: center; border: 0; background: transparent; color: var(--gray-500); }
.mcp-search button:hover { color: var(--gray-900); }

.mcp-state {
  min-height: 260px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 30px;
  background: var(--gray-0);
  border: 1px dashed var(--gray-150);
  border-radius: var(--radius-lg);
  color: var(--gray-500);
  text-align: center;
}
.mcp-state strong { color: var(--gray-800); font-size: 14px; }
.mcp-state span { font-size: 12px; max-width: 420px; }
.mcp-state .mcp-button { margin-top: 6px; }
.mcp-state.is-error strong { color: var(--color-error-700); }

.mcp-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(370px, 1fr));
  gap: 14px;
}
.mcp-card {
  display: flex;
  flex-direction: column;
  background: var(--gray-0);
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-lg);
  padding: 16px;
  transition: border-color .16s, box-shadow .16s;
}
.mcp-card:hover { border-color: var(--gray-200); box-shadow: var(--shadow-card); }

.mcp-card-header { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.mcp-card-icon {
  width: 34px;
  height: 34px;
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  border-radius: var(--radius-md);
  background: var(--gray-50);
  color: var(--gray-600);
}
.mcp-card-icon.is-running { background: var(--color-success-50); color: var(--color-success-700); }
.mcp-card-icon.is-error { background: var(--color-error-50); color: var(--color-error-700); }
.mcp-card-icon.is-connecting { background: var(--color-info-50); color: var(--color-info-700); }
.mcp-card-title { flex: 1; min-width: 0; }
.mcp-card-title strong { display: block; font-size: 14px; color: var(--gray-1000); overflow-wrap: anywhere; }
.mcp-card-title small { display: block; margin-top: 1px; color: var(--gray-500); font-size: 11px; font-family: var(--font-mono); }
.mcp-card-badges { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; flex: 0 0 auto; }

.mcp-state-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px;
  border-radius: var(--radius-full);
  background: var(--gray-100);
  color: var(--gray-600);
  font-size: 11px;
  font-weight: 550;
  white-space: nowrap;
}
.mcp-state-badge i { width: 6px; height: 6px; border-radius: 50%; background: var(--gray-400); }
.mcp-state-badge.is-running { background: var(--color-success-50); color: var(--color-success-700); }
.mcp-state-badge.is-running i { background: var(--color-success-700); }
.mcp-state-badge.is-connecting { background: var(--color-info-50); color: var(--color-info-700); }
.mcp-state-badge.is-connecting i { background: var(--color-info-700); }
.mcp-state-badge.is-error { background: var(--color-error-50); color: var(--color-error-700); }
.mcp-state-badge.is-error i { background: var(--color-error-500); }
.mcp-transport-badge {
  padding: 2px 7px;
  border-radius: var(--radius-sm);
  background: var(--gray-50);
  border: 1px solid var(--gray-150);
  color: var(--gray-600);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .02em;
}

.mcp-card-body { flex: 1; display: flex; flex-direction: column; gap: 10px; }
.mcp-card-desc { color: var(--gray-600); font-size: 12px; line-height: 1.55; }

.mcp-card-meta {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 10px 11px;
  background: var(--gray-25);
  border: 1px solid var(--gray-100);
  border-radius: var(--radius-md);
  font-size: 12px;
}
.mcp-card-meta > div { display: flex; gap: 8px; align-items: flex-start; }
.mcp-card-meta dt { flex: 0 0 62px; color: var(--gray-500); font-size: 11px; line-height: 1.6; }
.mcp-card-meta dd { flex: 1; min-width: 0; color: var(--gray-800); overflow-wrap: anywhere; line-height: 1.6; }
.mcp-card-meta dd.mono { font-family: var(--font-mono); font-size: 11px; }
.mcp-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.mcp-chip {
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  background: var(--gray-100);
  color: var(--gray-700);
  font-size: 11px;
}

.mcp-scope-row { display: flex; flex-direction: column; gap: 4px; }
.mcp-scope-tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  align-self: flex-start;
  padding: 2px 8px;
  border-radius: var(--radius-full);
  background: var(--gray-100);
  color: var(--gray-600);
  font-size: 11px;
  font-weight: 550;
}
.mcp-scope-tag.is-all { background: var(--color-warning-50); color: var(--color-warning-900); }
.mcp-scope-tag.is-custom { background: var(--color-success-50); color: var(--color-success-700); }
.mcp-scope-tag.is-none { background: var(--color-error-50); color: var(--color-error-700); }
.mcp-scope-list { color: var(--gray-500); font-size: 11px; font-family: var(--font-mono); overflow-wrap: anywhere; }

.mcp-source-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 7px;
  border-radius: var(--radius-full);
  background: var(--main-50);
  color: var(--main-700);
  font-size: 11px;
  text-decoration: none;
}
a.mcp-source-tag:hover { background: var(--main-100); }
.mcp-state-actions { display: flex; gap: 8px; margin-top: 6px; }

.mcp-tools { margin-top: auto; border-top: 1px dashed var(--gray-150); padding-top: 9px; }
.mcp-tools-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  border: 0;
  background: transparent;
  color: var(--gray-700);
  font-size: 12px;
  font-weight: 550;
}
.mcp-tools-toggle span { flex: 1; text-align: left; }
.mcp-tools-toggle svg:last-child { transition: transform .16s; }
.mcp-tools-toggle svg.is-open { transform: rotate(180deg); }
.mcp-tools-drawer {
  margin-top: 8px;
  padding: 9px 10px;
  background: var(--gray-25);
  border: 1px solid var(--gray-100);
  border-radius: var(--radius-md);
  max-height: 210px;
  overflow: auto;
}
.mcp-tools-hint { display: flex; align-items: center; gap: 6px; color: var(--gray-500); font-size: 11px; }
.mcp-tool-list { display: flex; flex-direction: column; gap: 8px; list-style: none; }
.mcp-tool-list li + li { border-top: 1px solid var(--gray-100); padding-top: 8px; }
.mcp-tool-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.mcp-tool-head code { font-family: var(--font-mono); font-size: 11px; color: var(--gray-800); overflow-wrap: anywhere; }
.mcp-tool-flag { flex: 0 0 auto; color: var(--color-error-700); font-size: 10px; }
.mcp-tool-flag.is-allowed { color: var(--color-success-700); }
.mcp-tool-list small { display: block; margin-top: 3px; color: var(--gray-600); font-size: 11px; line-height: 1.5; }

.mcp-card-footer {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  padding-top: 11px;
  border-top: 1px solid var(--gray-100);
}
.mcp-card-footer-gap { flex: 1; }
.mcp-action {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: var(--gray-700);
  font-size: 12px;
  font-weight: 550;
}
.mcp-action:hover:not(:disabled) { background: var(--gray-50); border-color: var(--gray-200); color: var(--gray-900); }
.mcp-action:disabled { opacity: .5; cursor: not-allowed; }
.mcp-icon-action {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--gray-500);
}
.mcp-icon-action:hover { background: var(--gray-100); color: var(--gray-900); }
.mcp-icon-action.is-danger:hover { background: var(--color-error-50); color: var(--color-error-700); }

.mcp-dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(0, 0, 0, .22);
}
.mcp-confirm {
  width: min(440px, 92vw);
  padding: 22px;
  background: var(--gray-0);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-deep);
}
.mcp-confirm h3 { font-size: 17px; font-weight: 600; color: var(--gray-1000); margin-bottom: 10px; }
.mcp-confirm p { padding: 10px 12px; background: var(--gray-50); border-radius: var(--radius-md); color: var(--gray-800); font-size: 13px; line-height: 1.6; }
.mcp-confirm small { display: block; margin: 10px 0 18px; color: var(--gray-500); font-size: 12px; }
.mcp-confirm-actions { display: flex; justify-content: flex-end; gap: 8px; }

.mcp-toasts {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 1100;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}
.mcp-toast {
  display: flex;
  align-items: center;
  gap: 7px;
  max-width: 380px;
  padding: 10px 13px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: var(--gray-800);
  font-size: 12px;
  box-shadow: var(--shadow-deep);
}
.mcp-toast.is-error { border-color: var(--color-error-500); color: var(--color-error-700); }

.spin { animation: mcp-spin .9s linear infinite; }
@keyframes mcp-spin { to { transform: rotate(360deg); } }

@media (max-width: 1080px) {
  .mcp-overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 760px) {
  .mcp-page { padding: 24px 16px 48px; }
  .mcp-header, .mcp-toolbar { flex-direction: column; align-items: stretch; }
  .mcp-search { width: 100%; }
  .mcp-grid { grid-template-columns: 1fr; }
}
</style>
