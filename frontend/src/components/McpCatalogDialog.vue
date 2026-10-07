<template>
  <Teleport to="body">
    <div class="mcp-catalog-overlay" @click.self="emit('close')">
      <div class="mcp-catalog" role="dialog" aria-modal="true" aria-label="MCP 广场">
        <header class="mcp-catalog-header">
          <div class="mcp-catalog-title">
            <h3>MCP 广场</h3>
            <p>
              目录来自
              <a href="https://www.modelscope.cn/mcp" target="_blank" rel="noopener">ModelScope MCP 广场</a>
              <template v-if="total">· 命中 {{ total }} 个服务</template>
              ；选中后填入凭据即可挂载到本机 Agent。
            </p>
          </div>
          <div class="mcp-catalog-header-actions">
            <label class="mcp-catalog-search">
              <Search :size="14" />
              <input v-model="query" type="search" placeholder="搜索服务：地图 / sqlite / fetch" aria-label="搜索 MCP 广场" />
            </label>
            <button type="button" class="mcp-icon-close" aria-label="关闭" @click="emit('close')"><X :size="16" /></button>
          </div>
        </header>

        <div class="mcp-catalog-body">
          <!-- 左：检索结果 -->
          <section class="mcp-catalog-list" aria-label="广场服务列表">
            <p v-if="loading" class="mcp-catalog-state">
              <LoaderCircle :size="20" class="spin" /><strong>正在检索广场</strong>
            </p>
            <p v-else-if="error" class="mcp-catalog-state is-error">
              <CircleAlert :size="20" /><strong>广场检索失败</strong><span>{{ error }}</span>
              <button type="button" class="mcp-button" @click="loadPage">重试</button>
            </p>
            <p v-else-if="!items.length" class="mcp-catalog-state">
              <SearchX :size="20" /><strong>没有匹配的服务</strong>
              <span>换个关键词，例如「数据库」「浏览器」「office」。</span>
            </p>
            <template v-else>
              <button
                v-for="item in items"
                :key="item.id"
                type="button"
                :class="['mcp-catalog-item', { 'is-active': item.id === selectedId }]"
                @click="selectItem(item)"
              >
                <img v-if="item.logo_url" class="mcp-catalog-logo" :src="item.logo_url" alt="" loading="lazy" />
                <span v-else class="mcp-catalog-logo is-fallback"><Blocks :size="15" /></span>
                <span class="mcp-catalog-item-main">
                  <span class="mcp-catalog-item-head">
                    <strong>{{ item.name }}</strong>
                    <span class="mcp-catalog-views"><Flame :size="11" />{{ formatCount(item.view_count) }}</span>
                  </span>
                  <code>{{ item.id }}</code>
                  <span class="mcp-catalog-item-desc">{{ item.description || '发布者未填写描述' }}</span>
                  <span v-if="item.categories.length" class="mcp-catalog-chips">
                    <span v-for="category in item.categories.slice(0, 3)" :key="category" class="mcp-chip">{{ category }}</span>
                  </span>
                </span>
              </button>
              <footer v-if="items.length" class="mcp-catalog-pager">
                <button type="button" class="mcp-button" :disabled="page <= 1 || loading" @click="goPage(page - 1)">
                  <ChevronLeft :size="13" /> 上一页
                </button>
                <span>第 {{ page }} / {{ maxPage }} 页</span>
                <button type="button" class="mcp-button" :disabled="page >= maxPage || loading" @click="goPage(page + 1)">
                  下一页 <ChevronRight :size="13" />
                </button>
              </footer>
            </template>
          </section>

          <!-- 右：安装表单 -->
          <section class="mcp-catalog-detail" aria-label="安装配置">
            <p v-if="!selectedId" class="mcp-catalog-state">
              <Package :size="20" /><strong>选择一个服务</strong>
              <span>从左侧点选，这里会给出安装计划与需要填写的凭据。</span>
            </p>
            <p v-else-if="detailLoading" class="mcp-catalog-state">
              <LoaderCircle :size="20" class="spin" /><strong>正在读取安装计划</strong>
            </p>
            <p v-else-if="detailError" class="mcp-catalog-state is-error">
              <CircleAlert :size="20" /><strong>详情不可用</strong><span>{{ detailError }}</span>
            </p>
            <template v-else-if="plan">
              <header class="mcp-catalog-detail-head">
                <img v-if="plan.logo_url" class="mcp-catalog-logo" :src="plan.logo_url" alt="" />
                <span v-else class="mcp-catalog-logo is-fallback"><Blocks :size="15" /></span>
                <div>
                  <strong>{{ plan.name }}</strong>
                  <small>
                    {{ plan.author || '未知发布者' }}
                    <a v-if="plan.source_url" :href="plan.source_url" target="_blank" rel="noopener">
                      <ExternalLink :size="11" /> 源码/主页
                    </a>
                  </small>
                </div>
              </header>

              <p class="mcp-catalog-desc">{{ plan.description || '发布者未填写描述。' }}</p>

              <p v-if="plan.kind === 'stdio'" class="mcp-banner is-info">
                <Terminal :size="13" />
                <span>本地 Stdio：<code>{{ plan.stdio_command.join(' ') }}</code></span>
              </p>
              <p v-else-if="plan.kind === 'remote'" class="mcp-banner is-info">
                <Link2 :size="13" /><span>远程端点：<code>{{ plan.url }}</code>（{{ TRANSPORT_LABELS[plan.transport] || plan.transport }}）</span>
              </p>
              <p v-else class="mcp-banner is-warn">
                <CircleAlert :size="13" />
                <span>广场未提供可直接运行的配置：该服务需先在 ModelScope 侧部署，或由你手动填写端点 URL。</span>
              </p>

              <p v-if="plan.runtime && plan.runtime.hint" :class="['mcp-banner', plan.runtime.available ? 'is-info' : 'is-warn']">
                <component :is="plan.runtime.available ? CheckCircle2 : ShieldAlert" :size="13" />
                <span>{{ plan.runtime.hint }}</span>
              </p>

              <p v-if="detail.already_installed" class="mcp-banner is-warn">
                <ShieldAlert :size="13" /><span>同名服务已存在，勾选「覆盖已存在服务」才会替换。</span>
              </p>

              <div class="mcp-catalog-form">
                <label class="mcp-field">
                  <span>服务标识符</span>
                  <input v-model.trim="form.server_id" type="text" placeholder="amap-amap-maps" />
                  <small v-if="errors.server_id" class="is-error">{{ errors.server_id }}</small>
                  <small v-else>作为工具命名空间前缀（{{ form.server_id }}__工具名）。</small>
                </label>

                <label class="mcp-field">
                  <span>展示名称</span>
                  <input v-model.trim="form.name" type="text" />
                </label>

                <template v-if="plan.kind === 'deploy_required'">
                  <label class="mcp-field">
                    <span>端点 URL</span>
                    <input v-model.trim="form.url" type="text" placeholder="https://mcp.example.com/mcp" />
                    <small v-if="errors.url" class="is-error">{{ errors.url }}</small>
                  </label>
                  <label class="mcp-field">
                    <span>传输协议</span>
                    <select v-model="form.transport">
                      <option value="streamable_http">Streamable HTTP</option>
                      <option value="sse">SSE</option>
                    </select>
                  </label>
                </template>

                <label v-for="key in plan.env_schema || []" :key="key" class="mcp-field">
                  <span>{{ key }}</span>
                  <input v-model="form.env[key]" type="password" :placeholder="`广场要求的环境变量 ${key}`" autocomplete="off" />
                  <small v-if="errors[`env.${key}`]" class="is-error">{{ errors[`env.${key}`] }}</small>
                </label>
                <p v-if="(plan.env_schema || []).length" class="mcp-banner is-warn">
                  <ShieldAlert :size="13" />
                  <span>
                    凭据会写入 config/mcp_servers.json（该文件受 git 跟踪）。生产环境建议填
                    <code>{{ envPlaceholder }}</code> 这类占位符，把真实值放到 .env 里。
                  </span>
                </p>

                <div class="mcp-catalog-form-row">
                  <label class="mcp-field">
                    <span>调用超时（秒）</span>
                    <input v-model.number="form.timeout_s" type="number" :min="TIMEOUT_RANGE.min" :max="TIMEOUT_RANGE.max" />
                  </label>
                  <label class="mcp-field">
                    <span>工具权限</span>
                    <select v-model="form.scope">
                      <option value="all">全部工具（*）</option>
                      <option value="custom">指定白名单</option>
                    </select>
                  </label>
                </div>

                <label v-if="form.scope === 'custom'" class="mcp-field">
                  <span>白名单工具名</span>
                  <input v-model="form.allowed_tools" type="text" placeholder="query, list_tables" />
                  <small v-if="errors.allowed_tools" class="is-error">{{ errors.allowed_tools }}</small>
                </label>

                <label class="mcp-checkbox">
                  <input v-model="form.enabled" type="checkbox" />
                  <span>安装后立即连接</span>
                </label>
                <label v-if="detail.already_installed" class="mcp-checkbox">
                  <input v-model="form.overwrite" type="checkbox" />
                  <span>覆盖已存在的同名服务</span>
                </label>
              </div>

              <p v-if="installError" class="mcp-banner is-error"><CircleAlert :size="13" /><span>{{ installError }}</span></p>
              <p v-else-if="firstError" class="mcp-banner is-error"><CircleAlert :size="13" /><span>{{ firstError }}</span></p>

              <footer class="mcp-catalog-detail-footer">
                <button type="button" class="mcp-button is-primary" :disabled="installing" @click="install">
                  <LoaderCircle v-if="installing" :size="14" class="spin" />
                  <Download v-else :size="14" />
                  {{ installing ? '正在安装…' : '安装到本机 Agent' }}
                </button>
              </footer>
            </template>
          </section>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import {
  Blocks, CheckCircle2, ChevronLeft, ChevronRight, CircleAlert, Download,
  ExternalLink, Flame, Link2, LoaderCircle, Package, Search, SearchX, ShieldAlert, Terminal, X,
} from 'lucide-vue-next'
import api from '../api'
import {
  TIMEOUT_RANGE,
  TRANSPORT_LABELS,
  formatCount,
  installFormToPayload,
  planToInstallForm,
  validateInstallForm,
} from '../utils/mcp-servers'

const emit = defineEmits(['close', 'installed'])

const PAGE_SIZE = 12
const MAX_ITEMS = 100 // ModelScope 限制 page × page_size ≤ 100

const query = ref('')
const page = ref(1)
const items = ref([])
const total = ref(0)
const loading = ref(false)
const error = ref('')

const selectedId = ref('')
const detail = ref(null)
const detailLoading = ref(false)
const detailError = ref('')

const form = reactive(planToInstallForm({}))
const errors = ref({})
const installing = ref(false)
const installError = ref('')

const plan = computed(() => detail.value?.plan || null)
const envPlaceholder = computed(() => {
  const key = (plan.value?.env_schema || [])[0] || 'ENV_NAME'
  return `\${${key}}` // 后端加载配置时会展开 ${VAR} / ${VAR:-default}
})
const maxPage = computed(() => Math.max(1, Math.floor(MAX_ITEMS / PAGE_SIZE)))
const firstError = computed(() => Object.values(errors.value)[0] || '')

function errorDetail(err) {
  const detailValue = err?.response?.data?.detail
  if (Array.isArray(detailValue)) return detailValue.map(item => item.msg || JSON.stringify(item)).join('；')
  return detailValue || err?.message || '未知错误'
}

async function loadPage() {
  loading.value = true
  error.value = ''
  try {
    const res = await api.get('/mcp/catalog', { search: query.value.trim(), page: page.value, page_size: PAGE_SIZE })
    items.value = res?.items || []
    total.value = Number(res?.total || 0)
  } catch (err) {
    error.value = errorDetail(err)
    items.value = []
  } finally {
    loading.value = false
  }
}

function goPage(next) {
  if (next < 1 || next > maxPage.value) return
  page.value = next
  loadPage()
}

let searchTimer = null
watch(query, () => {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => {
    page.value = 1
    loadPage()
  }, 350)
})
onBeforeUnmount(() => clearTimeout(searchTimer))

async function selectItem(item) {
  selectedId.value = item.id
  detail.value = null
  detailError.value = ''
  installError.value = ''
  errors.value = {}
  detailLoading.value = true
  try {
    const res = await api.get(`/mcp/catalog/${item.id}`)
    detail.value = res
    Object.assign(form, planToInstallForm(res?.plan || {}))
  } catch (err) {
    detailError.value = errorDetail(err)
  } finally {
    detailLoading.value = false
  }
}

async function install() {
  if (!plan.value || installing.value) return
  const found = validateInstallForm(form, plan.value)
  errors.value = found
  if (Object.keys(found).length) return
  installing.value = true
  installError.value = ''
  try {
    const definition = await api.post(`/mcp/catalog/${selectedId.value}/install`, installFormToPayload(form))
    emit('installed', definition)
  } catch (err) {
    installError.value = errorDetail(err)
  } finally {
    installing.value = false
  }
}

function onKeydown(event) {
  if (event.key === 'Escape') emit('close')
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  loadPage()
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<style scoped>
.mcp-catalog-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(0, 0, 0, .22);
}
.mcp-catalog {
  width: min(1080px, 96vw);
  height: min(88vh, 860px);
  display: flex;
  flex-direction: column;
  background: var(--gray-0);
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-deep);
  overflow: hidden;
}
.mcp-catalog-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding: 18px 22px 14px;
  border-bottom: 1px solid var(--gray-100);
}
.mcp-catalog-title h3 { font-size: 17px; font-weight: 600; color: var(--gray-1000); }
.mcp-catalog-title p { margin-top: 3px; color: var(--gray-500); font-size: 12px; }
.mcp-catalog-title a { color: var(--main-700); text-decoration: underline; }
.mcp-catalog-header-actions { display: flex; align-items: center; gap: 10px; flex: 0 0 auto; }
.mcp-catalog-search {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 260px;
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  color: var(--gray-500);
}
.mcp-catalog-search:focus-within { border-color: var(--gray-400); }
.mcp-catalog-search input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; color: var(--gray-900); font-size: 13px; }
.mcp-icon-close {
  width: 30px; height: 30px; display: grid; place-items: center;
  border: 0; border-radius: var(--radius-md); background: transparent; color: var(--gray-500);
}
.mcp-icon-close:hover { background: var(--gray-100); color: var(--gray-900); }

.mcp-catalog-body { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); }
.mcp-catalog-list {
  min-height: 0;
  overflow: auto;
  padding: 10px;
  border-right: 1px solid var(--gray-100);
  background: var(--gray-25);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mcp-catalog-item {
  display: flex;
  gap: 10px;
  width: 100%;
  padding: 10px 11px;
  text-align: left;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: inherit;
}
.mcp-catalog-item:hover { border-color: var(--gray-200); }
.mcp-catalog-item.is-active { border-color: var(--main-500); background: var(--main-5); }
.mcp-catalog-logo {
  width: 34px; height: 34px; flex: 0 0 auto;
  border-radius: var(--radius-md);
  object-fit: cover;
  background: var(--gray-50);
}
.mcp-catalog-logo.is-fallback { display: grid; place-items: center; color: var(--gray-500); }
.mcp-catalog-item-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.mcp-catalog-item-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.mcp-catalog-item-head strong { font-size: 13px; color: var(--gray-1000); }
.mcp-catalog-views { display: inline-flex; align-items: center; gap: 3px; color: var(--gray-500); font-size: 11px; flex: 0 0 auto; }
.mcp-catalog-item code { color: var(--gray-500); font-family: var(--font-mono); font-size: 10px; overflow-wrap: anywhere; }
.mcp-catalog-item-desc {
  color: var(--gray-600);
  font-size: 11px;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.mcp-catalog-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.mcp-chip { padding: 1px 6px; border-radius: var(--radius-sm); background: var(--gray-100); color: var(--gray-700); font-size: 10px; }
.mcp-catalog-pager {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 6px 2px 2px; color: var(--gray-500); font-size: 11px;
  position: sticky; bottom: 0; background: var(--gray-25);
}

.mcp-catalog-detail { min-height: 0; overflow: auto; padding: 16px 20px 18px; display: flex; flex-direction: column; gap: 10px; }
.mcp-catalog-detail-head { display: flex; align-items: center; gap: 10px; }
.mcp-catalog-detail-head strong { display: block; font-size: 14px; color: var(--gray-1000); }
.mcp-catalog-detail-head small { display: inline-flex; align-items: center; gap: 8px; color: var(--gray-500); font-size: 11px; }
.mcp-catalog-detail-head a { display: inline-flex; align-items: center; gap: 3px; color: var(--main-700); }
.mcp-catalog-desc { color: var(--gray-600); font-size: 12px; line-height: 1.6; }

.mcp-catalog-form { display: flex; flex-direction: column; gap: 10px; }
.mcp-catalog-form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.mcp-field { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.mcp-field > span { font-size: 12px; font-weight: 550; color: var(--gray-800); }
.mcp-field input, .mcp-field select {
  width: 100%; padding: 7px 9px; font-size: 12px;
  border: 1px solid var(--gray-200); border-radius: var(--radius-md);
  background: var(--gray-0); color: var(--gray-900); outline: 0;
}
.mcp-field input:focus, .mcp-field select:focus { border-color: var(--main-500); box-shadow: 0 0 0 3px var(--main-50); }
.mcp-field small { color: var(--gray-500); font-size: 11px; }
.mcp-field small.is-error { color: var(--color-error-700); }
.mcp-checkbox { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--gray-700); }

.mcp-banner {
  display: flex; align-items: flex-start; gap: 7px;
  padding: 8px 10px; border-radius: var(--radius-md); font-size: 11px; line-height: 1.55;
}
.mcp-banner svg { flex: 0 0 auto; margin-top: 1px; }
.mcp-banner code { font-family: var(--font-mono); font-size: 11px; overflow-wrap: anywhere; }
.mcp-banner.is-info { background: var(--gray-50); color: var(--gray-700); }
.mcp-banner.is-warn { background: var(--color-warning-50); color: var(--color-warning-900); }
.mcp-banner.is-error { background: var(--color-error-50); color: var(--color-error-700); }

.mcp-catalog-state {
  min-height: 180px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 7px; padding: 20px; text-align: center;
  color: var(--gray-500); font-size: 12px;
}
.mcp-catalog-state strong { color: var(--gray-800); font-size: 13px; }
.mcp-catalog-state span { max-width: 320px; }
.mcp-catalog-state.is-error strong { color: var(--color-error-700); }
.mcp-catalog-detail-footer { margin-top: auto; display: flex; justify-content: flex-end; padding-top: 6px; }

.mcp-button {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  padding: 7px 13px; font-size: 12px; font-weight: 550;
  border: 1px solid var(--gray-150); border-radius: var(--radius-md);
  background: var(--gray-0); color: var(--gray-700);
}
.mcp-button:hover:not(:disabled) { background: var(--gray-50); color: var(--gray-900); }
.mcp-button.is-primary { background: var(--main-700); border-color: var(--main-700); color: var(--gray-0); }
.mcp-button.is-primary:hover:not(:disabled) { background: var(--main-600); border-color: var(--main-600); }
.mcp-button:disabled { opacity: .5; cursor: not-allowed; }
.spin { animation: mcp-catalog-spin .9s linear infinite; }
@keyframes mcp-catalog-spin { to { transform: rotate(360deg); } }

@media (max-width: 900px) {
  .mcp-catalog-body { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) minmax(0, 1.2fr); }
  .mcp-catalog-list { border-right: 0; border-bottom: 1px solid var(--gray-100); }
  .mcp-catalog-header { flex-direction: column; }
  .mcp-catalog-search { width: 100%; }
}
</style>
