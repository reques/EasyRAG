<template>
  <Teleport to="body">
    <div class="mcp-dialog-overlay" @click.self="emit('close')">
      <div class="mcp-dialog" role="dialog" aria-modal="true" :aria-label="title">
        <header class="mcp-dialog-header">
          <div>
            <h3>{{ title }}</h3>
            <p>{{ subtitle }}</p>
          </div>
          <button type="button" class="mcp-dialog-close" aria-label="关闭" @click="emit('close')">
            <X :size="16" />
          </button>
        </header>

        <form class="mcp-dialog-form" novalidate @submit.prevent="submit">
          <div class="mcp-dialog-body">
            <!-- 基本信息 -->
            <section class="mcp-fieldset">
              <h4>基本信息</h4>
              <div class="mcp-field">
                <label for="mcp-server-id">服务标识符</label>
                <input
                  id="mcp-server-id"
                  ref="firstField"
                  v-model.trim="form.server_id"
                  type="text"
                  :disabled="isEdit"
                  placeholder="例如: postgres-query"
                  autocomplete="off"
                />
                <small v-if="isEdit">标识符不可修改：它是工具命名空间前缀（{{ form.server_id }}__工具名）。</small>
                <small v-else>仅英文、数字、下划线与连字符，作为工具命名空间前缀隔离同名工具。</small>
                <small v-if="errors.server_id" class="is-error">{{ errors.server_id }}</small>
              </div>

              <div class="mcp-field">
                <label for="mcp-server-name">展示名称</label>
                <input id="mcp-server-name" v-model.trim="form.name" type="text" placeholder="例如: 数据库只读查询" />
                <small v-if="errors.name" class="is-error">{{ errors.name }}</small>
              </div>

              <div class="mcp-field">
                <label for="mcp-server-desc">用途描述</label>
                <input id="mcp-server-desc" v-model.trim="form.description" type="text" placeholder="说明这个服务为 Agent 补充什么能力" />
              </div>
            </section>

            <!-- 传输协议 -->
            <section class="mcp-fieldset">
              <h4>传输协议</h4>
              <div class="mcp-option-list">
                <label
                  v-for="option in TRANSPORT_OPTIONS"
                  :key="option.value"
                  :class="['mcp-option', { 'is-active': form.transport === option.value, 'is-disabled': isEdit }]"
                >
                  <input v-model="form.transport" type="radio" :value="option.value" :disabled="isEdit" name="mcp-transport" />
                  <span class="mcp-option-mark" aria-hidden="true"></span>
                  <span class="mcp-option-copy">
                    <strong>{{ option.label }}</strong>
                    <small>{{ option.hint }}</small>
                  </span>
                </label>
              </div>
              <small v-if="isEdit">传输协议不可修改，如需更换请删除该服务后重新创建。</small>
              <small v-else>远程服务适合共享部署；Stdio 仅允许执行后端白名单程序（{{ TRUSTED_STDIO_BINARIES.join(' / ') }}）。</small>
            </section>

            <!-- 连接参数 -->
            <section class="mcp-fieldset">
              <h4>连接参数</h4>

              <template v-if="form.transport === 'stdio'">
                <div class="mcp-field">
                  <label for="mcp-stdio-command">启动命令（stdio_command）</label>
                  <textarea
                    id="mcp-stdio-command"
                    v-model="form.stdio_command"
                    rows="2"
                    :disabled="isEdit"
                    placeholder="python -m app.tools.mcp.health_server"
                  ></textarea>
                  <small v-if="isEdit" class="mcp-hint-locked"><Lock :size="12" /> 命令与协议一样按定义固化，编辑接口不支持修改。</small>
                  <small v-else>命令以空格分隔，例如 <code>npx -y @modelcontextprotocol/server-filesystem ./volumes</code>。</small>
                  <small v-if="errors.stdio_command" class="is-error">{{ errors.stdio_command }}</small>
                </div>

                <div class="mcp-field">
                  <label for="mcp-stdio-env">环境变量（可选）</label>
                  <textarea
                    id="mcp-stdio-env"
                    v-model="form.env_text"
                    rows="2"
                    placeholder="AMAP_MAPS_API_KEY=your-key"
                  ></textarea>
                  <small>每行一个 <code>KEY=VALUE</code>；与父进程环境合并（不会丢失 PATH），用于 API Key 等凭据。</small>
                  <small v-if="errors.env_text" class="is-error">{{ errors.env_text }}</small>
                </div>
              </template>

              <template v-else>
                <div class="mcp-field">
                  <label for="mcp-server-url">端点 URL</label>
                  <input id="mcp-server-url" v-model.trim="form.url" type="text" placeholder="https://mcp.example.com/sse" autocomplete="off" />
                  <small v-if="errors.url" class="is-error">{{ errors.url }}</small>
                </div>

                <div class="mcp-field">
                  <label for="mcp-server-token">认证令牌（可选）</label>
                  <input
                    id="mcp-server-token"
                    v-model="form.auth_token"
                    type="password"
                    placeholder="留空则不带 Authorization 头"
                    autocomplete="new-password"
                  />
                  <small v-if="isEdit">留空表示保持原有令牌不变，填写则覆盖。</small>
                  <small v-else>填写后会自动组装为 <code>Authorization: Bearer &lt;token&gt;</code> 请求头。</small>
                </div>
              </template>
            </section>

            <!-- 调用与权限 -->
            <section class="mcp-fieldset">
              <h4>调用与权限</h4>

              <div class="mcp-field-row">
                <div class="mcp-field">
                  <label for="mcp-server-timeout">单次调用超时（秒）</label>
                  <input
                    id="mcp-server-timeout"
                    v-model.number="form.timeout_s"
                    type="number"
                    :min="TIMEOUT_RANGE.min"
                    :max="TIMEOUT_RANGE.max"
                  />
                  <small v-if="errors.timeout_s" class="is-error">{{ errors.timeout_s }}</small>
                </div>

                <div class="mcp-field mcp-field-switch">
                  <label class="mcp-switch">
                    <input v-model="form.enabled" type="checkbox" />
                    <span class="mcp-switch-track" aria-hidden="true"><span></span></span>
                    <span>随应用启动时自动连接</span>
                  </label>
                  <small>关闭后服务保持已配置但待命状态，可在列表里手动连接。</small>
                </div>
              </div>

              <div class="mcp-field">
                <label>工具权限白名单</label>
                <div class="mcp-option-list is-inline">
                  <label :class="['mcp-option', { 'is-active': form.scope === 'all' }]">
                    <input v-model="form.scope" type="radio" value="all" name="mcp-scope" />
                    <span class="mcp-option-mark" aria-hidden="true"></span>
                    <span class="mcp-option-copy">
                      <strong>全部工具（*）</strong>
                      <small>放行该服务暴露的所有工具</small>
                    </span>
                  </label>
                  <label :class="['mcp-option', { 'is-active': form.scope === 'custom' }]">
                    <input v-model="form.scope" type="radio" value="custom" name="mcp-scope" />
                    <span class="mcp-option-mark" aria-hidden="true"></span>
                    <span class="mcp-option-copy">
                      <strong>指定白名单</strong>
                      <small>最小权限，仅放行列出的工具</small>
                    </span>
                  </label>
                </div>
                <textarea
                  v-if="form.scope === 'custom'"
                  v-model="form.allowed_tools"
                  rows="2"
                  placeholder="read_file, list_directory, get_file_info"
                  aria-label="工具白名单"
                ></textarea>
                <small v-if="errors.allowed_tools" class="is-error">{{ errors.allowed_tools }}</small>
                <small v-else-if="form.scope === 'all'" class="mcp-hint-warn">
                  <ShieldAlert :size="12" /> 全部放行会绕过最小权限原则，仅建议用于受信任的自建服务。
                </small>
                <small v-else>多个工具名用逗号或换行分隔；服务连接后可在列表里对照实际工具名。</small>
              </div>

              <div class="mcp-field">
                <label for="mcp-server-caps">沙箱能力标签（可选）</label>
                <input id="mcp-server-caps" v-model.trim="form.capabilities" type="text" placeholder="fs.read, db.read, ops.monitor" />
                <small>标签会继承给该服务注册的每个工具，供 sandbox 审计与策略判定使用。</small>
              </div>
            </section>

            <p v-if="error" class="mcp-dialog-alert">
              <CircleAlert :size="14" /> {{ error }}
            </p>
            <p v-else-if="errorList.length" class="mcp-dialog-alert">
              <CircleAlert :size="14" /> {{ errorList[0] }}
            </p>
          </div>

          <footer class="mcp-dialog-footer">
            <button type="button" class="mcp-button" @click="emit('close')">取消</button>
            <button type="submit" class="mcp-button is-primary" :disabled="submitting">
              <LoaderCircle v-if="submitting" :size="14" class="spin" />
              {{ submitting ? '正在保存…' : submitLabel }}
            </button>
          </footer>
        </form>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { CircleAlert, LoaderCircle, Lock, ShieldAlert, X } from 'lucide-vue-next'
import {
  TIMEOUT_RANGE,
  TRUSTED_STDIO_BINARIES,
  emptyForm,
  formToPayload,
  serverToForm,
  validateForm,
} from '../utils/mcp-servers'

const props = defineProps({
  /** 传入服务定义表示编辑，传 null 表示新建。 */
  server: { type: Object, default: null },
  submitting: { type: Boolean, default: false },
  /** 后端返回的提交错误，由父组件透传。 */
  error: { type: String, default: '' },
})

const emit = defineEmits(['close', 'submit'])

const TRANSPORT_OPTIONS = [
  { value: 'sse', label: 'SSE', hint: '远程 Server-Sent Events，生产推荐' },
  { value: 'streamable_http', label: 'Streamable HTTP', hint: '远程长连接 HTTP 传输' },
  { value: 'stdio', label: 'Stdio', hint: '本地受信任子进程，仅白名单程序' },
]

const firstField = ref(null)
const form = reactive(emptyForm())
const errors = ref({})

const isEdit = computed(() => Boolean(props.server))
const title = computed(() => (isEdit.value ? '编辑 MCP 服务' : '添加 MCP 服务'))
const subtitle = computed(() => (isEdit.value
  ? `服务 ${props.server?.server_id || ''} · 保存后按需重连`
  : '注册远程服务或本地受信任能力，工具会以命名空间注入 Agent'))
const submitLabel = computed(() => (isEdit.value ? '保存修改' : '创建并挂载'))
const errorList = computed(() => Object.values(errors.value))

watch(
  () => props.server,
  (server) => {
    Object.assign(form, server ? serverToForm(server) : emptyForm())
    errors.value = {}
  },
  { immediate: true },
)

function submit() {
  const result = validateForm(form, { mode: isEdit.value ? 'edit' : 'create' })
  errors.value = result
  if (Object.keys(result).length) return
  emit('submit', formToPayload(form, { mode: isEdit.value ? 'edit' : 'create' }))
}

function onKeydown(event) {
  if (event.key === 'Escape') emit('close')
}

onMounted(async () => {
  window.addEventListener('keydown', onKeydown)
  await nextTick()
  if (!isEdit.value) firstField.value?.focus()
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown))
</script>

<style scoped>
.mcp-dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(0, 0, 0, .22);
}
.mcp-dialog {
  width: min(640px, 94vw);
  max-height: min(88vh, 900px);
  display: flex;
  flex-direction: column;
  background: var(--gray-0);
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-deep);
  overflow: hidden;
}
.mcp-dialog-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 22px 16px;
  border-bottom: 1px solid var(--gray-100);
}
.mcp-dialog-header h3 { font-size: 17px; font-weight: 600; color: var(--gray-1000); }
.mcp-dialog-header p { margin-top: 3px; color: var(--gray-500); font-size: 12px; }
.mcp-dialog-close {
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--gray-500);
}
.mcp-dialog-close:hover { background: var(--gray-100); color: var(--gray-900); }
.mcp-dialog-close:focus-visible { outline: 2px solid var(--main-500); outline-offset: 2px; }

.mcp-dialog-form { display: flex; flex-direction: column; min-height: 0; }
.mcp-dialog-body { padding: 18px 22px; overflow: auto; display: flex; flex-direction: column; gap: 18px; }

.mcp-fieldset { display: flex; flex-direction: column; gap: 12px; }
.mcp-fieldset h4 {
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .05em;
  color: var(--gray-500);
}
.mcp-field { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.mcp-field > label { font-size: 13px; font-weight: 550; color: var(--gray-800); }
.mcp-field input[type="text"],
.mcp-field input[type="password"],
.mcp-field input[type="number"],
.mcp-field textarea {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--gray-200);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: var(--gray-900);
  font-size: 13px;
  outline: 0;
  resize: vertical;
}
.mcp-field input:focus,
.mcp-field textarea:focus { border-color: var(--main-500); box-shadow: 0 0 0 3px var(--main-50); }
.mcp-field input:disabled,
.mcp-field textarea:disabled { background: var(--gray-50); color: var(--gray-500); cursor: not-allowed; }
.mcp-field small { color: var(--gray-500); font-size: 11px; line-height: 1.5; display: inline-flex; align-items: center; gap: 4px; }
.mcp-field small.is-error { color: var(--color-error-700); }
.mcp-field small code { font-family: var(--font-mono); font-size: 11px; background: var(--gray-50); border-radius: 4px; padding: 1px 4px; }
.mcp-hint-warn { color: var(--color-warning-900); }
.mcp-hint-locked { color: var(--gray-500); }
.mcp-field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
.mcp-field-switch { gap: 6px; }

.mcp-option-list { display: flex; flex-direction: column; gap: 8px; }
.mcp-option-list.is-inline { flex-direction: row; flex-wrap: wrap; }
.mcp-option-list.is-inline .mcp-option { flex: 1 1 220px; }
.mcp-option {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: 9px;
  padding: 10px 12px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  cursor: pointer;
}
.mcp-option:hover { border-color: var(--gray-300); }
.mcp-option.is-active { border-color: var(--main-500); background: var(--main-5); }
.mcp-option.is-disabled { cursor: not-allowed; opacity: .6; }
.mcp-option input { position: absolute; opacity: 0; pointer-events: none; }
.mcp-option-mark {
  flex: 0 0 auto;
  width: 14px;
  height: 14px;
  margin-top: 3px;
  border: 1px solid var(--gray-300);
  border-radius: 50%;
  background: var(--gray-0);
}
.mcp-option.is-active .mcp-option-mark { border-color: var(--main-500); box-shadow: inset 0 0 0 3px var(--main-500); }
.mcp-option:focus-within { outline: 2px solid var(--main-500); outline-offset: 2px; }
.mcp-option-copy { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.mcp-option-copy strong { font-size: 13px; font-weight: 600; color: var(--gray-900); }
.mcp-option-copy small { color: var(--gray-500); font-size: 11px; }

.mcp-switch { position: relative; display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--gray-800); cursor: pointer; }
.mcp-switch input { position: absolute; opacity: 0; pointer-events: none; }
.mcp-switch-track {
  width: 34px;
  height: 20px;
  border-radius: var(--radius-full);
  background: var(--gray-200);
  display: inline-flex;
  align-items: center;
  padding: 2px;
  transition: background .16s;
}
.mcp-switch-track > span { width: 16px; height: 16px; border-radius: 50%; background: var(--gray-0); transition: transform .16s; }
.mcp-switch input:checked + .mcp-switch-track { background: var(--main-500); }
.mcp-switch input:checked + .mcp-switch-track > span { transform: translateX(14px); }
.mcp-switch:focus-within .mcp-switch-track { outline: 2px solid var(--main-500); outline-offset: 2px; }

.mcp-dialog-alert {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 9px 11px;
  border-radius: var(--radius-md);
  background: var(--color-error-50);
  color: var(--color-error-700);
  font-size: 12px;
}
.mcp-dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 9px;
  padding: 14px 22px;
  border-top: 1px solid var(--gray-100);
  background: var(--gray-10);
}
.mcp-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 15px;
  border: 1px solid var(--gray-150);
  border-radius: var(--radius-md);
  background: var(--gray-0);
  color: var(--gray-700);
  font-size: 13px;
  font-weight: 550;
}
.mcp-button:hover:not(:disabled) { background: var(--gray-50); color: var(--gray-900); }
.mcp-button.is-primary { background: var(--main-700); border-color: var(--main-700); color: var(--gray-0); }
.mcp-button.is-primary:hover:not(:disabled) { background: var(--main-600); border-color: var(--main-600); }
.mcp-button:disabled { opacity: .5; cursor: not-allowed; }
.spin { animation: mcp-dialog-spin .9s linear infinite; }
@keyframes mcp-dialog-spin { to { transform: rotate(360deg); } }

@media (max-width: 620px) {
  .mcp-field-row { grid-template-columns: 1fr; }
  .mcp-dialog { max-height: 92vh; }
}
</style>
