<template>
  <section class="agents-page" @click="menuId = ''" @keydown.esc="menuId = ''">
    <header class="agents-page-bar">
      <div class="agents-tabs" role="tablist" aria-label="智能体管理">
        <button v-for="tab in tabs" :id="`agent-tab-${tab.id}`" :key="tab.id" role="tab" :aria-selected="activeTab === tab.id" aria-controls="agents-content" :tabindex="activeTab === tab.id ? 0 : -1" :class="{ active: activeTab === tab.id }" @keydown.left.prevent="switchTab(tab.id === 'agents' ? 'providers' : 'agents', true)" @keydown.right.prevent="switchTab(tab.id === 'agents' ? 'providers' : 'agents', true)" @click="switchTab(tab.id)">{{ tab.label }}</button>
      </div>
      <div class="agents-counts" aria-label="配置概览">
        <template v-if="activeTab === 'agents'"><span>{{ agents.length }} 个智能体</span><span>{{ agents.filter(a => a.source === 'builtin').length }} 个全局</span><span>{{ agents.filter(a => a.can_edit).length }} 个个人</span></template>
        <template v-else><span>{{ providers.length }} 个供应商</span><span>{{ models.length }} 个模型</span><span>{{ models.filter(m => m.source === 'custom').length }} 个个人配置</span></template>
      </div>
    </header>

    <div id="agents-content" class="agents-content" role="tabpanel" :aria-labelledby="`agent-tab-${activeTab}`">
      <div class="agents-toolbar">
        <label class="agents-search"><Search :size="17" /><input v-model="search" type="search" :aria-label="activeTab === 'agents' ? '搜索智能体' : '搜索模型供应商'" :placeholder="activeTab === 'agents' ? '搜索智能体…' : '搜索供应商或模型…'" /></label>
        <div class="agents-toolbar-actions"><button class="agents-button primary" @click="activeTab === 'agents' ? openAgent() : openModel()"><Plus :size="16" />{{ activeTab === 'agents' ? '新增智能体' : '添加模型连接' }}</button><button class="agents-button icon" aria-label="刷新配置" title="刷新配置" :disabled="loading" @click="load"><RefreshCw :size="17" :class="{ spin: loading }" /></button></div>
      </div>
      <p v-if="notice" class="agents-notice" role="status">{{ notice }}</p>
      <div v-if="error" class="agents-error" role="alert"><CircleAlert :size="17" /><span>{{ error }}</span><button @click="load">重新加载</button></div>
      <div v-if="loading && !loaded" class="agents-empty"><LoaderCircle :size="24" class="spin" /><strong>正在读取配置</strong><p>加载智能体与模型目录。</p></div>
      <template v-else-if="!error">
        <template v-if="activeTab === 'agents'">
          <section v-for="section in sections" :key="section.kind" class="agents-section">
            <h2>{{ section.title }}<span>{{ section.items.length }}</span></h2>
            <div v-if="section.items.length" class="agents-grid">
              <article v-for="agent in section.items" :key="agent.id" class="managed-agent-card" :class="{ 'is-disabled': !agent.enabled }">
                <header><span class="managed-agent-avatar" :class="agent.kind === 'subagent' ? 'violet' : agent.mode === 'deepagents' ? 'rose' : ''"><component :is="agent.kind === 'subagent' ? Puzzle : agent.mode === 'deepagents' ? ScanSearch : Bot" :size="24" :stroke-width="1.8" /></span><div class="managed-agent-card-title"><h3>{{ agent.name }}</h3><p>{{ agent.slug }}</p></div>
                  <div class="agent-menu-anchor"><button class="agent-menu-toggle" :aria-label="`${agent.name}操作菜单`" :aria-expanded="menuId === agent.id" @click.stop="menuId = menuId === agent.id ? '' : agent.id"><MoreHorizontal :size="20" /></button><div v-if="menuId === agent.id" class="agent-menu" @click.stop><button @click="openAgent(agent, !agent.can_edit)"><Settings2 :size="14" />{{ agent.can_edit ? '编辑配置' : '查看配置' }}</button><button @click="openAgent(agent, false, true)"><Copy :size="14" />复制为个人智能体</button><template v-if="agent.can_edit"><button :disabled="busy" @click="toggleAgent(agent)"><Power :size="14" />{{ agent.enabled ? '停用' : '启用' }}</button><button class="danger" @click="confirmDelete('agent', agent)"><Trash2 :size="14" />删除</button></template></div></div>
                </header>
                <p class="agent-description">{{ agent.description || '为这个智能体添加角色说明，让它更好地完成特定任务。' }}</p>
                <div class="agent-model-caption"><Cpu :size="13" />{{ modelName(agent.model_id) }}<span v-if="agent.mode === 'deepagents'">· {{ agent.subagent_ids.length }} 个子智能体</span></div>
                <footer><div class="agent-tags"><span>{{ agent.can_edit ? '个人' : '只读全局' }}</span><span v-if="!agent.enabled" class="paused">已停用</span></div><button class="agent-use" :disabled="!agent.enabled" @click="useAgent(agent)">{{ agent.kind === 'main' ? '去对话' : '试用' }}<ChevronRight :size="14" /></button></footer>
              </article>
            </div>
            <div v-else class="agents-empty compact"><SearchX :size="22" /><strong>{{ search ? '没有匹配的智能体' : '还没有个人子智能体' }}</strong><p>{{ search ? '试试其他名称或关键词。' : '可新增子智能体，并在深度研究模式的主智能体中关联使用。' }}</p></div>
          </section>
        </template>

        <template v-else>
          <p class="providers-intro">连接模型供应商，为智能体选择合适的模型。全局预置由部署配置提供，个人连接可随时管理。</p>
          <div v-if="visibleProviders.length" class="provider-grid">
            <article v-for="provider in visibleProviders" :key="provider.id" class="provider-card">
              <header><span class="managed-agent-avatar"><component :is="provider.models.every(m => m.provider_type === 'local') ? HardDrive : Cloud" :size="23" /></span><div><h3>{{ provider.name }}</h3><p>{{ provider.models.length }} 个模型</p></div><button class="agent-use" :aria-label="`为${provider.name}添加模型`" @click="openModel(null, provider)"><Plus :size="15" />添加</button></header>
              <div v-for="model in provider.models" :key="model.id" class="provider-model">
                <div class="provider-model-copy"><strong>{{ model.name }}</strong><small>{{ model.api_model }}</small><span class="provider-model-state" :class="{ ready: model.available }"><i></i>{{ model.available ? '配置就绪' : '待完善配置' }}<em>{{ model.source === 'custom' ? '个人' : '全局预置' }}</em></span></div>
                <div class="provider-model-actions"><button v-if="model.source === 'custom'" :aria-label="`编辑${model.name}`" title="编辑模型连接" @click="openModel(model)"><Pencil :size="15" /></button><button v-if="model.source === 'custom'" :aria-label="`删除${model.name}`" title="删除模型连接" @click="confirmDelete('model', model)"><Trash2 :size="15" /></button><button :disabled="!model.available" :aria-label="`使用${model.name}`" title="在对话中使用" @click="useModel(model)"><ArrowUpRight :size="16" /></button></div>
              </div>
            </article>
          </div>
          <div v-else class="agents-empty"><SearchX :size="24" /><strong>没有匹配的供应商或模型</strong><p>更换关键词，或添加新的模型连接。</p></div>
        </template>
      </template>
    </div>

    <Teleport to="body">
      <dialog ref="editorDialog" class="agent-dialog" aria-label="智能体与模型配置" @cancel.prevent="closeEditor" @click="closeOnBackdrop">
        <form v-if="editor" @submit.prevent="save">
          <header class="agent-dialog-header"><div><h2>{{ editor.type === 'model' ? (editor.id ? '编辑模型连接' : '添加模型连接') : editor.readonly ? '智能体配置' : editor.id ? '编辑智能体' : '新增智能体' }}</h2><p>{{ editor.type === 'model' ? '连接已有的 OpenAI 兼容模型服务。' : '设置角色、模型与协作能力。保存后在下一次执行生效。' }}</p></div><button type="button" aria-label="关闭配置" :disabled="saving" @click="closeEditor"><X :size="19" /></button></header>
          <div class="agent-dialog-body">
            <fieldset v-if="editor.type === 'agent'" :disabled="saving || editor.readonly">
              <div class="agent-form-grid"><label>名称<input v-model="agentForm.name" maxlength="80" required placeholder="例如：文档研究助手" /></label><label>类型<select v-model="agentForm.kind" :disabled="!!editor.id"><option value="main">主智能体</option><option value="subagent">子智能体</option></select></label></div>
              <label>标识<input v-model="agentForm.slug" maxlength="64" pattern="[a-z][a-z0-9-]*" :disabled="!!editor.id" placeholder="留空自动生成，或使用小写英文与连字符" /></label>
              <label>说明<textarea v-model="agentForm.description" maxlength="500" rows="2" placeholder="这个智能体擅长什么，适合完成什么任务？" /></label>
              <div class="agent-form-grid"><label>运行模式<select v-model="agentForm.mode" :disabled="agentForm.kind === 'subagent'"><option value="dynamic">智能对话 · 工具按需调用</option><option value="deepagents">深度研究 · 子智能体协作</option></select></label><label>默认模型<select v-model="agentForm.model_id"><option :value="null">{{ agentForm.kind === 'subagent' ? '继承主智能体模型' : '跟随对话模型' }}</option><option v-for="model in models" :key="model.id" :value="model.id" :disabled="!model.available">{{ model.name }}{{ model.available ? '' : '（待配置）' }}</option></select></label></div>
              <label>角色指令<textarea v-model="agentForm.system_prompt" maxlength="20000" rows="6" placeholder="定义职责、回答风格和任务要求…" /></label><p class="agent-field-help">角色指令叠加在系统行为规则上；知识库权限、技能和工具安全策略持续生效。</p>
              <label class="agent-checkbox"><input v-model="defaultTools" type="checkbox" />使用默认可用工具</label>
              <div v-if="!defaultTools" class="agent-check-list"><label v-for="tool in tools" :key="tool.name" class="agent-checkbox" :title="tool.description"><input v-model="agentForm.tool_names" type="checkbox" :value="tool.name" /><span>{{ tool.name }}<small v-if="!tool.available">暂不可用</small></span></label><span v-if="!tools.length">暂无已注册工具</span></div>
              <template v-if="agentForm.kind === 'main' && agentForm.mode === 'deepagents'"><label>可委派的子智能体</label><div class="agent-check-list"><label v-for="child in agents.filter(a => a.kind === 'subagent')" :key="child.id" class="agent-checkbox"><input v-model="agentForm.subagent_ids" type="checkbox" :value="child.id" :disabled="!child.enabled && !agentForm.subagent_ids.includes(child.id)" /><span>{{ child.name }}<small>{{ child.enabled ? child.slug : '已停用' }}</small></span></label></div></template>
              <label class="agent-checkbox"><input v-model="agentForm.enabled" type="checkbox" />启用此智能体</label>
            </fieldset>
            <fieldset v-else :disabled="saving">
              <div class="agent-form-grid"><label>显示名称<input v-model="modelForm.name" required maxlength="80" placeholder="例如：本地 Qwen" /></label><label>供应商<input v-model="modelForm.provider_name" maxlength="80" required placeholder="例如：Ollama、DeepSeek" /></label></div>
              <label>连接类型<select v-model="modelForm.provider_type" @change="modelForm.requires_api_key = modelForm.provider_type === 'cloud'"><option value="local">本地 / 局域网模型</option><option value="cloud">云端模型</option></select></label>
              <label>API 地址<input v-model="modelForm.base_url" type="url" required maxlength="512" :placeholder="modelForm.provider_type === 'local' ? 'http://host.docker.internal:11434/v1' : 'https://api.example.com/v1'" /></label><p class="agent-field-help" v-if="modelForm.provider_type === 'local'">Docker 部署连接宿主机服务时，请使用 host.docker.internal。</p>
              <label>模型 ID<input v-model="modelForm.model_name" required maxlength="160" placeholder="供应商实际提供的模型名称" /></label>
              <label>API Key<input v-model="modelForm.api_key" type="password" autocomplete="new-password" maxlength="8192" :required="modelForm.requires_api_key && !modelForm.has_api_key" :placeholder="modelForm.has_api_key ? '已保存密钥；留空保留，填写则替换' : '填写访问密钥'" /></label>
              <div class="agent-form-grid"><label>温度<input v-model.number="modelForm.temperature" type="number" min="0" max="2" step="0.1" required /></label><div class="agent-form-checks"><label class="agent-checkbox"><input v-model="modelForm.requires_api_key" type="checkbox" />需要 API Key</label><label class="agent-checkbox"><input v-model="modelForm.supports_vision" type="checkbox" />支持图片输入</label></div></div>
            </fieldset>
            <p v-if="formError" class="agent-form-error" role="alert">{{ formError }}</p>
          </div>
          <footer class="agent-dialog-footer"><span v-if="editor.readonly">全局预置 · 只读</span><button type="button" class="agents-button" :disabled="saving" @click="closeEditor">{{ editor.readonly ? '关闭' : '取消' }}</button><button v-if="editor.readonly" type="button" class="agents-button primary" @click="copyCurrent">复制为个人智能体</button><button v-else type="submit" class="agents-button primary" :disabled="saving">{{ saving ? '保存中…' : '保存配置' }}</button></footer>
        </form>
      </dialog>
      <dialog ref="deleteDialog" class="agent-dialog agent-delete-dialog" aria-label="确认删除配置" @cancel.prevent="closeDelete">
        <form v-if="deleteTarget" @submit.prevent="remove"><div class="agent-dialog-body"><h2>删除{{ deleteTarget.type === 'agent' ? '智能体' : '模型连接' }}</h2><p>确定删除「{{ deleteTarget.item.name }}」？{{ deleteTarget.type === 'agent' ? '已有对话记录会保留。' : '引用此模型的智能体需要先解除关联。' }}</p><p v-if="formError" class="agent-form-error" role="alert">{{ formError }}</p></div><footer class="agent-dialog-footer"><button type="button" class="agents-button" :disabled="saving" @click="closeDelete">取消</button><button type="submit" class="agents-button danger" :disabled="saving">{{ saving ? '删除中…' : '确认删除' }}</button></footer></form>
      </dialog>
    </Teleport>
  </section>
</template>

<script setup>
import { computed, nextTick, onActivated, onBeforeUnmount, onDeactivated, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowUpRight, Bot, ChevronRight, CircleAlert, Cloud, Copy, Cpu, HardDrive, LoaderCircle, MoreHorizontal, Pencil, Plus, Power, Puzzle, RefreshCw, ScanSearch, Search, SearchX, Settings2, Trash2, X } from 'lucide-vue-next'
import api from '../api'
import { useChatStore } from '../stores/chat'
import { filterAgents, groupProviders } from '../utils/agent-management.js'

const route = useRoute(), router = useRouter(), chatStore = useChatStore()
const tabs = [{ id: 'agents', label: '智能体' }, { id: 'providers', label: '模型供应商' }]
const activeTab = computed(() => route.query.tab === 'providers' ? 'providers' : 'agents')
const agents = ref([]), models = ref([]), tools = ref([]), search = ref('')
const loading = ref(false), loaded = ref(false), busy = ref(false), error = ref(''), notice = ref(''), menuId = ref('')
const editor = ref(null), editorDialog = ref(null), deleteTarget = ref(null), deleteDialog = ref(null), saving = ref(false), formError = ref('')
const agentForm = reactive({}), modelForm = reactive({}), defaultTools = ref(true)
let generation = 0, noticeTimer
const providers = computed(() => groupProviders(models.value))
const visibleProviders = computed(() => providers.value.filter(p => !search.value.trim() || [p.name, ...p.models.flatMap(m => [m.name, m.api_model])].join(' ').toLocaleLowerCase().includes(search.value.trim().toLocaleLowerCase())))
const sections = computed(() => [{kind:'main',title:'主智能体'}, {kind:'subagent',title:'子智能体'}].map(s => ({...s,items:filterAgents(agents.value,search.value,s.kind)})))
const modelName = id => models.value.find(m => m.id === id)?.name || (id ? '模型已移除' : '跟随对话模型')
const message = (err, fallback) => typeof err.response?.data?.detail === 'string' ? err.response.data.detail : fallback
function notify(text) { notice.value = text; clearTimeout(noticeTimer); noticeTimer = setTimeout(() => { notice.value = '' }, 5000) }
async function switchTab(tab, focus = false) { await router.replace({path:'/agents',query:tab === 'providers' ? {tab} : {}}); search.value = ''; menuId.value = ''; if (focus) { await nextTick(); document.getElementById(`agent-tab-${tab}`)?.focus() } }
async function load() {
  const current = ++generation; loading.value = true; error.value = ''
  try { const [catalog, modelList] = await Promise.all([api.get('/agents'), api.get('/chat/models')]); if (current !== generation) return; agents.value = catalog.agents; tools.value = catalog.tools; models.value = modelList.models; loaded.value = true }
  catch (err) { if (current === generation) error.value = message(err,'配置加载失败，请重试。') }
  finally { if (current === generation) loading.value = false }
}
function closeEditor() { if (saving.value) return; editorDialog.value?.close(); editor.value = null; modelForm.api_key = ''; formError.value = '' }
function closeDelete() { if (saving.value) return; deleteDialog.value?.close(); deleteTarget.value = null; formError.value = '' }
function closeOnBackdrop(event) { if (event.target === editorDialog.value) { const bounds = editorDialog.value.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeEditor() } }
async function openAgent(item, readonly = false, copy = false) {
  menuId.value = ''; formError.value = ''
  Object.assign(agentForm, {name:copy ? `${item.name} 副本` : item?.name || '', slug:copy ? '' : item?.slug || '', kind:item?.kind || 'main', description:item?.description || '', mode:item?.mode || 'dynamic', system_prompt:item?.system_prompt || '', model_id:item?.model_id || null, tool_names:[...(item?.tool_names || [])], subagent_ids:[...(item?.subagent_ids || [])], enabled:copy ? true : item?.enabled ?? true})
  defaultTools.value = item?.tool_names == null
  editor.value = {type:'agent',id:copy ? null : item?.id, readonly}
  await nextTick(); if (!editorDialog.value.open) editorDialog.value.showModal()
}
function copyCurrent() { const copy = {...agentForm, tool_names:defaultTools.value ? null : [...agentForm.tool_names]}; void openAgent(copy,false,true) }
async function openModel(item, provider) {
  formError.value = ''; menuId.value = ''
  try {
    const config = item ? await api.get(`/chat/models/${encodeURIComponent(item.id)}/configuration`) : {}
    Object.assign(modelForm, {name:'', provider_name:provider?.name || '', provider_type:'cloud', base_url:'', model_name:'', requires_api_key:true, temperature:0, supports_vision:false, has_api_key:false, ...config, api_key:''})
    editor.value = {type:'model',id:item?.id,readonly:false}; await nextTick(); editorDialog.value.showModal()
  } catch (err) { notify(message(err,'无法读取模型配置，请重试。')) }
}
watch(() => agentForm.kind, kind => { if (kind === 'subagent') { agentForm.mode = 'dynamic'; agentForm.subagent_ids = [] } })
watch(() => agentForm.mode, mode => { if (mode === 'dynamic') agentForm.subagent_ids = [] })
async function save() {
  if (saving.value) return; saving.value = true; formError.value = ''
  try {
    const type = editor.value.type, id = editor.value.id
    const body = type === 'agent' ? {...agentForm,tool_names:defaultTools.value ? null : [...agentForm.tool_names]} : Object.fromEntries(Object.entries(modelForm).filter(([key]) => !['id','has_api_key'].includes(key)))
    const path = type === 'agent' ? '/agents' : '/chat/models'
    if (id) await api.put(`${path}/${encodeURIComponent(id)}`, body); else await api.post(path, body)
    saving.value = false; closeEditor(); await load(); notify('配置已保存，下一次执行将使用新配置。')
  } catch (err) { formError.value = message(err,'保存失败，请检查字段并重试。') }
  finally { saving.value = false }
}
async function toggleAgent(item) {
  if (busy.value) return; busy.value = true; menuId.value = ''
  try { await api.put(`/agents/${encodeURIComponent(item.id)}`, {...item,enabled:!item.enabled}); await load(); notify(item.enabled ? '智能体已停用。' : '智能体已启用。') }
  catch (err) { notify(message(err,'状态更新失败，请重试。')) } finally { busy.value = false }
}
async function confirmDelete(type,item) { menuId.value = ''; formError.value = ''; deleteTarget.value = {type,item}; await nextTick(); deleteDialog.value.showModal() }
async function remove() {
  if (saving.value) return; saving.value = true; formError.value = ''
  try { const target = deleteTarget.value; await api.delete(`${target.type === 'agent' ? '/agents' : '/chat/models'}/${encodeURIComponent(target.item.id)}`); saving.value = false; closeDelete(); await load(); notify('已删除。') }
  catch (err) { formError.value = message(err,'删除失败，请重试。') } finally { saving.value = false }
}
function useAgent(item) { chatStore.startNewConversation(); router.push({path:'/',query:{agent:item.id}}) }
function useModel(item) { chatStore.startNewConversation(); router.push({path:'/',query:{agent:'default-chatbot',model:item.id}}) }
onActivated(load)
onDeactivated(() => { editorDialog.value?.close(); deleteDialog.value?.close(); editor.value = null; deleteTarget.value = null; modelForm.api_key = ''; menuId.value = '' })
onBeforeUnmount(() => { ++generation; clearTimeout(noticeTimer); modelForm.api_key = '' })
</script>

<style src="../styles/agent-management.css"></style>
