<template>
  <div class="chat-view" :class="{ 'has-task-panel': taskPanel.tasks.length > 0 }">
    <!-- 消息区 -->
    <div class="chat-main">
    <!-- 消息列表 -->
    <div class="chat-messages" ref="msgContainer" :class="{ 'is-empty': messages.length === 0 && !sending }">
      <!-- 侧边状态栏展开入口：有任务且面板收起时显示 -->
      <div v-if="taskPanel.tasks.length && !taskPanel.visible" class="task-panel-toggle">
        <button
          type="button"
          class="task-panel-toggle-btn"
          title="展开任务状态栏"
          aria-label="展开任务状态栏"
          :aria-expanded="taskPanel.visible"
          @click="taskPanel.visible = true"
        >
          <ListChecks :size="14" />
          展开状态栏
          <span v-if="taskPanel.tasks.length" class="task-panel-toggle-meta">
            {{ taskProgress.done }}/{{ taskProgress.total }}
          </span>
        </button>
      </div>
      <div class="chat-column">
        <!-- 空状态 -->
        <div v-if="messages.length === 0 && !sending" class="chat-empty">
          <div class="chat-empty-mark"><Asterisk :size="26" :stroke-width="1.9" /></div>
          <span class="chat-empty-eyebrow"><i></i> KNOWLEDGE ASSISTANT</span>
          <h3>从一个好问题开始</h3>
          <p>连接知识库、联网搜索与多智能体，为你梳理复杂信息。</p>
          <div class="chat-starter-grid" aria-label="推荐问题">
            <button v-for="starter in chatStarters" :key="starter.prompt" type="button" @click="useStarter(starter.prompt)">
              <span><component :is="starter.icon" :size="16" /></span>
              <strong>{{ starter.title }}</strong>
              <small>{{ starter.description }}</small>
              <ArrowUpRight :size="14" />
            </button>
          </div>
        </div>

        <template v-for="(msg, i) in messages" :key="msg.ts || i">
          <!-- 等待首 token 时的空 assistant 占位不渲染，由下方「思考中」气泡代替，
               避免空灰条 + 思考中两个框同时出现 -->
          <div
            v-if="!(sending && msg.role === 'assistant' && !msg.content && !msg.steps?.length && !msg.progressSummaries?.length && i === messages.length - 1)"
            :class="['message', msg.role, { 'msg-enter': msg.enter }]"
          >
          <!-- 用户消息时间: 显示在气泡外上方, 左对齐时间标签, 不放进气泡里 -->
          <div v-if="msg.role === 'user' && msg.time && shouldShowTimeSeparator(i)" class="message-time-separator">{{ msg.time }}</div>
          <div class="message-body">
            <!-- AI 消息时间分隔条: 首条消息 / 距上一条超过 10 分钟时, 居中显示在消息上方 -->
            <div v-if="msg.role === 'assistant' && msg.time && shouldShowTimeSeparator(i)" class="message-time-separator">{{ msg.time }}</div>
            <!-- 操作流（Cursor/Copilot 风格：图标+英文动词+对象），绑定在该条消息上，
                 渲染在答案上方，按实际走的节点（意图/检索/工具/推理/生成）实时流转，
                 不被下一轮覆盖 -->
            <ProgressJournal
              v-if="msg.progressSummaries && msg.progressSummaries.length"
              :items="msg.progressSummaries"
              :running="msg.stepsLoading"
              :error="msg.error || ''"
              :stopped="!!msg.stopped"
            />
            <AgentActivity
              v-if="msg.steps && msg.steps.length"
              :steps="msg.steps"
              :artifacts="msg.artifacts"
              :running="msg.stepsLoading"
              :error="msg.error || ''"
            />
            <div v-if="msg.role === 'user' && (msg.skills?.length || msg.deepResearch)" class="message-skill-strip">
              <span v-if="msg.deepResearch" class="deep-research-chip">
                <Sparkles :size="11" /> 深度研究
              </span>
              <span v-for="skill in msg.skills" :key="skill.id" class="message-skill-chip">
                <WandSparkles :size="11" /> {{ skill.name }}
              </span>
            </div>
            <!-- 被终止的一轮：提示已停止且未保存（后端已删除该轮记录） -->
            <div v-if="msg.stopped" class="message-stopped-note">
              <Square :size="11" /> 已停止生成 · 本轮对话未保存
            </div>
            <div class="message-text" v-html="renderContent(msg.content)"></div>
            <div v-if="msg.role === 'user' && msg.image" class="message-image">
              <img :src="msg.image" alt="用户上传图片" />
            </div>
            <!-- 知识库 / 检索引用块 -->
            <div v-if="msg.sources && msg.sources.length" class="message-sources">
              <div class="sources-title">
                <BookOpen :size="13" /> 参考来源
              </div>
              <ol class="sources-list">
                <li v-for="(s, si) in msg.sources" :key="si">
                  <span v-if="s.type === 'kb'" class="source-tag kb">知识库</span>
                  <span v-else-if="s.type === 'knowledge_graph'" class="source-tag kg">图谱</span>
                  <span v-else-if="s.url" class="source-tag web">网页</span>
                  <a v-if="s.url" :href="s.url" target="_blank" rel="noopener noreferrer">{{ s.title || s.url }}</a>
                  <!-- 知识库引用: 有 file_id 时可点击跳转到文档详情 -->
                  <a
                    v-else-if="(s.type === 'kb' || s.type === 'knowledge_graph') && s.file_id"
                    class="source-link"
                    @click.prevent="goToSource(s)"
                    href="#"
                  >{{ s.title }}</a>
                  <span v-else>{{ s.title }}</span>
                </li>
              </ol>
            </div>
            <div v-if="msg.meta && (msg.meta.agentMode || msg.meta.intent || msg.meta.elapsed || msg.meta.modelName || msg.meta.skillNames?.length)" class="message-meta">
              <span v-if="msg.meta.agentMode" class="message-mode-badge" :class="`mode-${msg.meta.agentMode}`">{{ modeLabel(msg.meta.agentMode) }}</span>
              <span v-if="msg.meta.modelName">模型: {{ msg.meta.modelName }}</span>
              <span v-if="msg.meta.skillNames?.length">Skill: {{ msg.meta.skillNames.join('、') }}</span>
              <span v-if="msg.meta.intent">意图: {{ intentLabel(msg.meta.intent) }}</span>
              <span v-if="msg.meta.elapsed">耗时: {{ msg.meta.elapsed }}s</span>
            </div>
          </div>
          <div v-if="msg.content" class="message-actions">
            <button
              type="button"
              class="message-copy-btn"
              :title="copiedMessageIndex === i ? '已复制' : '复制文本'"
              @click="copyMessage(msg.content, i)"
            >
              <CheckCircle2 v-if="copiedMessageIndex === i" :size="13" />
              <Copy v-else :size="13" />
              {{ copiedMessageIndex === i ? '已复制' : '复制' }}
            </button>
          </div>
          </div>
        </template>

        <!-- 思考中占位：还没有任何状态步骤时的等待气泡（有步骤后由消息内面板接管） -->
        <div v-if="sending && statusSteps.length === 0 && !lastAssistantHasContent && !lastAssistantHasProgress" class="message assistant">
          <div class="message-body">
            <div class="message-text typing">思考中<span>.</span><span>.</span><span>.</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 输入区：空状态时垂直居中，有消息后固定底部 -->
    <div class="chat-input" :class="{ 'chat-input--center': messages.length === 0 && !sending }">
      <div class="chat-input-inner">
        <div v-if="selectedSkills.length" class="selected-skill-row">
          <span v-for="skill in selectedSkills" :key="skill.id" class="selected-skill-chip">
            <WandSparkles :size="12" />
            {{ skill.name }}
            <button type="button" :title="`移除 ${skill.name}`" @click="removeSkill(skill.id)">
              <X :size="12" />
            </button>
          </span>
        </div>
        <!-- 待发送图片预览 -->
        <div v-if="attachedImage" class="image-attach-preview">
          <img :src="attachedImage" alt="待发送图片" />
          <button type="button" class="image-attach-remove" title="移除图片" @click="removeImage">
            <X :size="14" />
          </button>
        </div>
        <div v-if="imageError" class="image-attach-error">{{ imageError }}</div>
        <textarea
          v-model="input"
          @keydown.enter.exact.prevent="send"
          @paste="onPaste"
          placeholder="和 EasyRAG 一起思考…"
          rows="1"
          ref="inputEl"
          @input="autoResize"
        ></textarea>
        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          class="hidden-file-input"
          @change="onFilePicked"
        />
        <div class="chat-input-actions">
          <div class="composer-control-group">
          <div ref="modelPickerEl" class="model-picker-shell">
            <button
              type="button"
              class="model-picker"
              :class="{ 'has-error': modelLoadError, open: modelMenuOpen }"
              :title="modelPickerTitle"
              :disabled="sending || modelsLoading"
              @click="modelMenuOpen = !modelMenuOpen"
            >
              <span class="model-status-dot" :class="{ ready: selectedModel?.available }"></span>
              <span class="model-picker-label">
                {{ modelsLoading ? '加载模型中…' : (selectedModel?.name || '添加自定义模型') }}
              </span>
              <ChevronDown :size="13" />
            </button>
            <div v-if="modelMenuOpen" class="model-dropdown">
              <div class="model-dropdown-title">对话模型</div>
              <button
                v-for="model in modelOptions"
                :key="model.id"
                type="button"
                class="model-dropdown-item"
                :class="{ selected: model.id === selectedModelId }"
                :disabled="!model.available"
                @click="selectModel(model)"
              >
                <span class="model-status-dot" :class="{ ready: model.available }"></span>
                <span class="model-option-copy">
                  <strong>{{ model.name }}</strong>
                  <small>{{ model.provider_type === 'local' ? '本地' : model.provider }}</small>
                </span>
                <span v-if="!model.available" class="model-unavailable">未配置</span>
                <CheckCircle2 v-else-if="model.id === selectedModelId" :size="14" />
              </button>
              <button type="button" class="model-dropdown-add" @click="openCustomModelModal">
                <Plus :size="14" /> 添加自定义模型
              </button>
            </div>
          </div>
          <div ref="skillPickerEl" class="skill-picker-shell">
            <button
              type="button"
              class="skill-picker"
              :class="{ open: skillMenuOpen, active: selectedSkills.length }"
              :disabled="sending || skillsLoading"
              @click="toggleSkillMenu"
            >
              <WandSparkles :size="14" />
              <span>{{ skillsLoading ? '加载 Skill…' : 'Skill' }}</span>
              <span v-if="selectedSkills.length" class="skill-picker-count">{{ selectedSkills.length }}</span>
              <ChevronDown :size="13" />
            </button>
            <div v-if="skillMenuOpen" class="skill-dropdown">
              <div class="skill-dropdown-head">
                <div>
                  <strong>为本次对话选择 Skill</strong>
                  <small>最多选择 {{ maxSelectedSkills }} 个</small>
                </div>
                <button type="button" title="管理 Skill" @click="openSkillConfigModal">
                  <Settings2 :size="15" />
                </button>
              </div>
              <label class="skill-search">
                <Search :size="13" />
                <input v-model="skillSearch" type="search" placeholder="搜索 Skill" />
              </label>
              <div v-if="skillLoadError" class="skill-inline-error">{{ skillLoadError }}</div>
              <div v-else class="skill-dropdown-list">
                <button
                  v-for="skill in filteredSkills"
                  :key="skill.id"
                  type="button"
                  class="skill-dropdown-item"
                  :class="{ selected: selectedSkillIds.includes(skill.id) }"
                  @click="toggleSkill(skill)"
                >
                  <span class="skill-item-icon"><WandSparkles :size="14" /></span>
                  <span class="skill-item-copy">
                    <strong>{{ skill.name }}</strong>
                    <small>{{ skill.description }}</small>
                    <em v-if="skill.tool_names?.length">{{ skill.tool_names.join(' · ') }}</em>
                  </span>
                  <CheckCircle2 v-if="selectedSkillIds.includes(skill.id)" :size="15" />
                  <span v-else class="skill-empty-check"></span>
                </button>
                <div v-if="!filteredSkills.length" class="skill-empty-state">没有匹配的 Skill</div>
              </div>
              <button type="button" class="skill-manage-entry" @click="openSkillConfigModal">
                <Settings2 :size="14" /> 配置与添加 Skill
              </button>
            </div>
          </div>
          </div>
          <!-- 右侧按钮组：图片 + 深度研究（紧贴发送按钮左侧）+ 停止/发送 -->
          <div class="composer-send-group">
            <button
              type="button"
              class="image-attach-btn"
              :class="{ active: attachedImage }"
              :disabled="sending"
              @click="fileInput?.click()"
              :title="attachedImage ? '已添加图片，点击可替换' : '粘贴或上传图片（模型支持时直接理解，否则自动 OCR 识别文字）'"
            >
              <ImageIcon :size="15" />
            </button>
            <button
              type="button"
              class="deep-research-toggle"
              :class="{ active: deepResearch }"
              :disabled="sending"
              @click="deepResearch = !deepResearch"
              :title="deepResearch ? '关闭深度研究（恢复自动模式）' : '开启深度研究：由主 Agent 调度研究子智能体，多步检索与推理，回答更深入'"
            >
              <Sparkles :size="13" :class="{ 'is-on': deepResearch }" />
              <span>深度研究</span>
            </button>
            <!-- 生成中显示"停止"按钮：终止当前对话轮（被终止的一轮不保存到记录） -->
            <button v-if="sending" type="button" class="btn-send btn-stop" @click="stopGeneration" title="停止生成">
              <Square :size="16" />
            </button>
            <button v-else @click="send" :disabled="!input.trim() || sending || !selectedModelId" class="btn-send" title="发送">
              <ArrowUp :size="16" />
            </button>
          </div>
        </div>
      </div>
    </div>
    </div><!-- /.chat-main -->

    <!-- 多智能体状态工作台：计划与 Agent 分层展示，过程产出默认折叠。 -->
    <Transition name="task-panel-drawer">
      <aside
        v-if="taskPanel.visible"
        class="task-panel"
        :class="{ 'is-resizing': panelResizing }"
        :style="{ width: taskPanelWidth + 'px' }"
      >
      <div
        class="task-panel-resizer"
        title="拖动调整状态栏宽度"
        @pointerdown="startTaskPanelResize"
      ></div>
      <div class="task-panel-header">
        <div>
          <span class="task-panel-title">
            <Loader2 v-if="taskPanel.status === 'running'" :size="14" class="spin" />
            <CheckCircle2 v-else :size="14" />
            状态 {{ taskPanelObjectCount }} 项
          </span>
          <span class="task-panel-subtitle">{{ runStateLabel }}</span>
        </div>
        <span class="task-panel-actions">
          <button class="task-panel-close" title="收起状态栏" @click="taskPanel.visible = false">
            <ChevronRight :size="14" />
            <span>收起</span>
          </button>
        </span>
      </div>
      <div class="task-panel-bar">
        <div class="task-panel-bar-fill" :style="{ width: taskProgress.pct + '%' }"></div>
      </div>

      <section class="workbench-section">
        <button class="workbench-section-header" @click="taskPanel.todosExpanded = !taskPanel.todosExpanded">
          <span><ListChecks :size="14" /> 待办</span>
          <span class="workbench-section-meta">
            {{ taskProgress.done }}/{{ taskProgress.total }}
            <ChevronDown v-if="taskPanel.todosExpanded" :size="14" />
            <ChevronRight v-else :size="14" />
          </span>
        </button>
        <div v-show="taskPanel.todosExpanded" class="todo-list">
          <div v-for="t in taskPanel.tasks" :key="'todo-' + t.task_id" class="todo-row" :class="'task-' + t.status">
            <span class="task-status-icon">
              <CheckCircle2 v-if="t.status === 'done'" :size="14" />
              <span v-else-if="t.status === 'error'" class="task-error-mark">✕</span>
              <span v-else-if="t.status === 'skipped'" class="task-skip-mark" title="已跳过">⏭</span>
              <Loader2 v-else-if="t.status === 'running'" :size="14" class="spin" />
              <span v-else class="task-pending-dot"></span>
            </span>
            <span class="todo-title">{{ t.goal }}</span>
          </div>
        </div>
      </section>

      <section class="workbench-section">
        <button class="workbench-section-header" @click="taskPanel.agentsExpanded = !taskPanel.agentsExpanded">
          <span><Bot :size="14" /> 子智能体</span>
          <span class="workbench-section-meta">
            {{ taskPanel.tasks.length }}
            <ChevronDown v-if="taskPanel.agentsExpanded" :size="14" />
            <ChevronRight v-else :size="14" />
          </span>
        </button>
        <div v-show="taskPanel.agentsExpanded" class="agent-list">
          <article v-for="t in taskPanel.tasks" :key="'agent-' + t.task_id" class="agent-card" :class="'task-' + t.status">
            <button class="agent-card-summary" @click="t.expanded = !t.expanded">
              <span class="agent-avatar"><Bot :size="14" /></span>
              <span class="agent-card-copy">
                <strong>{{ workerLabel(t.worker_hint) }}探索员</strong>
                <small>{{ t.goal }}</small>
              </span>
              <span class="agent-state">
                <CheckCircle2 v-if="t.status === 'done'" :size="14" />
                <span v-else-if="t.status === 'error'" class="task-error-mark">✕</span>
                <span v-else-if="t.status === 'skipped'" class="task-skip-mark" title="已跳过">⏭</span>
                <Loader2 v-else-if="t.status === 'running'" :size="14" class="spin" />
                <span v-else class="task-pending-dot"></span>
                <ChevronDown v-if="t.expanded" :size="14" />
                <ChevronRight v-else :size="14" />
              </span>
            </button>
            <div v-show="t.expanded" class="agent-card-detail">
              <div v-if="t.tools.length" class="task-tools">
                <div v-for="(tc, ti) in t.tools" :key="ti" class="task-tool-call">
                  <span class="task-tool-name">Call</span>
                  <span class="task-tool-args">{{ tc }}</span>
                </div>
              </div>
              <div v-if="t.output" class="agent-output">
                <span class="agent-output-label">子任务产出</span>
                <div class="agent-output-content" v-html="renderContent(t.output)"></div>
              </div>
              <div v-else-if="t.status === 'running'" class="agent-waiting">正在执行并回传结果…</div>
              <div v-else-if="t.status === 'skipped'" class="agent-waiting">已跳过（{{ t.error || '依赖任务未成功' }}）</div>
              <div v-else-if="t.status === 'pending'" class="agent-waiting">等待调度</div>
            </div>
          </article>
        </div>
      </section>
      </aside>
    </Transition>

    <Teleport to="body">
      <div v-if="customModelModalOpen" class="modal-overlay" @click.self="closeCustomModelModal">
        <div class="modal model-config-modal">
          <div class="model-config-header">
            <div>
              <h3>添加自定义模型</h3>
              <p>支持 OpenAI 兼容的本地服务或云端接口，配置会安全保存到后端。</p>
            </div>
            <button type="button" class="model-config-close" @click="closeCustomModelModal">×</button>
          </div>

          <form class="model-config-form" @submit.prevent="saveCustomModel">
            <div class="model-type-grid">
              <button
                type="button"
                :class="['model-type-card', { active: customModelForm.provider_type === 'local' }]"
                @click="setCustomModelType('local')"
              >
                <HardDrive :size="18" />
                <span><strong>本地模型</strong><small>Ollama、LM Studio 等</small></span>
              </button>
              <button
                type="button"
                :class="['model-type-card', { active: customModelForm.provider_type === 'cloud' }]"
                @click="setCustomModelType('cloud')"
              >
                <Cloud :size="18" />
                <span><strong>云端模型</strong><small>OpenAI 兼容 API</small></span>
              </button>
            </div>

            <div class="model-form-grid">
              <label>
                <span>显示名称</span>
                <input v-model="customModelForm.name" type="text" maxlength="80" placeholder="例如：本地 Qwen 32B" required />
              </label>
              <label>
                <span>供应商名称</span>
                <input v-model="customModelForm.provider_name" type="text" maxlength="80" placeholder="例如：Ollama / OpenRouter" />
              </label>
            </div>
            <label>
              <span>API Base URL</span>
              <input v-model="customModelForm.base_url" type="url" maxlength="512" :placeholder="customModelUrlPlaceholder" required />
              <small class="model-form-hint">填写 OpenAI 兼容接口根地址，通常以 /v1 结尾。</small>
            </label>
            <div class="model-form-grid">
              <label>
                <span>模型 ID</span>
                <input v-model="customModelForm.model_name" type="text" maxlength="160" placeholder="例如：qwen3:32b" required />
              </label>
              <label>
                <span>Temperature</span>
                <input v-model.number="customModelForm.temperature" type="number" min="0" max="2" step="0.1" />
              </label>
            </div>
            <label class="model-key-toggle">
              <input v-model="customModelForm.requires_api_key" type="checkbox" />
              <span>此接口需要 API Key</span>
            </label>
            <label class="model-key-toggle">
              <input v-model="customModelForm.supports_vision" type="checkbox" />
              <span>支持图片输入（多模态）</span>
            </label>
            <label v-if="customModelForm.requires_api_key">
              <span>API Key</span>
              <input v-model="customModelForm.api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="仅加密保存到后端" required />
            </label>

            <div v-if="customModelError" class="model-config-error">{{ customModelError }}</div>

            <div v-if="customModels.length" class="custom-model-list">
              <div class="custom-model-list-title">已添加</div>
              <div v-for="model in customModels" :key="model.id" class="custom-model-row">
                <span class="model-option-copy">
                  <strong>{{ model.name }}</strong>
                  <small>{{ model.provider_type === 'local' ? '本地模型' : model.provider }}</small>
                </span>
                <button type="button" title="删除模型" @click="deleteCustomModel(model)">
                  <Trash2 :size="14" />
                </button>
              </div>
            </div>

            <div class="modal-actions">
              <button type="button" class="btn-secondary" @click="closeCustomModelModal">取消</button>
              <button type="submit" class="btn-primary-sm" :disabled="savingCustomModel">
                {{ savingCustomModel ? '保存中…' : '保存并使用' }}
              </button>
            </div>
          </form>
        </div>
      </div>
    </Teleport>

    <Teleport to="body">
      <div v-if="skillConfigModalOpen" class="modal-overlay" @click.self="closeSkillConfigModal">
        <div class="modal skill-config-modal">
          <div class="skill-config-header">
            <div>
              <span class="skill-config-kicker"><WandSparkles :size="13" /> AGENT SKILLS</span>
              <h3>Skill 配置</h3>
              <p>用工作指令定义 Agent 的做事方式，并只授予完成任务需要的工具。</p>
            </div>
            <button type="button" class="model-config-close" @click="closeSkillConfigModal">×</button>
          </div>

          <div class="skill-config-layout">
            <section class="skill-library-panel">
              <div class="skill-panel-title">
                <div><strong>Skill 库</strong><small>{{ skillOptions.length }} 项</small></div>
                <button type="button" @click="startNewSkill"><Plus :size="13" /> 新建</button>
              </div>
              <div class="skill-library-list">
                <article v-for="skill in skillOptions" :key="skill.id" class="skill-library-card" :class="{ active: editingSkillId === skill.id }">
                  <button type="button" class="skill-library-main" @click="skill.can_edit ? editSkill(skill) : previewBuiltinSkill(skill)">
                    <span class="skill-item-icon"><WandSparkles :size="14" /></span>
                    <span>
                      <strong>{{ skill.name }}</strong>
                      <small>{{ skill.category }} · {{ skill.source === 'builtin' ? '内置' : '自定义' }}</small>
                    </span>
                  </button>
                  <span v-if="skill.can_edit" class="skill-library-actions">
                    <button type="button" title="编辑" @click="editSkill(skill)"><Pencil :size="13" /></button>
                    <button type="button" title="删除" @click="deleteCustomSkill(skill)"><Trash2 :size="13" /></button>
                  </span>
                </article>
              </div>
            </section>

            <form class="skill-editor" @submit.prevent="saveCustomSkill">
              <div class="skill-editor-heading">
                <div>
                  <strong>{{ skillFormReadOnly ? '内置 Skill 预览' : (editingSkillId ? '编辑自定义 Skill' : '创建自定义 Skill') }}</strong>
                  <small>{{ skillFormReadOnly ? '内置配置不可修改，可作为自定义 Skill 的参考。' : '保存后会同步到后端，并出现在对话框选择器中。' }}</small>
                </div>
                <span v-if="skillFormReadOnly" class="builtin-skill-badge"><ShieldCheck :size="12" /> 内置</span>
              </div>

              <div class="skill-form-grid">
                <label>
                  <span>名称</span>
                  <input v-model="customSkillForm.name" type="text" maxlength="80" placeholder="例如：竞品研究" :disabled="skillFormReadOnly" required />
                </label>
                <label>
                  <span>分类</span>
                  <input v-model="customSkillForm.category" type="text" maxlength="32" placeholder="例如：研究" :disabled="skillFormReadOnly" />
                </label>
              </div>
              <label>
                <span>一句话说明</span>
                <input v-model="customSkillForm.description" type="text" maxlength="300" placeholder="告诉用户这个 Skill 适合解决什么问题" :disabled="skillFormReadOnly" />
              </label>
              <label>
                <span>工作指令</span>
                <textarea v-model="customSkillForm.instructions" rows="8" maxlength="6000" placeholder="描述工作步骤、质量标准、输出格式和边界…" :disabled="skillFormReadOnly" required></textarea>
                <small class="model-form-hint">建议写清：先做什么、如何核验、最终交付什么，以及不能做什么。</small>
              </label>

              <fieldset class="skill-tool-fieldset" :disabled="skillFormReadOnly">
                <legend>允许使用的工具</legend>
                <p>未勾选的工具会在模型规划和实际调用两个阶段被拦截。</p>
                <label v-for="tool in skillTools" :key="tool.name" class="skill-tool-option" :class="{ unavailable: !tool.available }">
                  <input v-model="customSkillForm.tool_names" type="checkbox" :value="tool.name" />
                  <span>
                    <strong>{{ tool.name }}</strong>
                    <small>{{ tool.description }}</small>
                  </span>
                  <em>{{ tool.available ? '可用' : '未配置' }}</em>
                </label>
                <div v-if="!skillTools.length" class="skill-empty-state">当前没有已注册工具，Skill 仍可作为工作指令使用。</div>
              </fieldset>

              <div v-if="skillConfigError" class="model-config-error">{{ skillConfigError }}</div>
              <div class="modal-actions">
                <button type="button" class="btn-secondary" @click="closeSkillConfigModal">关闭</button>
                <button v-if="skillFormReadOnly" type="button" class="btn-primary-sm" @click="duplicateBuiltinSkill">复制为自定义</button>
                <button v-else type="submit" class="btn-primary-sm" :disabled="savingCustomSkill">
                  {{ savingCustomSkill ? '保存中…' : (editingSkillId ? '保存修改' : '创建 Skill') }}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick, onActivated, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '../stores/chat'
import { marked } from 'marked'
import {
  Asterisk,
  ArrowUp,
  ArrowUpRight,
  Bot,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Square,
  Sparkles,
  ChevronRight,
  Cloud,
  Copy,
  HardDrive,
  FileSearch2,
  Globe2,
  Image as ImageIcon,
  ListTree,
  ListChecks,
  Loader2,
  Pencil,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Trash2,
  WandSparkles,
  X,
} from 'lucide-vue-next'
import api from '../api'
import AgentActivity from '../components/AgentActivity.vue'
import ProgressJournal from '../components/ProgressJournal.vue'

// Render LLM markdown (bold, lists, links) to HTML. Links get target=_blank
// and rel=noopener so external sources open safely in a new tab.
marked.setOptions({ breaks: true, gfm: true })

function renderContent(text) {
  if (!text) return ''
  const html = marked.parse(text)
  // Only allow http/https links to open externally; force safe attrs.
  return html.replace(
    /<a\s+href="(https?:\/\/[^"]+)"([^>]*)>/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer"$2>'
  )
}

// 意图 → 中文展示名（与后端 intent_done 步骤文案保持一致）
const INTENT_LABELS = {
  knowledge_qa: '知识库问答',
  tool_use: '联网/工具查询',
  complex_task: '复杂任务',
  direct: '直接回答',
  chitchat: '闲聊',
  multi_agent: '多智能体',
  deepagents: '智能体',
}
function intentLabel(intent) {
  return INTENT_LABELS[intent] || intent
}

// Agent 路径徽标：本轮实际走了哪条执行链路
const MODE_LABELS = {
  deepagents: 'DeepAgents',
  multi: '多智能体',
  single: '单 Agent',
}
function modeLabel(mode) {
  return MODE_LABELS[mode] || mode || ''
}

const chatStore = useChatStore()
const router = useRouter()
const messages = ref([])

// 点击知识库引用 → 跳转到知识库页并定位到对应文档详情
function goToSource(s) {
  router.push({
    path: '/knowledge',
    query: { kb: s.knowledge_base_id, file: s.file_id },
  })
}

const input = ref('')
const sending = ref(false)
const chatStarters = [
  { icon: FileSearch2, title: '检索资料', description: '从知识库定位依据', prompt: '请帮我从知识库中查找并总结最相关的资料。' },
  { icon: ListTree, title: '梳理脉络', description: '把复杂内容变清晰', prompt: '请帮我梳理这个主题的关键概念、关系与结论。' },
  { icon: Globe2, title: '深度研究', description: '结合网络交叉验证', prompt: '请围绕这个主题进行深度研究，并给出有来源的结论。' },
]

function useStarter(prompt) {
  input.value = prompt
  nextTick(() => inputEl.value?.focus())
}
// 当前轮请求的 AbortController（"停止生成"用；终止的轮次不保存到记录）
let currentAbort = null
// 深度研究开关：选中后本轮请求走 DeepAgents 工作流（deep_research=true）
const deepResearch = ref(false)
const conversationId = ref(null)
const msgContainer = ref(null)
const inputEl = ref(null)
const copiedMessageIndex = ref(null)
let copyFeedbackTimer = null

// ── 图片输入：粘贴 / 上传，随对话请求以 data URL 发出 ──
const attachedImage = ref(null)        // 当前待发送图片的 data URL
const imageError = ref('')
const fileInput = ref(null)
const MAX_IMAGE_BYTES = 8 * 1024 * 1024  // 8MB 上限，避免请求体过大

function _readImageFile(file) {
  if (!file) return
  imageError.value = ''
  if (!file.type.startsWith('image/')) {
    imageError.value = '请选择图片文件'
    return
  }
  if (file.size > MAX_IMAGE_BYTES) {
    imageError.value = '图片不能超过 8MB'
    return
  }
  const reader = new FileReader()
  reader.onload = () => { attachedImage.value = reader.result }
  reader.onerror = () => { imageError.value = '图片读取失败' }
  reader.readAsDataURL(file)
}

function onPaste(event) {
  const items = event.clipboardData?.items
  if (!items) return
  for (const it of items) {
    if (it.type.startsWith('image/')) {
      event.preventDefault()
      _readImageFile(it.getAsFile())
      return
    }
  }
}

function onFilePicked(event) {
  const f = event.target.files?.[0]
  _readImageFile(f)
  event.target.value = ''  // 允许重复选同一文件
}

function removeImage() {
  attachedImage.value = null
  imageError.value = ''
}

async function copyMessage(content, index) {
  if (!content) return
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(content)
    } else {
      const textarea = document.createElement('textarea')
      textarea.value = content
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      textarea.remove()
    }
    copiedMessageIndex.value = index
    if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer)
    copyFeedbackTimer = window.setTimeout(() => {
      copiedMessageIndex.value = null
      copyFeedbackTimer = null
    }, 1600)
  } catch {
    copiedMessageIndex.value = null
  }
}

const TASK_PANEL_WIDTH_KEY = 'easyrag-task-panel-width'
const savedTaskPanelWidth = Number(localStorage.getItem(TASK_PANEL_WIDTH_KEY))
const taskPanelWidth = ref(
  Number.isFinite(savedTaskPanelWidth)
    ? Math.min(520, Math.max(280, savedTaskPanelWidth))
    : 340
)
const panelResizing = ref(false)
let panelResizeMoveHandler = null
let panelResizeUpHandler = null

function stopTaskPanelResize() {
  if (!panelResizing.value) return
  panelResizing.value = false
  document.body.classList.remove('task-panel-resizing')
  if (panelResizeMoveHandler) window.removeEventListener('pointermove', panelResizeMoveHandler)
  if (panelResizeUpHandler) window.removeEventListener('pointerup', panelResizeUpHandler)
  if (panelResizeUpHandler) window.removeEventListener('pointercancel', panelResizeUpHandler)
  panelResizeMoveHandler = null
  panelResizeUpHandler = null
  localStorage.setItem(TASK_PANEL_WIDTH_KEY, String(taskPanelWidth.value))
}

function startTaskPanelResize(event) {
  if (event.button !== 0) return
  event.preventDefault()
  const startX = event.clientX
  const startWidth = taskPanelWidth.value
  panelResizing.value = true
  document.body.classList.add('task-panel-resizing')
  panelResizeMoveHandler = moveEvent => {
    const nextWidth = startWidth + startX - moveEvent.clientX
    taskPanelWidth.value = Math.min(520, Math.max(280, Math.round(nextWidth)))
  }
  panelResizeUpHandler = stopTaskPanelResize
  window.addEventListener('pointermove', panelResizeMoveHandler)
  window.addEventListener('pointerup', panelResizeUpHandler, { once: true })
  window.addEventListener('pointercancel', panelResizeUpHandler, { once: true })
}

// 对话模型目录由后端环境配置生成；浏览器只保存公开 model_id，不接触供应商密钥。
const MODEL_STORAGE_KEY = 'easyrag-chat-model-id'
const modelOptions = ref([])
const selectedModelId = ref(localStorage.getItem(MODEL_STORAGE_KEY) || '')
const modelsLoading = ref(true)
const modelLoadError = ref('')
const modelMenuOpen = ref(false)
const modelPickerEl = ref(null)
const customModelModalOpen = ref(false)
const savingCustomModel = ref(false)
const customModelError = ref('')
const customModelForm = reactive({
  provider_type: 'local',
  name: '',
  provider_name: 'Ollama',
  base_url: 'http://localhost:11434/v1',
  model_name: '',
  api_key: '',
  requires_api_key: false,
  temperature: 0,
  supports_vision: false,
})
const selectedModel = computed(() => (
  modelOptions.value.find(model => model.id === selectedModelId.value) || null
))
const customModels = computed(() => (
  modelOptions.value.filter(model => model.source === 'custom')
))
const customModelUrlPlaceholder = computed(() => (
  customModelForm.provider_type === 'local'
    ? 'http://localhost:11434/v1'
    : 'https://api.example.com/v1'
))
const modelPickerTitle = computed(() => {
  if (modelLoadError.value) return modelLoadError.value
  if (selectedModel.value) return `${selectedModel.value.provider} · ${selectedModel.value.name}`
  return '添加自定义模型'
})

async function loadModels(preferredModelId = '') {
  modelsLoading.value = true
  modelLoadError.value = ''
  try {
    const data = await api.get('/chat/models')
    modelOptions.value = data.models || []
    const available = modelOptions.value.filter(model => model.available)
    const requested = available.find(model => model.id === preferredModelId)
    const saved = available.find(model => model.id === selectedModelId.value)
    const preferred = available.find(model => model.id === data.default_model_id)
    selectedModelId.value = (requested || saved || preferred || available[0])?.id || ''
  } catch (error) {
    modelOptions.value = []
    selectedModelId.value = ''
    modelLoadError.value = `模型列表加载失败：${error.message}`
  } finally {
    modelsLoading.value = false
  }
}

watch(selectedModelId, (modelId) => {
  if (modelId) localStorage.setItem(MODEL_STORAGE_KEY, modelId)
  else localStorage.removeItem(MODEL_STORAGE_KEY)
})

function selectModel(model) {
  if (!model.available) return
  selectedModelId.value = model.id
  modelMenuOpen.value = false
}

function setCustomModelType(type) {
  customModelForm.provider_type = type
  customModelForm.api_key = ''
  if (type === 'local') {
    customModelForm.provider_name = 'Ollama'
    customModelForm.base_url = 'http://localhost:11434/v1'
    customModelForm.requires_api_key = false
    customModelForm.temperature = 0
  } else {
    customModelForm.provider_name = ''
    customModelForm.base_url = ''
    customModelForm.requires_api_key = true
    customModelForm.temperature = 0.7
  }
}

function openCustomModelModal() {
  modelMenuOpen.value = false
  customModelError.value = ''
  Object.assign(customModelForm, {
    provider_type: 'local',
    name: '',
    provider_name: 'Ollama',
    base_url: 'http://localhost:11434/v1',
    model_name: '',
    api_key: '',
    requires_api_key: false,
    temperature: 0,
    supports_vision: false,
  })
  customModelModalOpen.value = true
}

function closeCustomModelModal() {
  if (savingCustomModel.value) return
  customModelModalOpen.value = false
  customModelError.value = ''
}

async function saveCustomModel() {
  if (savingCustomModel.value) return
  savingCustomModel.value = true
  customModelError.value = ''
  try {
    const created = await api.post('/chat/models', {
      provider_type: customModelForm.provider_type,
      name: customModelForm.name.trim(),
      provider_name: customModelForm.provider_name.trim(),
      base_url: customModelForm.base_url.trim(),
      model_name: customModelForm.model_name.trim(),
      api_key: customModelForm.api_key.trim(),
      requires_api_key: customModelForm.requires_api_key,
      temperature: customModelForm.temperature,
      supports_vision: customModelForm.supports_vision,
    })
    await loadModels(created.id)
    customModelModalOpen.value = false
  } catch (error) {
    customModelError.value = error.response?.data?.detail || error.message || '保存失败'
  } finally {
    savingCustomModel.value = false
  }
}

async function deleteCustomModel(model) {
  if (!window.confirm(`确定删除自定义模型“${model.name}”吗？`)) return
  customModelError.value = ''
  try {
    await api.delete(`/chat/models/${model.id}`)
    if (selectedModelId.value === model.id) selectedModelId.value = ''
    await loadModels()
  } catch (error) {
    customModelError.value = error.response?.data?.detail || error.message || '删除失败'
  }
}

function closeModelMenuOnOutsideClick(event) {
  if (modelPickerEl.value && !modelPickerEl.value.contains(event.target)) {
    modelMenuOpen.value = false
  }
  if (skillPickerEl.value && !skillPickerEl.value.contains(event.target)) {
    skillMenuOpen.value = false
  }
}

// Skill 目录和选择结果都只保存公开 ID；指令与工具权限由后端在每次请求时重新解析。
const SKILL_STORAGE_KEY = 'easyrag-chat-skill-ids'
let storedSkillIds = []
try {
  const parsed = JSON.parse(localStorage.getItem(SKILL_STORAGE_KEY) || '[]')
  if (Array.isArray(parsed)) storedSkillIds = parsed.filter(item => typeof item === 'string')
} catch { /* 损坏的本地缓存直接忽略 */ }

const skillOptions = ref([])
const skillTools = ref([])
const selectedSkillIds = ref(storedSkillIds)
const maxSelectedSkills = ref(3)
const skillsLoading = ref(true)
const skillLoadError = ref('')
const skillMenuOpen = ref(false)
const skillPickerEl = ref(null)
const skillSearch = ref('')
const skillConfigModalOpen = ref(false)
const savingCustomSkill = ref(false)
const skillConfigError = ref('')
const editingSkillId = ref('')
const skillFormReadOnly = ref(false)
const customSkillForm = reactive({
  name: '',
  description: '',
  instructions: '',
  tool_names: [],
  category: '自定义',
  icon: 'sparkles',
})

const selectedSkills = computed(() => selectedSkillIds.value
  .map(id => skillOptions.value.find(skill => skill.id === id))
  .filter(Boolean))
const filteredSkills = computed(() => {
  const query = skillSearch.value.trim().toLowerCase()
  if (!query) return skillOptions.value
  return skillOptions.value.filter(skill => [
    skill.name,
    skill.description,
    skill.category,
    ...(skill.tool_names || []),
  ].some(value => String(value || '').toLowerCase().includes(query)))
})

watch(selectedSkillIds, (ids) => {
  localStorage.setItem(SKILL_STORAGE_KEY, JSON.stringify(ids))
}, { deep: true })

async function loadSkills(preferredSkillId = '') {
  skillsLoading.value = true
  skillLoadError.value = ''
  try {
    const data = await api.get('/chat/skills')
    skillOptions.value = data.skills || []
    skillTools.value = data.tools || []
    maxSelectedSkills.value = data.max_selected || 3
    const validIds = new Set(skillOptions.value.map(skill => skill.id))
    selectedSkillIds.value = selectedSkillIds.value
      .filter(id => validIds.has(id))
      .slice(0, maxSelectedSkills.value)
    if (
      preferredSkillId
      && validIds.has(preferredSkillId)
      && !selectedSkillIds.value.includes(preferredSkillId)
      && selectedSkillIds.value.length < maxSelectedSkills.value
    ) {
      selectedSkillIds.value = [...selectedSkillIds.value, preferredSkillId]
    }
  } catch (error) {
    skillOptions.value = []
    skillTools.value = []
    selectedSkillIds.value = []
    skillLoadError.value = error.response?.data?.detail || `Skill 列表加载失败：${error.message}`
  } finally {
    skillsLoading.value = false
  }
}

function toggleSkillMenu() {
  modelMenuOpen.value = false
  skillMenuOpen.value = !skillMenuOpen.value
}

function toggleSkill(skill) {
  if (selectedSkillIds.value.includes(skill.id)) {
    removeSkill(skill.id)
    return
  }
  if (selectedSkillIds.value.length >= maxSelectedSkills.value) return
  selectedSkillIds.value = [...selectedSkillIds.value, skill.id]
}

function removeSkill(skillId) {
  selectedSkillIds.value = selectedSkillIds.value.filter(id => id !== skillId)
}

function assignSkillForm(skill = null) {
  Object.assign(customSkillForm, {
    name: skill?.name || '',
    description: skill?.description || '',
    instructions: skill?.instructions || '',
    tool_names: [...(skill?.tool_names || [])],
    category: skill?.category || '自定义',
    icon: skill?.icon || 'sparkles',
  })
}

function startNewSkill() {
  editingSkillId.value = ''
  skillFormReadOnly.value = false
  skillConfigError.value = ''
  assignSkillForm()
}

function previewBuiltinSkill(skill) {
  editingSkillId.value = skill.id
  skillFormReadOnly.value = true
  skillConfigError.value = ''
  assignSkillForm(skill)
}

function editSkill(skill) {
  editingSkillId.value = skill.id
  skillFormReadOnly.value = false
  skillConfigError.value = ''
  assignSkillForm(skill)
}

function duplicateBuiltinSkill() {
  customSkillForm.name = `${customSkillForm.name} 副本`
  editingSkillId.value = ''
  skillFormReadOnly.value = false
  skillConfigError.value = ''
}

function openSkillConfigModal() {
  skillMenuOpen.value = false
  skillConfigModalOpen.value = true
  const firstCustom = skillOptions.value.find(skill => skill.can_edit)
  if (firstCustom) editSkill(firstCustom)
  else startNewSkill()
}

function closeSkillConfigModal() {
  if (savingCustomSkill.value) return
  skillConfigModalOpen.value = false
  skillConfigError.value = ''
}

async function saveCustomSkill() {
  if (savingCustomSkill.value || skillFormReadOnly.value) return
  savingCustomSkill.value = true
  skillConfigError.value = ''
  const payload = {
    name: customSkillForm.name.trim(),
    description: customSkillForm.description.trim(),
    instructions: customSkillForm.instructions.trim(),
    tool_names: [...customSkillForm.tool_names],
    category: customSkillForm.category.trim() || '自定义',
    icon: customSkillForm.icon || 'sparkles',
  }
  try {
    const saved = editingSkillId.value
      ? await api.put(`/chat/skills/${editingSkillId.value}`, payload)
      : await api.post('/chat/skills', payload)
    await loadSkills(saved.id)
    editSkill(skillOptions.value.find(skill => skill.id === saved.id) || saved)
  } catch (error) {
    skillConfigError.value = error.response?.data?.detail || error.message || 'Skill 保存失败'
  } finally {
    savingCustomSkill.value = false
  }
}

async function deleteCustomSkill(skill) {
  if (!window.confirm(`确定删除自定义 Skill“${skill.name}”吗？`)) return
  skillConfigError.value = ''
  try {
    await api.delete(`/chat/skills/${skill.id}`)
    removeSkill(skill.id)
    await loadSkills()
    startNewSkill()
  } catch (error) {
    skillConfigError.value = error.response?.data?.detail || error.message || 'Skill 删除失败'
  }
}

// 状态步骤面板（思考过程时间线）
// statusSteps 只是当前轮次的缓冲——status 事件实时落到当前 assistant 消息的
// msg.steps 上（随消息保留，渲染在答案上方，不会被下一轮清空覆盖）
const statusSteps = ref([])

// 侧边任务进度面板（多智能体）：子任务清单 + 每个子任务的状态
const taskPanel = ref({
  visible: false,
  run_id: '',
  status: 'idle',
  todosExpanded: true,
  agentsExpanded: true,
  tasks: [],
})
function emptyTaskPanel() {
  return {
    visible: false,
    run_id: '',
    status: 'idle',
    todosExpanded: true,
    agentsExpanded: true,
    tasks: [],
  }
}
function findTask(taskId) {
  return taskPanel.value.tasks.find(t => t.task_id === taskId)
}
function setTaskStatus(taskId, status) {
  const t = findTask(taskId)
  if (t) t.status = status
}

// worker 名 → 友好标签
const WORKER_LABELS = { rag: '知识库', legal: '法律', code: '代码' }
function workerLabel(hint) {
  return WORKER_LABELS[hint] || hint || '通用'
}

// 任务面板进度：已完成数 / 总数
const taskProgress = computed(() => {
  const tasks = taskPanel.value.tasks
  if (!tasks.length) return { done: 0, total: 0, pct: 0 }
  const done = tasks.filter(t => t.status === 'done' || t.status === 'error').length
  return { done, total: tasks.length, pct: Math.round(done / tasks.length * 100) }
})
const taskPanelObjectCount = computed(() => taskPanel.value.tasks.length * 2)
const runStateLabel = computed(() => {
  if (taskPanel.value.status === 'running') return '执行中'
  if (taskPanel.value.status === 'failed') return '部分任务失败'
  if (taskPanel.value.status === 'cancelled') return '已取消'
  return taskPanel.value.tasks.length ? '已完成' : '等待计划'
})

function frontendTaskStatus(status) {
  if (status === 'completed' || status === 'done' || status === 'done_with_concerns') return 'done'
  if (status === 'failed' || status === 'error' || status === 'blocked') return 'error'
  if (status === 'running') return 'running'
  if (status === 'skipped') return 'skipped'
  return 'pending'
}

// 子任务产出 → 主对话框只展示总结性一句话（全文在右侧任务面板）
function summarizeWorkerOutput(text) {
  const flat = String(text || '').replace(/\s+/g, ' ').trim()
  if (!flat) return ''
  return flat.length > 120 ? `${flat.slice(0, 120)}…` : flat
}

function taskFromRun(task, agentRuns) {
  const agent = (agentRuns || []).find(item => item.task_id === task.task_id)
  return {
    task_id: task.task_id,
    goal: task.goal,
    worker_hint: task.worker_hint || agent?.worker_name || '',
    status: frontendTaskStatus(task.status),
    tools: [],
    output: agent?.output_summary || '',
    error: task.error_summary || agent?.error_summary || '',
    expanded: false,
  }
}

// 流式进行中: 最后一条 assistant 消息是否已开始收到内容
const lastAssistantHasContent = computed(() => {
  const last = messages.value[messages.value.length - 1]
  return !!(last && last.role === 'assistant' && last.content)
})

const lastAssistantHasProgress = computed(() => {
  const last = messages.value[messages.value.length - 1]
  return !!(last && last.role === 'assistant' && last.progressSummaries?.length)
})

function scrollBottom() {
  nextTick(() => {
    if (msgContainer.value) {
      msgContainer.value.scrollTop = msgContainer.value.scrollHeight
    }
  })
}

// 输入框随内容自动撑高（超单行时），上限由 CSS max-height 控制
function autoResize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${el.scrollHeight}px`
}

function resetInputHeight() {
  if (inputEl.value) inputEl.value.style.height = 'auto'
}

// 消息时间戳：当天显示 HH:MM，跨天显示 MM-DD HH:MM
function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const pad = n => String(n).padStart(2, '0')
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  const now = new Date()
  const sameDay = d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate()
  return sameDay ? hm : `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}`
}

// 时间分隔条: 首条消息显示; 之后距上一条超过 10 分钟再显示当前时间
const TIME_SEPARATOR_GAP_MS = 10 * 60 * 1000
function shouldShowTimeSeparator(i) {
  if (i === 0) return true
  const prev = messages.value[i - 1]
  const curr = messages.value[i]
  if (!prev?.ts || !curr?.ts) return false
  return curr.ts - prev.ts > TIME_SEPARATOR_GAP_MS
}

// ── 监听会话切换，从 DB 加载历史 ─────────────────────────────────
watch(() => chatStore.activeConversationId, async (newId, oldId) => {
  if (newId === oldId) return
  conversationId.value = newId
  messages.value = []
  input.value = ''
  // 切换会话 → 清空上一会话的任务面板（否则面板残留 pin 在右边）
  taskPanel.value = emptyTaskPanel()

  if (newId) {
    try {
      const res = await api.get(`/chat/conversations/${newId}/history`)
      // 历史消息 meta 中持久化了 sources 和 steps，重载时还原引用块 + 思考过程
      messages.value = (res.messages || []).map(m => ({
        role: m.role,
        content: m.content,
        image: m.image || null,
        sources: m.meta?.sources || [],
        skills: m.meta?.skills || [],
        meta: (m.meta?.agent_mode || m.meta?.intent || m.meta?.model_name || m.meta?.run_id || m.meta?.skills?.length) ? {
          agentMode: m.meta?.agent_mode || '',
          intent: m.meta?.intent || '',
          modelName: m.meta?.model_name || '',
          runId: m.meta?.run_id || '',
          skillNames: (m.meta?.skills || []).map(skill => skill.name),
        } : null,
        steps: m.meta?.steps || [],
        stepsExpanded: false,
        stepsLoading: false,
        progressSummaries: m.meta?.progress_summaries || [],
        // 中间产出（检索片段/工具结果/思维链）+ 多智能体子任务产出总结，历史回放
        artifacts: (m.meta?.artifacts || []).concat(
          (m.meta?.worker_outputs || []).map((wo, wi) => ({
            id: `worker-${wo.task_id || wi}`,
            kind: 'worker',
            stage: 'synthesize',
            title: `子任务 ${wo.task_id || wi}（${wo.worker || 'worker'}）完成`,
            content: wo.summary || summarizeWorkerOutput(wo.content),
            streaming: false,
          }))
        ),
        time: formatTime(m.created_at),
        ts: m.created_at ? Date.parse(m.created_at) : null,
      }))
      try {
        const runData = await api.get(`/chat/conversations/${newId}/runs`)
        const latestRun = runData.runs?.[0]
        if (latestRun?.tasks?.length) {
          taskPanel.value = {
            visible: false,
            run_id: latestRun.id,
            status: latestRun.status,
            todosExpanded: true,
            agentsExpanded: true,
            tasks: latestRun.tasks.map(task => taskFromRun(task, latestRun.agent_runs)),
          }
        }
      } catch { /* 旧会话可能没有 Run，保持无面板 */ }
      scrollBottom()
    } catch { /* 忽略 */ }
  }
  nextTick(() => inputEl.value?.focus())
}, { immediate: true })

async function send() {
  const text = input.value.trim()
  if (!text || sending.value) return
  const requestSkills = selectedSkills.value.map(skill => ({ id: skill.id, name: skill.name }))
  input.value = ''
  resetInputHeight()
  sending.value = true
  // 停止生成：终止当前对话轮（被终止的一轮后端不保存到记录）
  currentAbort = new AbortController()

  const userTs = Date.now()
  messages.value.push({
    role: 'user',
    content: text,
    image: attachedImage.value || null,
    skills: requestSkills,
    deepResearch: deepResearch.value,
    time: formatTime(new Date(userTs).toISOString()),
    ts: userTs,
    enter: true, // 新消息进入动画（历史加载不带动画）
  })
  // 先插入一条空的 assistant 消息, 流式 delta 逐步填充其 content
  // steps: 本轮思考过程（绑定在这条消息上，不会被下一轮覆盖）
  // 注意: assistant 的时间戳独立取"当前时刻"——不要复用 user 的 ts,
  // 否则同秒差值为 0, 分隔条条件 >10min 永不成立, 回复后就看不到新时间了。
  const asstTs = Date.now()
  messages.value.push({
    role: 'assistant',
    content: '',
    sources: [],
    meta: {
      modelName: selectedModel.value?.name || '',
      skillNames: requestSkills.map(skill => skill.name),
    },
    steps: [],
    stepsExpanded: false,
    stepsLoading: true,
    error: '',
    artifacts: [],
    progressSummaries: [],
    time: formatTime(new Date(asstTs).toISOString()),
    ts: asstTs,
    enter: true,
  })
  const msgIndex = messages.value.length - 1
  scrollBottom()

  let gotError = ''
  // 重置当前轮次的状态缓冲 + 任务面板
  statusSteps.value = []
  taskPanel.value = emptyTaskPanel()

  try {
    await api.streamChat('/chat/stream', {
      query: text,
      conversation_id: conversationId.value,
      model_id: selectedModelId.value,
      skill_ids: requestSkills.map(skill => skill.id),
      deep_research: deepResearch.value,
      image: attachedImage.value || undefined,
    }, (ev) => {
      if (ev.type === 'conversation_id') {
        conversationId.value = ev.conversation_id
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = {
          ...m,
          meta: {
            ...(m.meta || {}),
            agentMode: ev.agent_mode || m.meta?.agentMode || '',
            runId: ev.run_id || m.meta?.runId || '',
            modelName: ev.model_name || m.meta?.modelName || '',
            skillNames: (ev.skills || requestSkills).map(skill => skill.name),
          },
        }
        taskPanel.value.run_id = ev.run_id || ''
      } else if (ev.type === 'sub_tasks') {
        // 拆解完成：初始化侧边任务面板的待办清单（全部 pending）
        taskPanel.value = {
          visible: true,
          run_id: ev.run_id || taskPanel.value.run_id || '',
          status: 'running',
          todosExpanded: true,
          agentsExpanded: true,
          tasks: (ev.tasks || []).map(t => ({
            task_id: t.task_id,
            goal: t.goal,
            worker_hint: t.worker_hint,
            status: 'pending',
            tools: [],
            output: '',
            error: '',
            expanded: false,
          })),
        }
      } else if (ev.type === 'tool_call') {
        const task = findTask(ev.task_id)
        if (task) task.tools.push(ev.detail)
      } else if (ev.type === 'progress_summary') {
        const pm = messages.value[msgIndex]
        const list = [...(pm.progressSummaries || [])]
        if (!ev.id || !list.some(item => item.id === ev.id)) {
          list.push({
            id: ev.id || `progress-${Date.now()}-${list.length}`,
            sequence: ev.sequence || list.length + 1,
            phase: ev.phase || 'info',
            status: ev.status || 'running',
            text: ev.text || '',
            created_at: ev.created_at || Date.now(),
          })
          messages.value[msgIndex] = { ...pm, progressSummaries: list }
          scrollBottom()
        }
      } else if (ev.type === 'status') {
        // 状态事件：落到当前 assistant 消息的 steps（随消息保留）
        // _ts 记录到达时刻，AgentActivity 用它计算每步耗时与总耗时
        const st = { step: ev.step, detail: ev.detail, _ts: Date.now() }
        statusSteps.value.push(st)
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = { ...m, steps: [...(m.steps || []), st] }
        if (ev.step === 'task_started') setTaskStatus(ev.task_id, 'running')
        scrollBottom()
      } else if (ev.type === 'worker_output') {
        // 过程产出全文进入右侧工作台；主对话框只放一行总结性描述
        const task = findTask(ev.task_id)
        if (task) {
          task.output = ev.content || ''
          task.error = (ev.status === 'error' || ev.status === 'skipped')
            ? (ev.error || ev.content)
            : ''
          task.status = frontendTaskStatus(ev.status)
        }
        const wm = messages.value[msgIndex]
        const workerId = `worker-${ev.task_id}`
        if (wm && !(wm.artifacts || []).some(a => a.id === workerId)) {
          messages.value[msgIndex] = {
            ...wm,
            artifacts: [...(wm.artifacts || []), {
              id: workerId,
              kind: 'worker',
              stage: 'synthesize',
              title: `子任务 ${ev.task_id}（${ev.worker || 'worker'}）完成`,
              content: ev.summary || summarizeWorkerOutput(ev.content),
              streaming: false,
            }],
          }
          scrollBottom()
        }
      } else if (ev.type === 'artifact') {
        // 中间产出实时流：检索片段/工具结果/思维链（thinking 增量按 id 追加）
        const am = messages.value[msgIndex]
        const list = [...(am.artifacts || [])]
        const last = list[list.length - 1]
        if (ev.streaming && last && ev.id && last.id === ev.id) {
          last.content += ev.content || ''
          messages.value[msgIndex] = { ...am, artifacts: list }
        } else if (ev.streaming === false && last && ev.id && last.id === ev.id) {
          last.streaming = false
          messages.value[msgIndex] = { ...am, artifacts: list }
        } else {
          list.push({
            id: ev.id || `art-${Date.now()}-${list.length}`,
            kind: ev.kind || 'info',
            stage: ev.stage || '',
            title: ev.title || '',
            content: ev.content || '',
            streaming: !!ev.streaming,
          })
          messages.value[msgIndex] = { ...am, artifacts: list }
        }
        scrollBottom()
      } else if (ev.type === 'delta') {
        // 触发响应式更新: 替换数组元素
        const m = messages.value[msgIndex]
        m.content += ev.content
        messages.value[msgIndex] = { ...m }
        scrollBottom()
      } else if (ev.type === 'done') {
        const m = messages.value[msgIndex]
        messages.value[msgIndex] = {
          ...m,
          sources: ev.sources || [],
          meta: {
            intent: ev.intent,
            agentMode: ev.agent_mode || m.meta?.agentMode || '',
            elapsed: ev.elapsed_seconds,
            runId: ev.run_id || m.meta?.runId || '',
            modelName: ev.model_name || m.meta?.modelName || '',
            skillNames: (ev.skills || requestSkills).map(skill => skill.name),
          },
          // 优先保留客户端实时收到的 steps（带 _ts 可算耗时）；
          // done 携带的完整 steps 仅在流式中断时兜底补齐
          steps: (m.steps && m.steps.length) ? m.steps : (ev.steps || []),
          artifacts: (m.artifacts && m.artifacts.length) ? m.artifacts : (ev.artifacts || []),
          progressSummaries: (m.progressSummaries && m.progressSummaries.length)
            ? m.progressSummaries
            : (ev.progress_summaries || []),
          stepsLoading: false,
        }
        taskPanel.value.status = taskPanel.value.tasks.some(t => t.status === 'error')
          ? 'failed'
          : 'completed'
        // DeepAgents 委派落库在流结束后才创建 run：done 携带 run_id 时回填面板/消息元数据
        if (ev.run_id) {
          taskPanel.value.run_id = ev.run_id
          const dm = messages.value[msgIndex]
          messages.value[msgIndex] = {
            ...dm,
            meta: { ...(dm.meta || {}), runId: ev.run_id },
          }
        }
      } else if (ev.type === 'error') {
        gotError = ev.detail || '生成失败'
        const em = messages.value[msgIndex]
        messages.value[msgIndex] = { ...em, error: gotError, stepsLoading: false }
        if (taskPanel.value.tasks.length) taskPanel.value.status = 'failed'
      }
    }, { signal: currentAbort.signal })

    if (gotError) {
      const m = messages.value[msgIndex]
      messages.value[msgIndex] = { ...m, content: m.content || `❌ ${gotError}`, stepsLoading: false, error: gotError }
    }
    // 刷新侧边栏列表
    await chatStore.refreshAfterSend(conversationId.value)
  } catch (e) {
    const m = messages.value[msgIndex]
    // 停止生成：AbortError 属正常终止（后端不保存本轮），非错误
    const aborted = e && (e.name === 'AbortError' || /abort/i.test(String(e.message || '')))
    messages.value[msgIndex] = {
      ...m,
      content: aborted ? (m.content || '') : (m.content || `❌ 请求失败: ${e.message}`),
      stopped: aborted || undefined,
      stepsLoading: false,
      error: aborted ? undefined : e.message,
    }
  } finally {
    sending.value = false
    if (currentAbort) { currentAbort = null }
    attachedImage.value = null
    imageError.value = ''
    scrollBottom()
    nextTick(() => inputEl.value?.focus())
  }
}

// 停止生成：终止当前对话轮。后端收到客户端断开后会删除该轮记录
// （用户消息 + 新建空会话），前端本地消息标记 stopped 表示"已停止"。
function stopGeneration() {
  if (currentAbort) currentAbort.abort()
}

onActivated(() => {
  // 从知识库页切回时滚动到底部并恢复焦点
  scrollBottom()
  nextTick(() => inputEl.value?.focus())
})

onMounted(() => {
  loadModels()
  loadSkills()
  document.addEventListener('click', closeModelMenuOnOutsideClick)
})

onUnmounted(() => {
  document.removeEventListener('click', closeModelMenuOnOutsideClick)
  stopTaskPanelResize()
  if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer)
})
</script>
