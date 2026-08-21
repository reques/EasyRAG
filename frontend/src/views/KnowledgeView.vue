<template>
  <div class="knowledge-view kbw-shell">
    <template v-if="!activeKb">
      <div class="kbw-catalog-scroll">
        <header class="kbw-page-header">
          <div>
            <span class="kbw-eyebrow">KNOWLEDGE WORKSPACE</span>
            <h1>知识库</h1>
            <p>集中管理可供 Agent 查阅的资料，并为检索与评估流程做好准备。</p>
          </div>
          <button class="kbw-primary-button" @click="showCreate = true">
            <Plus :size="15" /> 新建知识库
          </button>
        </header>

        <section class="kbw-overview-grid" aria-label="知识库概览">
          <article>
            <span>知识库总数</span>
            <strong>{{ kbList.length }}</strong>
            <small>当前账号可访问</small>
          </article>
          <article>
            <span>最近创建</span>
            <strong>{{ recentlyCreatedCount }}</strong>
            <small>近 30 天</small>
          </article>
          <article>
            <span>支持格式</span>
            <strong>9</strong>
            <small>文档与常见图片</small>
          </article>
        </section>

        <section class="kbw-catalog-section">
          <div class="kbw-section-heading">
            <div>
              <h2>全部知识库</h2>
              <p>{{ filteredKbs.length }} 个结果</p>
            </div>
            <label class="kbw-search-box">
              <Search :size="15" />
              <input v-model="catalogQuery" type="search" placeholder="搜索名称或描述" />
            </label>
          </div>

          <div v-if="loading" class="kbw-state-card">
            <LoaderCircle :size="22" class="spin" />
            <strong>正在加载知识库</strong>
            <span>正在读取当前账号可访问的目录。</span>
          </div>
          <div v-else-if="catalogError" class="kbw-state-card is-error">
            <CircleAlert :size="22" />
            <strong>知识库加载失败</strong>
            <span>{{ catalogError }}</span>
            <button class="kbw-secondary-button" @click="loadKbs">重新加载</button>
          </div>
          <div v-else-if="kbList.length === 0" class="kbw-state-card">
            <Database :size="24" />
            <strong>还没有知识库</strong>
            <span>新建一个知识库，然后上传可供 Agent 检索的资料。</span>
            <button class="kbw-primary-button" @click="showCreate = true">
              <Plus :size="14" /> 新建知识库
            </button>
          </div>
          <div v-else-if="filteredKbs.length === 0" class="kbw-state-card">
            <SearchX :size="24" />
            <strong>没有匹配结果</strong>
            <span>换一个关键词，或清空当前搜索条件。</span>
            <button class="kbw-secondary-button" @click="catalogQuery = ''">清空搜索</button>
          </div>
          <div v-else class="kbw-library-grid">
            <button
              v-for="kb in filteredKbs"
              :key="kb.id"
              class="kbw-library-card"
              @click="selectKb(kb)"
            >
              <span class="kbw-library-icon"><Database :size="20" /></span>
              <span class="kbw-library-card-main">
                <span class="kbw-library-title-row">
                  <strong>{{ kb.name }}</strong>
                  <ChevronRight :size="16" />
                </span>
                <span class="kbw-library-description">
                  {{ kb.description || '尚未填写知识库描述。' }}
                </span>
                <span class="kbw-library-meta">
                  <span><Layers3 :size="12" /> {{ shortCollectionName(kb.collection_name) }}</span>
                  <span><CalendarDays :size="12" /> {{ formatDate(kb.created_at) }}</span>
                </span>
              </span>
            </button>
          </div>
        </section>
      </div>
    </template>

    <template v-else>
      <header class="kbw-detail-header">
        <button class="kbw-back-button" title="返回知识库列表" @click="leaveKb">
          <ArrowLeft :size="18" />
        </button>
        <span class="kbw-detail-icon"><Database :size="22" /></span>
        <div class="kbw-detail-copy">
          <div class="kbw-detail-title-row">
            <h1>{{ activeKb.name }}</h1>
            <span class="kbw-live-badge"><i></i> 可用</span>
          </div>
          <p>{{ activeKb.description || '这个知识库暂时没有描述。' }}</p>
        </div>
        <div class="kbw-detail-actions">
          <button class="kbw-secondary-button" @click="copyKbId">
            <Copy :size="14" /> 复制 ID
          </button>
          <button class="kbw-primary-button" @click="openEditKb">
            <Pencil :size="14" /> 编辑
          </button>
        </div>
      </header>

      <nav class="kbw-tabs" aria-label="知识库功能">
        <button
          v-for="tab in tabItems"
          :key="tab.id"
          :class="{ active: activeTab === tab.id }"
          @click="selectTab(tab.id)"
        >
          <component :is="tab.icon" :size="16" />
          <span>{{ tab.label }}</span>
          <em v-if="tab.id !== 'files'">前端</em>
        </button>
      </nav>

      <main class="kbw-detail-scroll">
        <section v-if="activeTab === 'files'" class="kbw-workspace-section">
          <div class="kbw-workspace-heading">
            <div>
              <span class="kbw-eyebrow">DOCUMENTS</span>
              <h2>文件管理</h2>
              <p>上传、查看和管理 Agent 可检索的源文件。</p>
            </div>
            <div class="kbw-heading-actions">
              <button class="kbw-secondary-button" :disabled="filesLoading" @click="loadFiles()">
                <RefreshCw :size="14" :class="{ spin: filesLoading }" /> 刷新
              </button>
              <button class="kbw-primary-button" @click="showUpload = true">
                <Upload :size="14" /> 上传文件
              </button>
            </div>
          </div>

          <div class="kbw-metric-grid">
            <article>
              <span class="kbw-metric-icon"><FileStack :size="17" /></span>
              <div><small>文件总数</small><strong>{{ fileList.length }}</strong></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><CheckCircle2 :size="17" /></span>
              <div><small>索引完成</small><strong>{{ completedFiles.length }}</strong></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><Blocks :size="17" /></span>
              <div><small>内容分块</small><strong>{{ formatNumber(totalChunks) }}</strong></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><Binary :size="17" /></span>
              <div><small>字符总数</small><strong>{{ formatCompactNumber(totalCharacters) }}</strong></div>
            </article>
          </div>

          <div v-if="deleteSuccess" class="kbw-inline-notice is-success">
            <CheckCircle2 :size="15" /> {{ deleteSuccess }}
          </div>

          <div class="kbw-table-card">
            <div class="kbw-table-toolbar">
              <div>
                <strong>源文件</strong>
                <span>{{ fileList.length }} 项</span>
              </div>
              <span class="kbw-id-label">ID {{ shortId(activeKb.id) }}</span>
            </div>
            <div v-if="filesLoading" class="kbw-table-state">
              <LoaderCircle :size="20" class="spin" /> 正在读取文件列表
            </div>
            <div v-else-if="fileList.length === 0" class="kbw-table-state is-empty">
              <FileUp :size="25" />
              <strong>知识库中还没有文件</strong>
              <span>支持 TXT、Markdown、PDF、DOCX 和常见图片格式。</span>
              <button class="kbw-primary-button" @click="showUpload = true">
                <Upload :size="14" /> 上传第一个文件
              </button>
            </div>
            <div v-else class="kbw-table-wrap">
              <table class="kbw-file-table">
                <thead>
                  <tr>
                    <th>文件名</th>
                    <th>类型</th>
                    <th>解析器</th>
                    <th>分块</th>
                    <th>字符数</th>
                    <th>状态</th>
                    <th>上传时间</th>
                    <th><span class="sr-only">操作</span></th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="file in fileList" :key="file.id">
                    <td>
                      <button
                        class="kbw-file-link"
                        :disabled="file.status !== 'completed'"
                        :title="file.status === 'completed' ? '预览文件' : '索引完成后可预览'"
                        @click="openPreview(file)"
                      >
                        <span><FileText :size="16" /></span>
                        <span><strong>{{ file.filename }}</strong><small>{{ shortId(file.id) }}</small></span>
                      </button>
                    </td>
                    <td><span class="kbw-type-badge">{{ file.file_type || 'FILE' }}</span></td>
                    <td>
                      <span class="kbw-parser-cell" :title="parserDetails(file)">
                        <span :class="['kbw-parser-badge', { 'is-mineru': file.parser_name === 'mineru' }]">
                          {{ parserLabel(file) }}
                        </span>
                        <small v-if="file.parser_version">v{{ file.parser_version }}</small>
                        <small v-if="file.parser_task_id">任务 {{ shortId(file.parser_task_id) }}</small>
                      </span>
                    </td>
                    <td>{{ formatNumber(file.chunk_count) }}</td>
                    <td>{{ formatNumber(file.char_count) }}</td>
                    <td>
                      <span class="kbw-file-status-cell">
                        <span :class="['kbw-status-badge', file.status]">
                          <i></i>{{ statusLabel(file.status) }}
                        </span>
                        <small v-if="file.status === 'processing'">
                          {{ file.progress }}% · {{ file.progress_message || stageLabel(file.processing_stage) }}
                        </small>
                      </span>
                    </td>
                    <td>{{ formatDate(file.created_at) }}</td>
                    <td>
                      <button
                        class="kbw-icon-button"
                        :class="{ 'is-busy': reindexingIds.includes(file.id) }"
                        :title="reindexingIds.includes(file.id) ? '重新索引中' : '重新索引'"
                        :disabled="reindexingIds.includes(file.id)"
                        @click="reindexFile(file)"
                      >
                        <RefreshCw :size="15" :class="{ spin: reindexingIds.includes(file.id) }" />
                      </button>
                      <button class="kbw-icon-danger" title="删除文件" @click="confirmDelete(file)">
                        <Trash2 :size="15" />
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section v-else-if="activeTab === 'retrieval'" class="kbw-workspace-section">
          <div class="kbw-workspace-heading">
            <div>
              <span class="kbw-eyebrow">RETRIEVAL LAB</span>
              <h2>检索测试</h2>
              <p>对当前知识库执行真实向量检索，检查召回内容与排序。</p>
            </div>
            <span class="kbw-module-status"><CheckCircle2 :size="14" /> 已接入后端</span>
          </div>

          <div class="kbw-split-layout">
            <article class="kbw-panel-card kbw-query-card">
              <div class="kbw-panel-title">
                <div><ScanSearch :size="17" /><strong>测试查询</strong></div>
                <span>配置</span>
              </div>
              <label class="kbw-field">
                <span>问题</span>
                <textarea v-model="retrievalQuery" rows="6" placeholder="输入一个真实问题，例如：这份文档的核心结论是什么？"></textarea>
              </label>
              <div class="kbw-field">
                <span>检索模式</span>
                <div class="kbw-mode-switch">
                  <button
                    type="button"
                    class="kbw-mode-btn"
                    :class="{ 'is-active': retrievalMode === 'basic' }"
                    @click="retrievalMode = 'basic'"
                  >基础向量检索</button>
                  <button
                    type="button"
                    class="kbw-mode-btn"
                    :class="{ 'is-active': retrievalMode === 'enhanced' }"
                    @click="retrievalMode = 'enhanced'"
                  >增强检索（图谱+融合）</button>
                </div>
                <small v-if="retrievalMode === 'enhanced'">走查询分解 + 四路并行检索 + 图谱融合重排，相似度阈值对融合后的知识块评分生效。</small>
              </div>
              <div class="kbw-field-grid">
                <label class="kbw-field">
                  <span>Top K</span>
                  <select v-model.number="retrievalTopK">
                    <option :value="3">3</option>
                    <option :value="5">5</option>
                    <option :value="10">10</option>
                    <option :value="20">20</option>
                  </select>
                </label>
                <label class="kbw-field">
                  <span>最低相似度</span>
                  <select v-model.number="retrievalThreshold">
                    <option :value="0">不过滤</option>
                    <option :value="0.3">0.30</option>
                    <option :value="0.5">0.50</option>
                    <option :value="0.7">0.70</option>
                  </select>
                </label>
              </div>
              <button class="kbw-primary-button is-wide" :disabled="retrievalLoading" @click="runRetrievalPreview">
                <LoaderCircle v-if="retrievalLoading" :size="14" class="spin" />
                <Play v-else :size="14" /> {{ retrievalLoading ? '检索中' : '开始测试' }}
              </button>
            </article>

            <article class="kbw-panel-card kbw-contract-card">
              <div class="kbw-panel-title">
                <div><Braces :size="17" /><strong>请求预览</strong></div>
                <span>供后端接入</span>
              </div>
              <pre>{{ retrievalContract }}</pre>
              <div class="kbw-contract-foot">
                <Info :size="14" />
                <span>请求只会检索当前知识库，不会读取其他知识库的内容。</span>
              </div>
            </article>
          </div>

          <article class="kbw-panel-card kbw-result-panel">
            <div class="kbw-panel-title">
              <div><ListFilter :size="17" /><strong>{{ retrievalRun?.mode === 'enhanced' ? '增强检索结果' : '召回结果' }}</strong></div>
              <span v-if="retrievalRun">{{ retrievalRun.total }} 条 · {{ retrievalRun.elapsed_ms }} ms</span>
              <span v-else>0 条</span>
            </div>
            <div v-if="retrievalLoading" class="kbw-result-empty">
              <LoaderCircle :size="28" class="spin" />
              <strong>正在执行检索</strong>
              <span>{{ retrievalMode === 'enhanced' ? '正在查询分解、四路并行检索并融合重排。' : '正在生成查询向量并从当前知识库召回内容。' }}</span>
            </div>
            <div v-else-if="retrievalError" class="kbw-result-empty">
              <CircleAlert :size="28" />
              <strong>检索失败</strong>
              <span>{{ retrievalError }}</span>
              <button class="kbw-secondary-button" @click="runRetrievalPreview">重新测试</button>
            </div>
            <div v-else-if="!hasRetrievalResult" class="kbw-result-empty">
              <Waypoints :size="28" />
              <strong>{{ retrievalAttempted ? '没有符合条件的召回结果' : '等待一次测试查询' }}</strong>
              <span>{{ retrievalAttempted ? '可以降低最低相似度或换一个更贴近文档内容的问题。' : '结果区将展示命中文件、分块正文、相似度、排名和耗时。' }}</span>
            </div>
            <div v-else-if="retrievalRun?.mode === 'enhanced'" class="kbw-enhanced-results">
              <div v-if="retrievalRun.query_decomposition" class="kbw-decomp-card">
                <div class="kbw-decomp-head">
                  <strong>查询分解</strong>
                  <span class="kbw-decomp-tags">
                    <em>{{ retrievalRun.query_decomposition.query_type || 'unknown' }}</em>
                    <em>{{ retrievalRun.query_decomposition.complexity || 'unknown' }}</em>
                    <em v-if="retrievalRun.gap_rounds">缺口补充 {{ retrievalRun.gap_rounds }} 轮</em>
                  </span>
                </div>
                <div v-if="retrievalRun.query_decomposition.sub_questions?.length" class="kbw-decomp-row">
                  <span class="kbw-decomp-label">子问题</span>
                  <ul><li v-for="(sq, si) in retrievalRun.query_decomposition.sub_questions" :key="si">{{ sq }}</li></ul>
                </div>
                <div v-if="retrievalRun.query_decomposition.explicit_entities?.length" class="kbw-decomp-row">
                  <span class="kbw-decomp-label">实体</span>
                  <span>{{ retrievalRun.query_decomposition.explicit_entities.map(e => e.name).join('、') }}</span>
                </div>
                <div v-if="retrievalRun.query_decomposition.relation_patterns?.length" class="kbw-decomp-row">
                  <span class="kbw-decomp-label">关系模式</span>
                  <span>{{ retrievalRun.query_decomposition.relation_patterns.map(r => `${r.subject} → ${r.predicate} → ${r.object}`).join('；') }}</span>
                </div>
              </div>

              <div class="kbw-subq-tabs">
                <button
                  type="button"
                  class="kbw-subq-tab"
                  :class="{ 'is-active': selectedSubQuestion < 0 }"
                  @click="selectedSubQuestion = -1"
                >全部</button>
                <button
                  v-for="(sq, si) in (retrievalRun.query_decomposition?.sub_questions || [])"
                  :key="si"
                  type="button"
                  class="kbw-subq-tab"
                  :class="{ 'is-active': selectedSubQuestion === si }"
                  :title="sq"
                  @click="selectedSubQuestion = si"
                >子问题 {{ si + 1 }}</button>
              </div>

              <div v-if="!filteredBlocks.length" class="kbw-result-empty">
                <Waypoints :size="24" />
                <span>该子问题下没有可展示的知识块。</span>
              </div>

              <div v-for="(block, bi) in filteredBlocks" :key="block.block_id || bi" class="kbw-block-card">
                <header class="kbw-block-head">
                  <strong>知识块 #{{ bi + 1 }}</strong>
                  <span v-if="block.block_score != null">评分 {{ block.block_score }}</span>
                </header>
                <p v-if="block.summary" class="kbw-block-summary">{{ block.summary }}</p>
                <div v-if="block.sub_questions?.length" class="kbw-block-relations">
                  <span class="kbw-decomp-label">回答子问题</span>
                  <ul>
                    <li v-for="(sq, si) in block.sub_questions" :key="si">{{ sq }}</li>
                  </ul>
                </div>
                <div v-if="block.entities?.length" class="kbw-block-entities">
                  <span class="kbw-decomp-label">实体</span>
                  <span>{{ block.entities.join('、') }}</span>
                </div>
                <div v-if="block.relations?.length" class="kbw-block-relations">
                  <span class="kbw-decomp-label">关系</span>
                  <ul>
                    <li v-for="(r, ri) in block.relations" :key="ri">
                      {{ r.source || r.subject }} --{{ r.relation || r.predicate }}--> {{ r.target || r.object }}
                    </li>
                  </ul>
                </div>
                <div v-for="(doc, di) in block.docs" :key="di" class="kbw-block-doc">
                  <div class="kbw-block-doc-meta">
                    <span v-if="doc.retrieval_path" class="kbw-path-tag">{{ doc.retrieval_path }}</span>
                    <span v-if="doc.cross_path_hits > 1" class="kbw-path-tag">多路命中 ×{{ doc.cross_path_hits }}</span>
                    <span v-if="doc.graph_entities?.length" class="kbw-path-tag">图谱实体：{{ doc.graph_entities.join('、') }}</span>
                    <span v-if="doc.score != null" class="kbw-path-tag">相似度 {{ formatSimilarity(doc.score) }}</span>
                  </div>
                  <p>{{ doc.content }}</p>
                </div>
              </div>
            </div>
            <div v-else class="kbw-retrieval-results">
              <article v-for="hit in retrievalRun.results" :key="`${hit.rank}-${hit.file_id || hit.source || 'hit'}`" class="kbw-retrieval-hit">
                <header>
                  <div class="kbw-hit-source">
                    <span class="kbw-hit-rank">#{{ hit.rank }}</span>
                    <div>
                      <strong :title="hit.source || '未知来源'">{{ hit.source || '未知来源' }}</strong>
                      <small>
                        <template v-if="hit.chunk_index !== null && hit.chunk_index !== undefined">分块 {{ hit.chunk_index }}</template>
                        <template v-if="pageRange(hit)"> · {{ pageRange(hit) }}</template>
                        <template v-if="hit.parser_name"> · {{ parserDisplayName(hit.parser_name) }}</template>
                      </small>
                    </div>
                  </div>
                  <div class="kbw-hit-score">
                    <small>相似度</small>
                    <strong>{{ formatSimilarity(hit.score) }}</strong>
                  </div>
                </header>
                <p>{{ hit.content || '该分块没有可展示的正文。' }}</p>
                <footer v-if="hit.section_path || hit.retrieval_path">
                  <span v-if="hit.section_path">章节：{{ hit.section_path }}</span>
                  <span v-if="hit.retrieval_path">路径：{{ hit.retrieval_path }}</span>
                </footer>
              </article>
            </div>
          </article>
        </section>

        <section v-else-if="activeTab === 'graph'" class="kbw-workspace-section">
          <div class="kbw-workspace-heading">
            <div>
              <span class="kbw-eyebrow">KNOWLEDGE GRAPH</span>
              <h2>知识图谱</h2>
              <p>实体与关系网络：颜色区分实体类型，节点大小代表连接度，点击节点查看详情。</p>
            </div>
            <div class="kbw-heading-actions">
              <span v-if="graphData" class="kbw-module-status"><CheckCircle2 :size="14" /> {{ graphData.entities.length }} 实体 · {{ graphData.relations.length }} 关系</span>
              <button type="button" class="kbw-secondary-button" @click="loadGraph"><RefreshCw :size="14" /> 刷新</button>
            </div>
          </div>

          <div class="kbw-graph-layout">
            <article class="kbw-panel-card kbw-graph-canvas">
              <div class="kbw-panel-title">
                <div><Network :size="17" /><strong>实体关系图</strong></div>
              </div>
              <div v-if="graphLoading" class="kbw-result-empty">
                <LoaderCircle :size="28" class="kbw-spin" />
                <span>正在加载图谱…</span>
              </div>
              <div v-else-if="graphError" class="kbw-result-empty">
                <CircleAlert :size="28" />
                <strong>加载失败</strong>
                <span>{{ graphError }}</span>
              </div>
              <div v-else-if="!graphData || !graphData.entities?.length" class="kbw-result-empty">
                <Network :size="28" />
                <strong>尚未抽取实体</strong>
                <span>上传并处理文件后，系统会自动抽取实体与关系。点击右上角「刷新」重新加载。</span>
              </div>
              <div
                ref="graphContainer"
                v-show="graphData && graphData.entities?.length && !graphLoading && !graphError"
                class="kbw-graph-stage"
              ></div>
            </article>
            <aside class="kbw-panel-card kbw-graph-sidebar">
              <div class="kbw-panel-title">
                <div><PanelRight :size="17" /><strong>图例与详情</strong></div>
              </div>
              <div class="kbw-graph-legend">
                <span v-for="(color, type) in graphTypeColors" :key="type" class="kbw-legend-item">
                  <i :style="{ background: color }"></i>{{ type }}
                </span>
              </div>
              <div v-if="graphNodeDetail" class="kbw-node-detail">
                <strong>{{ graphNodeDetail.label }}</strong>
                <span class="kbw-node-type" :style="{ color: entityColor(graphNodeDetail.type) }">{{ graphNodeDetail.type }}</span>
                <p v-if="graphNodeDetail.description" class="kbw-node-desc">{{ graphNodeDetail.description }}</p>
                <div class="kbw-node-neighbors">
                  <div class="kbw-neighbors-head">
                    <span>关联关系</span>
                    <em v-if="graphNodeDetail.neighborTotal != null">{{ graphNodeDetail.neighborTotal }}</em>
                  </div>
                  <div v-if="graphNodeDetail.neighborsLoading" class="kbw-neighbors-state">加载中…</div>
                  <div v-else-if="graphNodeDetail.neighborsError" class="kbw-neighbors-state is-error">{{ graphNodeDetail.neighborsError }}</div>
                  <div v-else-if="!graphNodeDetail.neighbors?.length" class="kbw-neighbors-state">暂无关联关系</div>
                  <ul v-else class="kbw-neighbors-list">
                    <li
                      v-for="(n, ni) in graphNodeDetail.neighbors"
                      :key="ni"
                      class="kbw-neighbor-item"
                      :title="n.relation_description || ''"
                      @click="focusNeighbor(n)"
                    >
                      <span class="kbw-neighbor-arrow" :class="n.direction">{{ n.direction === 'out' ? '→' : '←' }}</span>
                      <span class="kbw-neighbor-rel">{{ n.relation_type }}</span>
                      <span class="kbw-neighbor-name">{{ n.name }}</span>
                      <span class="kbw-neighbor-type" :style="{ color: entityColor(n.entity_type) }">{{ n.entity_type }}</span>
                    </li>
                  </ul>
                </div>
              </div>
              <div v-else class="kbw-sidebar-note">
                <Info :size="15" />
                <p><strong>提示</strong><span>点击实体聚焦并高亮其关联关系；拖拽旋转视角，滚轮缩放，右键平移。点击空白处复位。</span></p>
              </div>
            </aside>
          </div>
        </section>

        <section v-else-if="activeTab === 'map'" class="kbw-workspace-section">
          <div class="kbw-workspace-heading">
            <div>
              <span class="kbw-eyebrow">KNOWLEDGE MAP</span>
              <h2>知识导图</h2>
              <p>以知识库为根节点整理文件目录，后续可扩展到章节和知识点。</p>
            </div>
            <span class="kbw-module-status"><CircleDashed :size="14" /> 内容解析待接入</span>
          </div>
          <article class="kbw-panel-card kbw-map-card">
            <div class="kbw-map-root">
              <span><Database :size="19" /></span>
              <div><small>知识库</small><strong>{{ activeKb.name }}</strong></div>
            </div>
            <div v-if="fileList.length" class="kbw-map-branches">
              <div v-for="file in fileList" :key="file.id" class="kbw-map-branch">
                <i></i>
                <div>
                  <span><FileText :size="15" /></span>
                  <p><strong>{{ file.filename }}</strong><small>{{ statusLabel(file.status) }} · {{ formatNumber(file.chunk_count) }} 个分块</small></p>
                </div>
              </div>
            </div>
            <div v-else class="kbw-result-empty">
              <GitBranch :size="28" />
              <strong>暂无可整理的文件</strong>
              <span>上传文件后，这里会先显示文件级目录结构。</span>
            </div>
          </article>
        </section>

        <section v-else-if="activeTab === 'evaluation'" class="kbw-workspace-section">
          <div class="kbw-workspace-heading">
            <div>
              <span class="kbw-eyebrow">RAG EVALUATION</span>
              <h2>RAG 评估</h2>
              <p>评估检索召回和回答质量；当前先完成页面、状态与配置流程。</p>
            </div>
            <button class="kbw-primary-button" @click="openEvaluationSetup">
              <Play :size="14" /> 开始评估
            </button>
          </div>

          <article class="kbw-evaluation-card">
            <div class="kbw-evaluation-main">
              <div class="kbw-score-ring"><strong>—</strong><span>暂无评分</span></div>
              <div>
                <span class="kbw-module-status"><CircleDashed :size="14" /> 等待配置</span>
                <h3>还没有评估运行记录</h3>
                <p>选择评估基准和指标后即可创建首次运行；提交动作将在后端接口接入后启用。</p>
                <div class="kbw-readiness-row">
                  <span><CheckCircle2 :size="13" /> {{ completedFiles.length }} 个可评估文件</span>
                  <span><ListChecks :size="13" /> 0 个评估基准</span>
                </div>
              </div>
            </div>
            <div class="kbw-evaluation-metrics">
              <div><span>Recall@10</span><strong>—</strong><small>召回率</small></div>
              <div><span>耗时</span><strong>—</strong><small>运行耗时</small></div>
              <div><span>数据量</span><strong>0</strong><small>评估问题</small></div>
              <div><span>完成率</span><strong>—</strong><small>执行进度</small></div>
            </div>
          </article>

          <article class="kbw-panel-card kbw-history-card">
            <div class="kbw-panel-title">
              <div><History :size="17" /><strong>历史评估记录</strong></div>
              <button class="kbw-text-button" @click="frontendOnly('评估记录刷新')"><RefreshCw :size="13" /> 刷新</button>
            </div>
            <div class="kbw-history-table-head">
              <span>评估名称</span><span>评估基准</span><span>数据量</span><span>耗时</span><span>Recall@10</span><span>综合评分</span><span>状态</span>
            </div>
            <div class="kbw-result-empty is-compact">
              <BarChart3 :size="25" />
              <strong>暂无历史评估</strong>
              <span>运行结果会按时间保留在这里。</span>
            </div>
          </article>
        </section>

        <section v-else class="kbw-workspace-section">
          <div class="kbw-workspace-heading">
            <div>
              <span class="kbw-eyebrow">EVALUATION STANDARD</span>
              <h2>评估基准</h2>
              <p>定义评估数据集和指标组合，为 RAG 评估提供可复用标准。</p>
            </div>
            <button class="kbw-primary-button" @click="frontendOnly('新建评估基准')">
              <Plus :size="14" /> 新建基准
            </button>
          </div>

          <div class="kbw-criteria-grid">
            <article v-for="criterion in criteria" :key="criterion.id" class="kbw-criterion-card">
              <div>
                <span><component :is="criterion.icon" :size="17" /></span>
                <button
                  class="kbw-toggle"
                  :class="{ active: criterion.enabled }"
                  :aria-pressed="criterion.enabled"
                  @click="toggleCriterion(criterion)"
                ><i></i></button>
              </div>
              <strong>{{ criterion.name }}</strong>
              <p>{{ criterion.description }}</p>
              <small>{{ criterion.group }}</small>
            </article>
          </div>

          <article class="kbw-panel-card kbw-baseline-card">
            <div class="kbw-panel-title">
              <div><ClipboardCheck :size="17" /><strong>评估数据集</strong></div>
              <span>0 个</span>
            </div>
            <div class="kbw-result-empty">
              <FileQuestion :size="28" />
              <strong>还没有评估基准</strong>
              <span>后续可上传“问题、期望答案、相关文档”组成的数据集。</span>
              <button class="kbw-secondary-button" @click="frontendOnly('评估基准上传')">了解待接入字段</button>
            </div>
          </article>
        </section>
      </main>
    </template>

    <div v-if="moduleNotice" class="kbw-toast" role="status">
      <Info :size="15" /> {{ moduleNotice }}
    </div>

    <div v-if="showCreate" class="modal-overlay" @click.self="showCreate = false">
      <div class="modal kbw-modal">
        <div class="kbw-modal-heading">
          <div><span><Database :size="18" /></span><div><h3>新建知识库</h3><p>创建后即可进入详情页上传文件。</p></div></div>
          <button @click="showCreate = false"><X :size="17" /></button>
        </div>
        <label class="kbw-field">
          <span>名称</span>
          <input v-model="newKb.name" type="text" maxlength="80" placeholder="例如：产品技术文档" @keyup.enter="createKb" />
        </label>
        <label class="kbw-field">
          <span>描述（可选）</span>
          <textarea v-model="newKb.description" rows="3" maxlength="300" placeholder="说明这个知识库收录什么内容"></textarea>
        </label>
        <p v-if="createError" class="kbw-form-error">{{ createError }}</p>
        <div class="modal-actions">
          <button class="kbw-secondary-button" @click="showCreate = false">取消</button>
          <button class="kbw-primary-button" :disabled="creating || !newKb.name.trim()" @click="createKb">
            <LoaderCircle v-if="creating" :size="14" class="spin" />
            <Plus v-else :size="14" /> {{ creating ? '创建中' : '创建知识库' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showEdit" class="modal-overlay" @click.self="showEdit = false">
      <div class="modal kbw-modal">
        <div class="kbw-modal-heading">
          <div><span><Pencil :size="18" /></span><div><h3>编辑知识库</h3><p>修改「{{ activeKb?.name }}」的名称与描述。</p></div></div>
          <button @click="showEdit = false"><X :size="17" /></button>
        </div>
        <label class="kbw-field">
          <span>名称</span>
          <input v-model="editForm.name" type="text" maxlength="80" placeholder="例如：产品技术文档" @keyup.enter="saveKb" />
        </label>
        <label class="kbw-field">
          <span>描述（可选）</span>
          <textarea v-model="editForm.description" rows="3" maxlength="300" placeholder="说明这个知识库收录什么内容"></textarea>
        </label>
        <p v-if="editError" class="kbw-form-error">{{ editError }}</p>
        <div class="modal-actions">
          <button class="kbw-secondary-button" @click="showEdit = false">取消</button>
          <button class="kbw-primary-button" :disabled="saving || !editForm.name.trim()" @click="saveKb">
            <LoaderCircle v-if="saving" :size="14" class="spin" />
            <CheckCircle2 v-else :size="14" /> {{ saving ? '保存中' : '保存修改' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showUpload" class="modal-overlay" @click.self="closeUpload">
      <div class="modal kbw-modal">
        <div class="kbw-modal-heading">
          <div><span><FileUp :size="18" /></span><div><h3>上传文件</h3><p>上传到「{{ activeKb?.name }}」</p></div></div>
          <button :disabled="uploading && uploadPhase === 'transferring'" @click="closeUpload"><X :size="17" /></button>
        </div>
        <label
          class="file-label kbw-dropzone"
          :class="{ 'file-label--dragover': dragOver, 'file-label--has-file': uploadFile }"
          @dragover.prevent="onDragOver"
          @dragleave.prevent="onDragLeave"
          @drop.prevent="onDrop"
        >
          <input type="file" :disabled="uploading" accept=".txt,.md,.pdf,.docx,.pptx,.xlsx,.png,.jpg,.jpeg,.bmp,.webp,.gif,.tif,.tiff,.jp2" @change="onFileSelect" />
          <span v-if="!uploadFile" class="file-label-hint">
            <UploadCloud :size="23" class="file-label-icon" />
            <strong>{{ dragOver ? '松开以上传' : '点击选择或拖拽文件到这里' }}</strong>
            <em>TXT、MD、PDF、DOCX 与常见图片</em>
          </span>
          <span v-else class="kbw-selected-file">
            <FileText :size="20" /><span><strong>{{ uploadFile.name }}</strong><small>{{ formatSize(uploadFile.size) }}</small></span>
          </span>
        </label>
        <div class="kbw-parser-picker">
          <div class="kbw-parser-picker-heading">
            <strong>文档解析器</strong>
            <span>PDF 默认使用 MinerU</span>
          </div>
          <div class="kbw-parser-options" role="radiogroup" aria-label="选择文档解析器">
            <label
              v-for="option in parserOptions"
              :key="option.value"
              :class="[
                'kbw-parser-option',
                { 'is-selected': uploadParserChoice === option.value, 'is-disabled': !parserOptionAvailable(option.value) },
              ]"
            >
              <input
                v-model="uploadParserChoice"
                type="radio"
                name="document-parser"
                :value="option.value"
                :disabled="uploading || !parserOptionAvailable(option.value)"
              />
              <span><strong>{{ option.label }}</strong><small>{{ option.description }}</small></span>
            </label>
          </div>
        </div>
        <div v-if="uploading || uploadPhase !== 'idle'" class="upload-progress">
          <div class="upload-progress-label"><span>{{ progressLabel }}</span><span>{{ displayProgress }}%</span></div>
          <div class="progress-track">
            <div
              class="progress-fill"
              :class="{ indeterminate: uploadPhase === 'indexing' && indexProgress === 0, failed: uploadPhase === 'failed' }"
              :style="{ width: `${displayProgress}%` }"
            ></div>
          </div>
          <div v-if="uploadParser?.parser_name" class="kbw-upload-parser" :title="parserDetails(uploadParser)">
            <span>当前解析器</span>
            <strong>{{ parserLabel(uploadParser) }}</strong>
            <small v-if="uploadParser.parser_version">v{{ uploadParser.parser_version }}</small>
            <small v-if="uploadParser.parser_backend">{{ uploadParser.parser_backend }}</small>
            <small v-if="uploadParser.parser_task_id">任务 {{ shortId(uploadParser.parser_task_id) }}</small>
          </div>
          <div v-if="uploadPhase === 'indexing'" class="kbw-live-progress-meta">
            <span><i></i>{{ stageLabel(uploadParser?.processing_stage) }}</span>
            <span v-if="uploadParser?.progress_total">
              {{ uploadParser.progress_current }}/{{ uploadParser.progress_total }} 项
            </span>
            <span>已用时 {{ formatDuration(uploadElapsedSeconds) }}</span>
          </div>
          <p v-if="uploadConnectionIssue" class="kbw-progress-warning">
            暂时无法获取最新进度，正在自动重试；后台任务不会因此中断。
          </p>
          <p v-else-if="uploadProgressIdleSeconds >= 30 && uploadPhase === 'indexing'" class="kbw-progress-hint">
            当前步骤已运行 {{ formatDuration(uploadProgressIdleSeconds) }}，复杂文档或模型调用可能需要更长时间。
          </p>
        </div>
        <p v-if="uploadMsg" :class="['kbw-inline-notice', uploadOk ? 'is-success' : 'is-error']">{{ uploadMsg }}</p>
        <div class="modal-actions">
          <button class="kbw-secondary-button" :disabled="uploading && uploadPhase === 'transferring'" @click="closeUpload">
            {{ uploading ? '后台继续' : '关闭' }}
          </button>
          <button class="kbw-primary-button" :disabled="!uploadFile || uploading" @click="doUpload">
            <LoaderCircle v-if="uploading" :size="14" class="spin" />
            <Upload v-else :size="14" /> {{ uploading ? '处理中' : '上传并索引' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showPreview" class="modal-overlay" @click.self="closePreview">
      <div class="modal preview-modal kbw-preview-modal">
        <div class="preview-header kbw-preview-header">
          <div><span><FileText :size="18" /></span><div><h3>{{ previewFile?.filename }}</h3><p>{{ previewFile?.file_type?.toUpperCase() }} · {{ formatNumber(previewFile?.char_count) }} 字符</p></div></div>
          <button class="kbw-icon-button" @click="closePreview"><X :size="17" /></button>
        </div>
        <div v-if="previewLoading" class="preview-loading"><LoaderCircle :size="20" class="spin" /> 正在加载预览</div>
        <div v-else-if="previewError" class="preview-error">{{ previewError }}</div>
        <div v-else-if="previewContentType === 'binary'" class="preview-binary-wrap">
          <iframe v-if="previewFile?.file_type === 'pdf'" :src="rawUrl" class="preview-frame" />
          <img v-else :src="rawUrl" :alt="previewFile?.filename" class="preview-image" />
        </div>
        <div v-else class="preview-text-wrap"><pre class="preview-text">{{ previewText }}</pre></div>
      </div>
    </div>

    <div v-if="showDeleteConfirm" class="modal-overlay" @click.self="showDeleteConfirm = false">
      <div class="modal kbw-modal kbw-danger-modal">
        <div class="kbw-modal-heading">
          <div><span><Trash2 :size="18" /></span><div><h3>删除文件</h3><p>此操作不可恢复</p></div></div>
          <button @click="showDeleteConfirm = false"><X :size="17" /></button>
        </div>
        <p class="delete-warning">确定删除「<strong>{{ deleteTarget?.filename }}</strong>」吗？源文件和对应向量索引都会被删除。</p>
        <div class="modal-actions">
          <button class="kbw-secondary-button" @click="showDeleteConfirm = false">取消</button>
          <button class="btn-danger-sm" :disabled="deleting" @click="doDelete">
            <LoaderCircle v-if="deleting" :size="14" class="spin" />
            <Trash2 v-else :size="14" /> {{ deleting ? '删除中' : '确认删除' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showEvaluationSetup" class="modal-overlay" @click.self="showEvaluationSetup = false">
      <div class="modal kbw-modal kbw-eval-modal">
        <div class="kbw-modal-heading">
          <div><span><BarChart3 :size="18" /></span><div><h3>配置 RAG 评估</h3><p>前端配置预览，暂不提交运行。</p></div></div>
          <button @click="showEvaluationSetup = false"><X :size="17" /></button>
        </div>
        <label class="kbw-field">
          <span>评估基准</span>
          <select disabled><option>暂无基准，请先在“评估基准”页创建</option></select>
        </label>
        <div class="kbw-eval-checks">
          <span>评估指标</span>
          <label v-for="criterion in enabledCriteria" :key="criterion.id">
            <input type="checkbox" checked />
            <span><strong>{{ criterion.name }}</strong><small>{{ criterion.group }}</small></span>
          </label>
        </div>
        <div class="kbw-inline-notice"><Info :size="14" /> 后端接入后，此处将创建评估运行并实时更新进度。</div>
        <div class="modal-actions">
          <button class="kbw-secondary-button" @click="showEvaluationSetup = false">取消</button>
          <button class="kbw-primary-button" @click="frontendOnly('RAG 评估运行'); showEvaluationSetup = false">
            <Play :size="14" /> 保存前端配置
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  ArrowLeft, BarChart3, Binary, Blocks, Braces, CalendarDays, CheckCircle2,
  ChevronRight, CircleAlert, CircleDashed, ClipboardCheck, Copy, Database,
  FileQuestion, FileStack, FileText, FileUp, GitBranch, History, Info,
  Layers3, ListChecks, ListFilter, LoaderCircle, Network, PanelRight, Pencil,
  Play, Plus, RefreshCw, ScanSearch, Search, SearchX, Sparkles, Trash2, Upload,
  UploadCloud, Waypoints, X,
} from 'lucide-vue-next'
import api from '../api'
import * as echarts from 'echarts'

const route = useRoute()
const router = useRouter()

const tabItems = [
  { id: 'files', label: '文件管理', icon: FileText },
  { id: 'retrieval', label: '检索测试', icon: Search },
  { id: 'graph', label: '知识图谱', icon: Network },
  { id: 'map', label: '知识导图', icon: GitBranch },
  { id: 'evaluation', label: 'RAG 评估', icon: BarChart3 },
  { id: 'benchmarks', label: '评估基准', icon: ClipboardCheck },
]
const validTabs = new Set(tabItems.map((tab) => tab.id))

const kbList = ref([])
const activeKb = ref(null)
const activeTab = ref('files')

// ── 知识图谱 ──────────────────────────────────────────────────────────
const graphData = ref(null)
const graphLoading = ref(false)
const graphError = ref('')
const graphContainer = ref(null)
const graphNodeDetail = ref(null)
let graphChart = null

const PALETTE = ['#5b8def', '#34c77b', '#f5a623', '#e45b5b', '#9b6bf0', '#2ab8c2', '#e05ba0', '#8a9b6b', '#6b8f9b', '#c2a05b', '#8d6e63', '#5c6bc0', '#26a69a', '#ec407a', '#7cb342', '#ab47bc']

// 从实际图谱数据动态统计实体类型并分配颜色（通用：不预设任何领域类型）
const graphTypeColors = computed(() => {
  const types = []
  const seen = new Set()
  for (const e of (graphData.value?.entities || [])) {
    const t = e.entity_type || 'other'
    if (!seen.has(t)) { seen.add(t); types.push(t) }
  }
  const map = {}
  types.forEach((t, i) => { map[t] = PALETTE[i % PALETTE.length] })
  return map
})

function entityColor(type) {
  return graphTypeColors.value[type] || '#9aa0a6'
}

async function loadGraph() {
  const kbId = activeKb.value?.id
  if (!kbId) return
  graphLoading.value = true
  graphError.value = ''
  graphNodeDetail.value = null
  let fetched = null
  try {
    fetched = await api.get(`/knowledge/bases/${kbId}/graph`, { limit: 300 })
    graphData.value = fetched
  } catch (e) {
    graphError.value = e.response?.data?.detail || '图谱加载失败，请稍后重试。'
  } finally {
    graphLoading.value = false
  }
  // 等 graphLoading=false 后容器才通过 v-show 显示（否则尺寸为 0，echarts.init 失败）
  if (fetched) {
    await nextTick()
    try {
      renderGraph(fetched)
    } catch (e) {
      console.error('[graph] renderGraph failed:', e)
      graphError.value = '图谱渲染失败：' + (e && e.message ? e.message : String(e))
    }
  }
}

function renderGraph(data) {
  if (graphChart) {
    if (graphChart.__resizeHandler) window.removeEventListener('resize', graphChart.__resizeHandler)
    graphChart.dispose()
    graphChart = null
  }
  const el = graphContainer.value
  if (!el || !data) return
  // 容器必须已有实际尺寸（v-show 尚未生效时 clientWidth/Height 为 0，echarts 会渲染空白）
  if (el.clientWidth === 0 || el.clientHeight === 0) {
    requestAnimationFrame(() => renderGraph(data))
    return
  }
  const entities = data.entities || []
  const relations = data.relations || []

  // 计算连接度
  const degree = {}
  for (const r of relations) {
    degree[r.source_entity] = (degree[r.source_entity] || 0) + 1
    degree[r.target_entity] = (degree[r.target_entity] || 0) + 1
  }

  // 类型 → 索引（categories）
  const typeIndex = {}
  const typeList = []
  for (const e of entities) {
    const t = e.entity_type || 'other'
    if (!(t in typeIndex)) {
      typeIndex[t] = typeList.length
      typeList.push(t)
    }
  }

  const nodes = entities.map((e) => ({
    id: e.name,
    name: e.name,
    value: degree[e.name] || 0,
    category: typeIndex[e.entity_type || 'other'],
    description: e.description || '',
    symbolSize: 8 + Math.min(degree[e.name] || 0, 12) * 2,
  }))
  const links = relations.map((r) => ({
    source: r.source_entity,
    target: r.target_entity,
    value: r.relation_type,
  }))

  const colors = typeList.map((_, i) => PALETTE[i % PALETTE.length])

  graphChart = echarts.init(el)
  graphChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (p) => {
        if (p.dataType === 'node') {
          const t = p.data.category !== undefined ? typeList[p.data.category] : 'other'
          return `${p.data.name} · ${t}`
        }
        return p.data.value || ''
      },
    },
    series: [{
      type: 'graph',
      layout: 'force',
      force: { repulsion: 400, edgeLength: 120, gravity: 0.1, layoutAnimation: false },
      data: nodes,
      links: links,
      categories: typeList.map((t) => ({ name: t })),
      roam: true,
      draggable: true,
      focusNodeAdjacency: true,
      color: colors,
      label: { show: true, position: 'right', formatter: '{b}', fontSize: 10, color: '#5a5e64' },
      lineStyle: { color: '#c3c9d1', width: 1.2, curveness: 0.15, opacity: 0.6 },
      edgeLabel: {
        show: true,
        formatter: (p) => p.data.value || '',
        fontSize: 8,
        color: '#8b9096',
        position: 'middle',
      },
      itemStyle: {
        shadowBlur: 8,
        shadowColor: 'rgba(0,0,0,0.25)',
        shadowOffsetY: 3,
        borderColor: '#ffffff',
        borderWidth: 1.5,
      },
      emphasis: {
        focus: 'adjacency',
        lineStyle: { color: '#4348ff', width: 2.5 },
        itemStyle: { shadowBlur: 16, shadowColor: 'rgba(67,72,255,0.5)' },
        label: { fontSize: 12, color: '#2c2e31' },
      },
    }],
  })

  // 容器尺寸变化时自适应
  const onResize = () => { if (graphChart) graphChart.resize() }
  window.addEventListener('resize', onResize)
  graphChart.__resizeHandler = onResize

  graphChart.on('click', (params) => {
    if (params.dataType === 'node') {
      clearHighlight()
      graphNodeDetail.value = {
        label: params.data.name,
        type: params.data.category !== undefined ? typeList[params.data.category] : 'other',
        description: params.data.description,
        neighbors: null,
        neighborTotal: 0,
        neighborsLoading: true,
        neighborsError: '',
      }
      loadNeighbors(params.data.name)
    } else {
      clearHighlight()
      graphNodeDetail.value = null
    }
  })
}

// 高亮的节点索引（点击邻居跳转时切换高亮）
let lastHighlightedIndex = -1

function clearHighlight() {
  if (graphChart && lastHighlightedIndex >= 0) {
    graphChart.dispatchAction({ type: 'downplay', seriesIndex: 0, dataIndex: lastHighlightedIndex })
    lastHighlightedIndex = -1
  }
}

// 加载选中实体的全部邻居（不受图可视化的 top-N 截断影响）
async function loadNeighbors(name) {
  const kbId = activeKb.value?.id
  const detail = graphNodeDetail.value
  if (!kbId || !detail) return
  detail.neighborsLoading = true
  detail.neighborsError = ''
  try {
    const res = await api.get(`/knowledge/bases/${kbId}/graph/neighbors`, { entity: name, limit: 50 })
    detail.neighbors = res.neighbors || []
    detail.neighborTotal = res.total
  } catch (e) {
    detail.neighborsError = e.response?.data?.detail || '关联关系加载失败'
    detail.neighbors = []
    detail.neighborTotal = 0
  } finally {
    detail.neighborsLoading = false
  }
}

// 点击右侧栏邻居：加载该邻居的详情与它的邻居，并高亮图中的对应节点
async function focusNeighbor(n) {
  const kbId = activeKb.value?.id
  if (!kbId) return
  clearHighlight()
  graphNodeDetail.value = {
    label: n.name,
    type: n.entity_type || 'other',
    description: n.description || '',
    neighbors: null,
    neighborTotal: 0,
    neighborsLoading: true,
    neighborsError: '',
  }
  await loadNeighbors(n.name)
  if (graphChart) {
    const data = graphChart.getOption()?.series?.[0]?.data || []
    const idx = data.findIndex((d) => d && d.name === n.name)
    if (idx >= 0) {
      lastHighlightedIndex = idx
      graphChart.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex: idx })
    }
  }
}

onBeforeUnmount(() => {
  if (graphChart) {
    if (graphChart.__resizeHandler) window.removeEventListener('resize', graphChart.__resizeHandler)
    graphChart.dispose()
    graphChart = null
  }
})

// 切到图谱页时自动加载（点击 tab 或 URL 刷新初始化都会触发）。
// 同时监听 activeKb：URL 初始化时 activeTab 先变、activeKb 后设置，
// 只监听 activeTab 会在 activeKb 还是 null 时触发 loadGraph 而直接返回。
watch([activeTab, () => activeKb.value?.id], ([tab, kbId]) => {
  if (tab === 'graph' && kbId) loadGraph()
})
const fileList = ref([])
const loading = ref(true)
const filesLoading = ref(false)
const catalogError = ref('')
const catalogQuery = ref('')

const showCreate = ref(false)
const newKb = reactive({ name: '', description: '' })
const creating = ref(false)
const createError = ref('')

const showEdit = ref(false)
const editForm = reactive({ name: '', description: '' })
const saving = ref(false)
const editError = ref('')

const showUpload = ref(false)
const uploadFile = ref(null)
const uploading = ref(false)
const uploadMsg = ref('')
const uploadOk = ref(false)
const uploadPhase = ref('idle')
const transferProgress = ref(0)
const indexProgress = ref(0)
const uploadParser = ref(null)
const uploadParserChoice = ref('mineru')
const uploadSourceKbName = ref('')
const uploadElapsedSeconds = ref(0)
const uploadProgressIdleSeconds = ref(0)
const uploadConnectionIssue = ref(false)
const dragOver = ref(false)
let pollTimer = null
let uploadClockTimer = null
let lastProgressSignature = ''
let lastProgressChangedAt = 0
let knowledgeSelectionRevision = 0
let fileRequestRevision = 0

const showPreview = ref(false)
const previewFile = ref(null)
const previewLoading = ref(false)
const previewError = ref('')
const previewText = ref('')
const previewContentType = ref('')
const rawUrl = ref('')

const showDeleteConfirm = ref(false)
const deleteTarget = ref(null)
const deleting = ref(false)
const deleteSuccess = ref('')

const retrievalQuery = ref('')
const retrievalTopK = ref(5)
const retrievalThreshold = ref(0.3)
const retrievalMode = ref('basic')
const retrievalAttempted = ref(false)
const retrievalLoading = ref(false)
const retrievalRun = ref(null)
const retrievalError = ref('')
const selectedSubQuestion = ref(-1)  // -1 = 全部，0/1/2... = 对应子问题
let retrievalRequestRevision = 0
const showEvaluationSetup = ref(false)
const moduleNotice = ref('')
let noticeTimer = null

const criteria = reactive([
  { id: 'recall', name: 'Recall@K', description: '衡量相关内容是否被检索到。', group: '检索质量', icon: ScanSearch, enabled: true },
  { id: 'mrr', name: 'MRR', description: '衡量首个相关结果的排序位置。', group: '排序质量', icon: ListFilter, enabled: true },
  { id: 'relevance', name: '上下文相关性', description: '判断召回内容与问题的相关程度。', group: '语义质量', icon: Waypoints, enabled: true },
  { id: 'faithfulness', name: '回答忠实度', description: '检查回答是否得到上下文支持。', group: '生成质量', icon: CheckCircle2, enabled: true },
])

const ACCEPT_EXTS = [
  '.txt', '.md', '.pdf', '.docx', '.pptx', '.xlsx',
  '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif', '.tif', '.tiff', '.jp2',
]
const MINERU_EXTS = new Set([
  '.pdf', '.docx', '.pptx', '.xlsx', '.png', '.jpg', '.jpeg',
  '.bmp', '.webp', '.gif', '.tif', '.tiff', '.jp2',
])
const LOCAL_PARSER_EXTS = new Set([
  '.txt', '.md', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.bmp', '.webp',
])
const parserOptions = [
  { value: 'mineru', label: 'MinerU', description: '版面、表格与公式识别' },
  { value: 'auto', label: '自动推荐', description: '优先 MinerU，可自动回退' },
  { value: 'local', label: '本地解析', description: '适合纯文本和简单文档' },
]

const filteredKbs = computed(() => {
  const keyword = catalogQuery.value.trim().toLowerCase()
  if (!keyword) return kbList.value
  return kbList.value.filter((kb) => `${kb.name || ''} ${kb.description || ''}`.toLowerCase().includes(keyword))
})

const recentlyCreatedCount = computed(() => {
  const cutoff = Date.now() - 30 * 24 * 60 * 60 * 1000
  return kbList.value.filter((kb) => {
    const time = Date.parse(kb.created_at)
    return Number.isFinite(time) && time >= cutoff
  }).length
})

const completedFiles = computed(() => fileList.value.filter((file) => file.status === 'completed'))
const totalChunks = computed(() => fileList.value.reduce((sum, file) => sum + Number(file.chunk_count || 0), 0))
const totalCharacters = computed(() => fileList.value.reduce((sum, file) => sum + Number(file.char_count || 0), 0))
const enabledCriteria = computed(() => criteria.filter((criterion) => criterion.enabled))

const displayProgress = computed(() => {
  if (uploadPhase.value === 'transferring') return transferProgress.value
  if (uploadPhase.value === 'indexing') return Math.max(indexProgress.value, 5)
  if (uploadPhase.value === 'done' || uploadPhase.value === 'failed') return 100
  return 0
})

const progressLabel = computed(() => {
  if (uploadPhase.value === 'transferring') return '正在上传文件'
  if (uploadPhase.value === 'indexing') {
    if (uploadParser.value?.progress_message) return uploadParser.value.progress_message
    if (!uploadParser.value?.parser_name) return '已上传，等待分配解析器'
    const name = parserLabel(uploadParser.value)
    return indexProgress.value < 30
      ? `${name} 正在解析文档`
      : `${name} 解析完成，正在建立索引`
  }
  if (uploadPhase.value === 'done') return '处理完成'
  if (uploadPhase.value === 'failed') return '处理失败'
  return ''
})

const retrievalContract = computed(() => JSON.stringify({
  method: 'POST',
  endpoint: `/api/v1/knowledge/bases/${activeKb.value?.id || '<knowledge_base_id>'}/retrieval/test`,
  body: {
    query: retrievalQuery.value || '<用户问题>',
    top_k: retrievalTopK.value,
    score_threshold: retrievalThreshold.value,
    mode: retrievalMode.value,
  },
}, null, 2))

const hasRetrievalResult = computed(() => {
  const run = retrievalRun.value
  if (!run) return false
  if (run.mode === 'enhanced') return (run.knowledge_blocks?.length || 0) > 0
  return (run.results?.length || 0) > 0
})

// 按当前选中的子问题过滤知识块（-1 = 全部）
const filteredBlocks = computed(() => {
  const run = retrievalRun.value
  if (!run || run.mode !== 'enhanced') return []
  const blocks = run.knowledge_blocks || []
  if (selectedSubQuestion.value < 0) return blocks
  const subQuestions = run.query_decomposition?.sub_questions || []
  const target = subQuestions[selectedSubQuestion.value]
  if (!target) return blocks
  return blocks.filter((b) => (b.sub_questions || []).includes(target))
})

const graphPreviewNodes = computed(() => {
  const positions = [
    { x: 110, y: 82 }, { x: 340, y: 62 }, { x: 570, y: 82 },
    { x: 105, y: 268 }, { x: 340, y: 292 }, { x: 575, y: 268 },
  ]
  return fileList.value.slice(0, positions.length).map((file, index) => ({
    id: file.id,
    x: positions[index].x,
    y: positions[index].y,
    type: (file.file_type || 'FILE').toUpperCase(),
    label: truncate(file.filename, 10),
  }))
})

function formatNumber(value) {
  return Number(value || 0).toLocaleString('zh-CN')
}

function formatCompactNumber(value) {
  const number = Number(value || 0)
  if (number < 1000) return String(number)
  if (number < 10000) return `${(number / 1000).toFixed(1)}k`
  if (number < 100000000) return `${(number / 10000).toFixed(1)}万`
  return `${(number / 100000000).toFixed(1)}亿`
}

function formatSize(bytes) {
  const value = Number(bytes || 0)
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

function formatDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10)
  return new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(date)
}

function statusLabel(status) {
  return { completed: '已完成', pending: '等待中', processing: '处理中', failed: '失败' }[status] || status || '未知'
}

function shortId(value) {
  const text = String(value || '')
  return text.length > 12 ? `${text.slice(0, 8)}…` : text || '—'
}

function stageLabel(stage) {
  return {
    queued: '等待处理',
    parsing: '文档解析',
    chunking: '内容分块',
    indexing: '向量索引',
    graph: '知识图谱抽取',
    completed: '处理完成',
    failed: '处理失败',
  }[stage] || '准备处理'
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Number(totalSeconds || 0))
  if (seconds < 60) return `${seconds}秒`
  const minutes = Math.floor(seconds / 60)
  const remaining = seconds % 60
  return `${minutes}分${String(remaining).padStart(2, '0')}秒`
}

function parserLabel(file) {
  if (file?.parser_name === 'mineru') return 'MinerU'
  if (file?.parser_name === 'local') return '本地解析器'
  return file?.parser_name || '未记录'
}

function parserDetails(file) {
  if (!file?.parser_name) return '该文件尚未记录解析器信息'
  return [
    `解析器：${parserLabel(file)}`,
    file.parser_version ? `版本：${file.parser_version}` : '',
    file.parser_backend ? `后端：${file.parser_backend}` : '',
    file.parse_method ? `方式：${file.parse_method}` : '',
    file.parser_task_id ? `任务 ID：${file.parser_task_id}` : '',
    file.parser_warnings ? `警告：${file.parser_warnings}` : '',
  ].filter(Boolean).join('\n')
}

function shortCollectionName(value) {
  const text = String(value || '默认集合')
  return text.length > 22 ? `${text.slice(0, 20)}…` : text
}

function truncate(value, length) {
  const text = String(value || '')
  return text.length > length ? `${text.slice(0, length - 1)}…` : text
}

function formatSimilarity(value) {
  const score = Number(value)
  return Number.isFinite(score) ? score.toFixed(4) : '—'
}

function pageRange(hit) {
  const start = hit?.page_start
  const end = hit?.page_end
  if (start === null || start === undefined) return ''
  if (end === null || end === undefined || Number(end) === Number(start)) return `第 ${start} 页`
  return `第 ${start}–${end} 页`
}

function parserDisplayName(value) {
  return String(value || '').toLowerCase() === 'mineru' ? 'MinerU' : String(value || '')
}

function notify(message) {
  moduleNotice.value = message
  if (noticeTimer) clearTimeout(noticeTimer)
  noticeTimer = setTimeout(() => { moduleNotice.value = '' }, 3600)
}

function openEditKb() {
  editForm.name = activeKb.value?.name || ''
  editForm.description = activeKb.value?.description || ''
  editError.value = ''
  showEdit.value = true
}

async function saveKb() {
  editError.value = ''
  if (!editForm.name.trim()) {
    editError.value = '请输入知识库名称。'
    return
  }
  saving.value = true
  try {
    const updated = await api.patch(`/knowledge/bases/${activeKb.value.id}`, {
      name: editForm.name.trim(),
      description: editForm.description.trim(),
    })
    // 同步详情页标题与列表页条目
    if (activeKb.value) {
      activeKb.value.name = updated.name
      activeKb.value.description = updated.description
    }
    const idx = kbList.value.findIndex((kb) => kb.id === updated.id)
    if (idx !== -1) kbList.value[idx] = { ...kbList.value[idx], ...updated }
    showEdit.value = false
    notify('知识库信息已更新。')
  } catch (error) {
    editError.value = error.response?.data?.detail || '保存失败，请稍后重试。'
  } finally {
    saving.value = false
  }
}

function frontendOnly(action) {
  notify(`${action}已完成前端入口，后端接口将在后续阶段接入。`)
}

async function loadKbs() {
  loading.value = true
  catalogError.value = ''
  try {
    kbList.value = await api.get('/knowledge/bases')
  } catch (error) {
    catalogError.value = error.response?.data?.detail || '无法连接知识库服务，请稍后重试。'
  } finally {
    loading.value = false
  }
}

async function loadFiles(kbId = activeKb.value?.id, selectionRevision = knowledgeSelectionRevision) {
  if (!kbId) return
  const requestRevision = ++fileRequestRevision
  filesLoading.value = true
  try {
    const files = await api.get(`/knowledge/bases/${kbId}/files`)
    if (
      requestRevision === fileRequestRevision
      && selectionRevision === knowledgeSelectionRevision
      && String(activeKb.value?.id || '') === String(kbId)
    ) {
      fileList.value = files
    }
  } catch (error) {
    if (
      requestRevision === fileRequestRevision
      && selectionRevision === knowledgeSelectionRevision
      && String(activeKb.value?.id || '') === String(kbId)
    ) {
      fileList.value = []
      notify(error.response?.data?.detail || '文件列表加载失败。')
    }
  } finally {
    if (requestRevision === fileRequestRevision) filesLoading.value = false
  }
}

async function createKb() {
  createError.value = ''
  if (!newKb.name.trim()) {
    createError.value = '请输入知识库名称。'
    return
  }
  creating.value = true
  try {
    const created = await api.post('/knowledge/bases', { name: newKb.name.trim(), description: newKb.description.trim() })
    showCreate.value = false
    newKb.name = ''
    newKb.description = ''
    await loadKbs()
    const target = kbList.value.find((kb) => kb.id === created?.id) || created
    if (target?.id) await selectKb(target)
  } catch (error) {
    createError.value = error.response?.data?.detail || '创建失败，请稍后重试。'
  } finally {
    creating.value = false
  }
}

async function selectKb(kb, syncRoute = true) {
  const selectionRevision = ++knowledgeSelectionRevision
  resetRetrievalState()
  activeKb.value = kb
  fileList.value = []
  activeTab.value = validTabs.has(String(route.query.tab)) ? String(route.query.tab) : 'files'
  if (syncRoute) {
    await router.replace({ query: { ...route.query, kb: kb.id, tab: activeTab.value, file: undefined } })
  }
  if (selectionRevision !== knowledgeSelectionRevision) return
  await loadFiles(kb.id, selectionRevision)
}

async function leaveKb() {
  knowledgeSelectionRevision += 1
  fileRequestRevision += 1
  resetRetrievalState()
  activeKb.value = null
  activeTab.value = 'files'
  fileList.value = []
  await router.replace({ query: { ...route.query, kb: undefined, tab: undefined, file: undefined } })
}

async function selectTab(tabId) {
  if (!validTabs.has(tabId)) return
  activeTab.value = tabId
  await router.replace({ query: { ...route.query, kb: activeKb.value?.id, tab: tabId, file: undefined } })
}

async function copyKbId() {
  const value = String(activeKb.value?.id || '')
  if (!value) return
  try {
    await navigator.clipboard.writeText(value)
  } catch {
    const textarea = document.createElement('textarea')
    textarea.value = value
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    textarea.remove()
  }
  notify('知识库 ID 已复制。')
}

function resetRetrievalState() {
  retrievalRequestRevision += 1
  retrievalLoading.value = false
  retrievalAttempted.value = false
  retrievalRun.value = null
  retrievalError.value = ''
  selectedSubQuestion.value = -1
}

async function runRetrievalPreview() {
  const query = retrievalQuery.value.trim()
  const kbId = String(activeKb.value?.id || '')
  if (!query) {
    notify('请先输入一个用于检索测试的问题。')
    return
  }
  if (!kbId) {
    notify('请先选择一个知识库。')
    return
  }

  const requestRevision = ++retrievalRequestRevision
  retrievalAttempted.value = true
  retrievalLoading.value = true
  retrievalRun.value = null
  retrievalError.value = ''

  try {
    const response = await api.post(`/knowledge/bases/${kbId}/retrieval/test`, {
      query,
      top_k: retrievalTopK.value,
      score_threshold: retrievalThreshold.value,
      mode: retrievalMode.value,
    })
    if (
      requestRevision === retrievalRequestRevision
      && String(activeKb.value?.id || '') === kbId
    ) {
      retrievalRun.value = response
    }
  } catch (error) {
    if (
      requestRevision === retrievalRequestRevision
      && String(activeKb.value?.id || '') === kbId
    ) {
      retrievalError.value = error.response?.data?.detail || '检索服务暂时不可用，请稍后重试。'
    }
  } finally {
    if (requestRevision === retrievalRequestRevision) retrievalLoading.value = false
  }
}

function toggleCriterion(criterion) {
  criterion.enabled = !criterion.enabled
  notify(`已在前端${criterion.enabled ? '启用' : '停用'}「${criterion.name}」，尚未保存到后端。`)
}

function openEvaluationSetup() {
  if (!completedFiles.value.length) {
    notify('至少需要一个已完成索引的文件才能准备评估。')
    return
  }
  showEvaluationSetup.value = true
}

function fileExtension(file) {
  if (!file?.name || !file.name.includes('.')) return ''
  return `.${file.name.split('.').pop().toLowerCase()}`
}

function parserOptionAvailable(parser, file = uploadFile.value) {
  if (!file) return true
  const extension = fileExtension(file)
  if (parser === 'mineru') return MINERU_EXTS.has(extension)
  if (parser === 'local') return LOCAL_PARSER_EXTS.has(extension)
  return ACCEPT_EXTS.includes(extension)
}

function setUploadFile(file) {
  uploadFile.value = file || null
  if (!file || parserOptionAvailable(uploadParserChoice.value, file)) return
  uploadParserChoice.value = MINERU_EXTS.has(fileExtension(file)) ? 'mineru' : 'local'
}

function onFileSelect(event) {
  setUploadFile(event.target.files?.[0] || null)
  uploadMsg.value = ''
}

function onDragOver(event) {
  if (uploading.value) return
  dragOver.value = true
  event.dataTransfer.dropEffect = 'copy'
}

function onDragLeave() {
  dragOver.value = false
}

function onDrop(event) {
  dragOver.value = false
  if (uploading.value) return
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  const extension = `.${(file.name.split('.').pop() || '').toLowerCase()}`
  if (!ACCEPT_EXTS.includes(extension)) {
    uploadMsg.value = `不支持 ${extension} 文件，请选择：${ACCEPT_EXTS.join(' ')}`
    uploadOk.value = false
    return
  }
  setUploadFile(file)
  uploadMsg.value = ''
}

function closeUpload() {
  if (uploading.value && uploadPhase.value === 'transferring') return
  showUpload.value = false
  dragOver.value = false
  if (!uploading.value) {
    uploadFile.value = null
    uploadMsg.value = ''
    uploadPhase.value = 'idle'
    transferProgress.value = 0
    indexProgress.value = 0
    uploadParser.value = null
    uploadParserChoice.value = 'mineru'
  }
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function startUploadClock() {
  if (uploadClockTimer) clearInterval(uploadClockTimer)
  uploadElapsedSeconds.value = 0
  uploadProgressIdleSeconds.value = 0
  lastProgressChangedAt = Date.now()
  uploadClockTimer = setInterval(() => {
    uploadElapsedSeconds.value += 1
    uploadProgressIdleSeconds.value = Math.floor((Date.now() - lastProgressChangedAt) / 1000)
  }, 1000)
}

function stopUploadClock() {
  if (uploadClockTimer) {
    clearInterval(uploadClockTimer)
    uploadClockTimer = null
  }
}

async function doUpload() {
  if (!uploadFile.value || !activeKb.value) return
  uploading.value = true
  uploadMsg.value = ''
  uploadOk.value = false
  uploadPhase.value = 'transferring'
  transferProgress.value = 0
  indexProgress.value = 0
  uploadParser.value = null
  uploadConnectionIssue.value = false
  lastProgressSignature = ''
  startUploadClock()
  const kbId = activeKb.value.id
  uploadSourceKbName.value = activeKb.value.name || '原知识库'
  try {
    const formData = new FormData()
    formData.append('file', uploadFile.value)
    formData.append('parser', uploadParserChoice.value)
    const uploadResult = await api.upload(`/knowledge/bases/${kbId}/upload`, formData, (event) => {
      if (event.total) transferProgress.value = Math.round((event.loaded / event.total) * 100)
    })
    uploadPhase.value = 'indexing'
    await pollIndexing(kbId, uploadResult.file_id)
  } catch (error) {
    uploadPhase.value = 'failed'
    uploadMsg.value = error.response?.data?.detail || '上传失败，请稍后重试。'
    uploadOk.value = false
    uploading.value = false
    stopUploadClock()
  }
}

async function pollIndexing(kbId, fileId) {
  stopPolling()
  return new Promise((resolve) => {
    pollTimer = setInterval(async () => {
      try {
        const files = await api.get(`/knowledge/bases/${kbId}/files`)
        if (String(activeKb.value?.id || '') === String(kbId)) {
          fileList.value = files
        }
        const target = files.find((file) => file.id === fileId)
        if (!target) return
        uploadConnectionIssue.value = false
        uploadParser.value = target
        const progressSignature = [
          target.progress,
          target.processing_stage,
          target.progress_current,
          target.progress_total,
          target.progress_message,
        ].join('|')
        if (progressSignature !== lastProgressSignature) {
          lastProgressSignature = progressSignature
          lastProgressChangedAt = Date.now()
          uploadProgressIdleSeconds.value = 0
        }
        if (target.status === 'processing' || target.status === 'pending') {
          indexProgress.value = target.progress ?? 0
          return
        }
        stopPolling()
        const latest = target
        if (latest?.status === 'failed') {
          uploadPhase.value = 'failed'
          uploadMsg.value = `索引失败：${latest.error_message || '未知错误'}`
          uploadOk.value = false
        } else {
          uploadPhase.value = 'done'
          const parser = parserLabel(latest)
          const version = latest?.parser_version ? ` ${latest.parser_version}` : ''
          uploadMsg.value = `上传完成，${parser}${version} 已解析并建立 ${latest?.chunk_count ?? 0} 个分块。`
          uploadOk.value = true
          uploadFile.value = null
          notify(`「${uploadSourceKbName.value}」中的文件索引已完成。`)
        }
        uploading.value = false
        stopUploadClock()
        resolve()
      } catch {
        uploadConnectionIssue.value = true
        // 短暂网络抖动时保留轮询，下一轮继续尝试。
      }
    }, 1000)
  })
}

async function openPreview(file) {
  if (file.status !== 'completed') return
  closeObjectUrl()
  showPreview.value = true
  previewFile.value = file
  previewLoading.value = true
  previewError.value = ''
  previewText.value = ''
  previewContentType.value = ''
  try {
    const kbId = activeKb.value.id
    if (['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'webp'].includes(file.file_type)) {
      const { data: blob } = await api.getBlob(`/knowledge/bases/${kbId}/files/${file.id}/raw`)
      rawUrl.value = URL.createObjectURL(blob)
      previewContentType.value = 'binary'
    } else {
      const data = await api.get(`/knowledge/bases/${kbId}/files/${file.id}/preview`)
      previewContentType.value = data.content_type
      previewText.value = data.text_content || '（空文件）'
    }
  } catch (error) {
    previewError.value = error.response?.data?.detail || error.message || '预览失败。'
  } finally {
    previewLoading.value = false
  }
}

function closeObjectUrl() {
  if (rawUrl.value) URL.revokeObjectURL(rawUrl.value)
  rawUrl.value = ''
}

function closePreview() {
  showPreview.value = false
  previewFile.value = null
  previewText.value = ''
  closeObjectUrl()
}

function confirmDelete(file) {
  deleteTarget.value = file
  showDeleteConfirm.value = true
}

async function doDelete() {
  if (!deleteTarget.value || !activeKb.value) return
  deleting.value = true
  deleteSuccess.value = ''
  const filename = deleteTarget.value.filename
  try {
    await api.delete(`/knowledge/bases/${activeKb.value.id}/files/${deleteTarget.value.id}`)
    showDeleteConfirm.value = false
    deleteTarget.value = null
    deleteSuccess.value = `「${filename}」已删除。`
    await loadFiles()
    setTimeout(() => { deleteSuccess.value = '' }, 3000)
  } catch (error) {
    notify(error.response?.data?.detail || '删除失败，请稍后重试。')
  } finally {
    deleting.value = false
  }
}

const reindexingIds = ref([])

async function reindexFile(file) {
  if (!activeKb.value || reindexingIds.value.includes(file.id)) return
  reindexingIds.value = [...reindexingIds.value, file.id]
  try {
    await api.post(`/knowledge/bases/${activeKb.value.id}/files/${file.id}/reindex`)
    notify(`「${file.filename}」已开始重新索引。`)
    await loadFiles()
    // 后台轮询完成状态（不阻塞：即使被打断导致文件卡在 processing，
    // 按钮也能立即恢复可点，由 pollReindex 超时兜底结束轮询）
    pollReindex(activeKb.value.id, file.id).then(async () => {
      await loadFiles()
      notify(`「${file.filename}」重新索引完成。`)
    })
  } catch (error) {
    notify(error.response?.data?.detail || '重新索引失败，请稍后重试。')
  } finally {
    reindexingIds.value = reindexingIds.value.filter((id) => id !== file.id)
  }
}

function pollReindex(kbId, fileId, timeoutMs = 10 * 60 * 1000) {
  return new Promise((resolve) => {
    const start = Date.now()
    const timer = setInterval(async () => {
      // 超时兜底：reindex 被打断（进程重启等）时文件会永久停在 processing，
      // 无超时会让轮询永不结束、按钮永久 disabled。
      if (Date.now() - start > timeoutMs) {
        clearInterval(timer)
        resolve()
        return
      }
      try {
        const files = await api.get(`/knowledge/bases/${kbId}/files`)
        if (String(activeKb.value?.id || '') === String(kbId)) {
          fileList.value = files
        }
        const target = files.find((f) => f.id === fileId)
        if (!target) { clearInterval(timer); resolve(); return }
        if (target.status !== 'processing' && target.status !== 'pending') {
          clearInterval(timer); resolve(); return
        }
      } catch {
        clearInterval(timer); resolve()
      }
    }, 2000)
  })
}

async function applyRouteQuery() {
  const kbId = typeof route.query.kb === 'string' ? route.query.kb : ''
  const tab = typeof route.query.tab === 'string' && validTabs.has(route.query.tab) ? route.query.tab : 'files'
  const fileId = typeof route.query.file === 'string' ? route.query.file : ''
  if (!kbId) return
  if (!kbList.value.length) await loadKbs()
  const kb = kbList.value.find((item) => String(item.id) === kbId)
  if (!kb) return
  activeTab.value = tab
  if (activeKb.value?.id !== kb.id) await selectKb(kb, false)
  if (fileId) {
    const file = fileList.value.find((item) => String(item.id) === fileId)
    if (file) await openPreview(file)
  }
}

onMounted(async () => {
  await loadKbs()
  await applyRouteQuery()
})

watch(() => [route.query.kb, route.query.tab, route.query.file], () => {
  if (route.path === '/knowledge') applyRouteQuery()
})

onBeforeUnmount(() => {
  stopPolling()
  stopUploadClock()
  closeObjectUrl()
  if (noticeTimer) clearTimeout(noticeTimer)
})
</script>

<style scoped src="../styles/knowledge-workspace.css"></style>
