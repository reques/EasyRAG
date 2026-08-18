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
          <button class="kbw-primary-button" @click="frontendOnly('知识库编辑')">
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
              <div><ListFilter :size="17" /><strong>召回结果</strong></div>
              <span v-if="retrievalRun">{{ retrievalRun.total }} 条 · {{ retrievalRun.elapsed_ms }} ms</span>
              <span v-else>0 条</span>
            </div>
            <div v-if="retrievalLoading" class="kbw-result-empty">
              <LoaderCircle :size="28" class="spin" />
              <strong>正在执行检索</strong>
              <span>正在生成查询向量并从当前知识库召回内容。</span>
            </div>
            <div v-else-if="retrievalError" class="kbw-result-empty">
              <CircleAlert :size="28" />
              <strong>检索失败</strong>
              <span>{{ retrievalError }}</span>
              <button class="kbw-secondary-button" @click="runRetrievalPreview">重新测试</button>
            </div>
            <div v-else-if="!retrievalRun?.results?.length" class="kbw-result-empty">
              <Waypoints :size="28" />
              <strong>{{ retrievalAttempted ? '没有符合条件的召回结果' : '等待一次测试查询' }}</strong>
              <span>{{ retrievalAttempted ? '可以降低最低相似度或换一个更贴近文档内容的问题。' : '结果区将展示命中文件、分块正文、相似度、排名和耗时。' }}</span>
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
              <p>从已入库 chunks 抽取实体与关系写入 Neo4j，建立 Milvus 语义索引，检索时与向量结果 RRF 融合。</p>
            </div>
            <span class="kbw-module-status" :class="{ 'is-online': graphConfig.neo4j_connected }">
              <CircleDashed :size="14" />
              Neo4j {{ graphConfig.neo4j_connected ? '已连接' : '未连接' }}
            </span>
          </div>

          <div class="kbw-table-toolbar kbw-graph-toolbar">
            <label class="kbw-graph-tool-label">抽取器
              <select v-model="graphExtractor" class="kbw-select">
                <option value="llm">LLM 抽取器</option>
              </select>
            </label>
            <button
              class="kbw-primary-button"
              :disabled="graphBuilding || !graphConfig.neo4j_connected"
              @click="startGraphBuild()"
            >
              <Play :size="14" /> {{ graphBuilding ? '构建中…' : '开始构建' }}
            </button>
            <button class="kbw-secondary-button" :disabled="graphStatusLoading" @click="loadGraphStatus()">
              <RefreshCw :size="14" :class="{ spin: graphStatusLoading }" /> 刷新状态
            </button>
            <button
              class="kbw-secondary-button"
              :disabled="!graphConfig.neo4j_connected"
              title="查看本知识库已抽取的实体名称，点击名称检索其子图"
              @click="openGraphEntities()"
            >
              <Database :size="14" /> Neo4j 实体
            </button>
            <button class="kbw-secondary-button is-danger" @click="resetGraph()">
              <Trash2 :size="14" /> 重置图谱
            </button>
          </div>

          <div v-if="graphStatus.run && graphStatus.run.status === 'running'" class="kbw-inline-notice">
            <LoaderCircle :size="14" class="spin" />
            正在构建：已处理 {{ graphStatus.run.processed_chunks }}/{{ graphStatus.run.total_chunks }} 个 chunks（抽取器：{{ graphStatus.run.extractor }}）
          </div>
          <div v-else-if="graphStatus.run && graphStatus.run.status === 'failed'" class="kbw-inline-notice is-error">
            <CircleAlert :size="14" />
            构建失败：{{ graphStatus.run.error_message || '未知错误' }}
          </div>
          <div v-else-if="graphStatus.run && graphStatus.run.status === 'completed'" class="kbw-inline-notice is-ok">
            <CheckCircle2 :size="14" />
            最近构建完成：{{ graphStatus.run.entities_found }} 实体 / {{ graphStatus.run.relations_found }} 关系，
            索引 {{ graphStatus.run.entities_indexed }} 实体 + {{ graphStatus.run.relations_indexed }} 三元组
          </div>

          <div class="kbw-metric-grid">
            <article>
              <span class="kbw-metric-icon"><Network :size="17" /></span>
              <div><strong>{{ graphStatus.neo4j.entities ?? '—' }}</strong><span>Neo4j 实体</span></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><Waypoints :size="17" /></span>
              <div><strong>{{ graphStatus.neo4j.relations ?? '—' }}</strong><span>Neo4j 关系</span></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><Database :size="17" /></span>
              <div><strong>{{ graphStatus.pg_entities }}</strong><span>PostgreSQL 实体</span></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><Layers3 :size="17" /></span>
              <div><strong>{{ graphStatus.pg_relations }}</strong><span>PostgreSQL 关系</span></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><Braces :size="17" /></span>
              <div><strong>{{ graphStatus.indexed }}</strong><span>Milvus 语义索引</span></div>
            </article>
            <article>
              <span class="kbw-metric-icon"><History :size="17" /></span>
              <div>
                <strong>{{ graphStatus.run ? runStatusLabel(graphStatus.run.status) : '未构建' }}</strong>
                <span>最近构建</span>
              </div>
            </article>
          </div>

          <div class="kbw-graph-layout">
            <article class="kbw-panel-card kbw-graph-canvas">
              <div class="kbw-panel-title">
                <div><ScanSearch :size="17" /><strong>子图搜索</strong></div>
                <div class="kbw-graph-search">
                  <input
                    v-model="graphQuery"
                    class="kbw-search-input"
                    placeholder="输入实体名关键词，回车搜索"
                    @keyup.enter="searchGraph()"
                  />
                  <select v-model="graphDepth" class="kbw-select" title="子图扩展深度">
                    <option :value="1">1 跳</option>
                    <option :value="2">2 跳</option>
                    <option :value="3">3 跳</option>
                  </select>
                  <button class="kbw-primary-button" :disabled="!graphQuery.trim()" @click="searchGraph()">
                    <Search :size="14" /> 搜索
                  </button>
                </div>
              </div>
              <div v-if="graphSubgraph.nodes.length" class="kbw-graph-stage">
                <svg viewBox="0 0 680 400" role="img" aria-label="图谱子图">
                  <line
                    v-for="(edge, i) in graphSubgraph.edges"
                    :key="`edge-${i}`"
                    :x1="graphNodePositions[edge.source]?.x ?? 340"
                    :y1="graphNodePositions[edge.source]?.y ?? 200"
                    :x2="graphNodePositions[edge.target]?.x ?? 340"
                    :y2="graphNodePositions[edge.target]?.y ?? 200"
                    class="graph-edge"
                  />
                  <g v-for="node in graphSubgraph.nodes" :key="node.id" :class="{ 'graph-root': node.id === graphCenter }">
                    <circle
                      :cx="graphNodePositions[node.id]?.x ?? 340"
                      :cy="graphNodePositions[node.id]?.y ?? 200"
                      r="30"
                    />
                    <text
                      :x="graphNodePositions[node.id]?.x ?? 340"
                      :y="(graphNodePositions[node.id]?.y ?? 200) - 36"
                      class="graph-node-label"
                    >{{ truncate(node.name, 12) }}</text>
                    <text
                      :x="graphNodePositions[node.id]?.x ?? 340"
                      :y="(graphNodePositions[node.id]?.y ?? 200) + 8"
                      class="graph-node-type"
                    >{{ node.entity_type || 'concept' }}</text>
                  </g>
                </svg>
              </div>
              <div v-else class="kbw-result-empty">
                <Network :size="28" />
                <strong>{{ graphSubgraph.entities.length ? '未找到关联子图' : '搜索知识图谱子图' }}</strong>
                <span>输入实体名关键词，展示以命中实体为中心的 1-3 跳邻居。</span>
              </div>
            </article>
            <aside class="kbw-panel-card kbw-graph-sidebar">
              <div class="kbw-panel-title">
                <div><ListFilter :size="17" /><strong>命中实体</strong></div>
              </div>
              <ul v-if="graphSubgraph.entities.length" class="kbw-graph-entity-list">
                <li v-for="entity in graphSubgraph.entities" :key="entity.name">
                  <strong>{{ entity.name }}</strong>
                  <span>{{ entity.entity_type || 'concept' }}</span>
                </li>
              </ul>
              <p v-else class="kbw-sidebar-note">暂无命中实体。</p>
              <div class="kbw-panel-title" style="margin-top: 14px;">
                <div><Waypoints :size="17" /><strong>子图关系</strong></div>
              </div>
              <ul v-if="graphSubgraph.edges.length" class="kbw-graph-entity-list">
                <li v-for="(edge, i) in graphSubgraph.edges" :key="i">
                  <span class="kbw-graph-rel">
                    {{ edge.relation_type }}
                    <em>{{ graphSubgraph.edges.length > 8 ? '' : edge.source.split(':').pop() + ' → ' + edge.target.split(':').pop() }}</em>
                  </span>
                </li>
              </ul>
              <p v-else class="kbw-sidebar-note">暂无子图关系。</p>
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

    <div v-if="graphEntityModal" class="modal-overlay" @click.self="graphEntityModal = false">
      <div class="modal kbw-modal kbw-graph-entity-modal">
        <div class="kbw-modal-heading">
          <div><span><Database :size="18" /></span><div><h3>Neo4j 实体</h3><p>本知识库已抽取的实体名称，点击即可检索其子图</p></div></div>
          <button @click="graphEntityModal = false"><X :size="17" /></button>
        </div>
        <div class="kbw-graph-entity-search">
          <Search :size="14" />
          <input v-model="graphEntityFilter" type="text" placeholder="过滤实体名称…" />
          <span v-if="graphEntityFilter" class="kbw-graph-entity-count">{{ filteredGraphEntities.length }} / {{ graphEntities.length }}</span>
        </div>
        <div class="kbw-graph-entity-list">
          <button
            v-for="entity in filteredGraphEntities"
            :key="entity.name"
            @click="selectGraphEntity(entity.name)"
          >
            <span>{{ entity.name }}</span>
            <small>{{ entity.entity_type || 'concept' }}</small>
          </button>
          <div v-if="!filteredGraphEntities.length" class="kbw-graph-entity-empty">
            {{ graphEntities.length ? '没有匹配的实体' : '暂无实体 —— 请先点击"开始构建"生成图谱' }}
          </div>
        </div>
      </div>
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
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
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
const fileList = ref([])
const loading = ref(true)
const filesLoading = ref(false)
const catalogError = ref('')
const catalogQuery = ref('')

const showCreate = ref(false)
const newKb = reactive({ name: '', description: '' })
const creating = ref(false)
const createError = ref('')

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
const retrievalAttempted = ref(false)
const retrievalLoading = ref(false)
const retrievalRun = ref(null)
const retrievalError = ref('')
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
  },
}, null, 2))

// ── 知识图谱 (GraphRAG 阶段 5) ─────────────────────────────────────────────
const graphConfig = reactive({
  graph_enabled: false,
  neo4j_uri: '',
  neo4j_connected: false,
  extractors: [],
  entity_collection: '',
})
const graphStatus = reactive({
  run: null,
  neo4j: {},
  indexed: 0,
  pg_entities: 0,
  pg_relations: 0,
})
const graphExtractor = ref('llm')
const graphBuilding = ref(false)
const graphStatusLoading = ref(false)
const graphQuery = ref('')
const graphDepth = ref(1)
const graphSubgraph = reactive({ entities: [], nodes: [], edges: [] })
const graphCenter = ref('')
let graphPollTimer = null

async function loadGraphConfig() {
  if (!activeKb.value) return
  try {
    Object.assign(graphConfig, await api.get(`/knowledge/bases/${activeKb.value.id}/graph/config`))
  } catch (error) {
    notify(error.response?.data?.detail || '图谱配置加载失败。')
  }
}

// 图谱面板统一加载入口：点击 Tab / 选择知识库 / F5 刷新路由恢复时都调用
function loadGraphPanel() {
  if (activeTab.value !== 'graph' || !activeKb.value) return
  loadGraphConfig()
  loadGraphStatus()
  loadGraphEntities()
}

const graphEntities = ref([])
const graphEntityModal = ref(false)
const graphEntityFilter = ref('')

const filteredGraphEntities = computed(() => {
  const keyword = graphEntityFilter.value.trim()
  if (!keyword) return graphEntities.value
  return graphEntities.value.filter((entity) => entity.name.includes(keyword))
})

async function loadGraphEntities() {
  if (!activeKb.value) return
  try {
    const data = await api.get(`/knowledge/bases/${activeKb.value.id}/graph/entities`, {
      limit: 100,
    })
    graphEntities.value = data.entities || []
  } catch (error) {
    // Neo4j 未连接/未构建时保持空列表（面板已有连接状态提示）
    graphEntities.value = []
  }
}

function openGraphEntities() {
  graphEntityFilter.value = ''
  graphEntityModal.value = true
  loadGraphEntities()
}

function selectGraphEntity(name) {
  graphEntityModal.value = false
  graphQuery.value = name
  searchGraph()
}

async function loadGraphStatus() {
  if (!activeKb.value) return
  graphStatusLoading.value = true
  try {
    const data = await api.get(`/knowledge/bases/${activeKb.value.id}/graph/status`)
    graphStatus.run = data.run
    graphStatus.neo4j = data.neo4j || {}
    graphStatus.indexed = data.indexed || 0
    graphStatus.pg_entities = data.pg_entities || 0
    graphStatus.pg_relations = data.pg_relations || 0
    // 页面刷新后若看到 running 且没有活动轮询，自动恢复轮询
    if (data.run && data.run.status === 'running' && !graphPollTimer) {
      startGraphPolling()
    }
  } catch (error) {
    notify(error.response?.data?.detail || '图谱状态加载失败。')
  } finally {
    graphStatusLoading.value = false
  }
}

async function startGraphBuild() {
  if (!activeKb.value || graphBuilding.value) return
  graphBuilding.value = true
  try {
    const form = new FormData()
    form.append('extractor', graphExtractor.value)
    await api.post(`/knowledge/bases/${activeKb.value.id}/graph/build`, form)
    notify('图谱构建已开始，正在后台抽取实体与关系…')
    await loadGraphStatus()
    startGraphPolling()
  } catch (error) {
    notify(error.response?.data?.detail || '图谱构建启动失败。')
  } finally {
    graphBuilding.value = false
  }
}

function startGraphPolling() {
  stopGraphPolling()
  // 超时保护：3s × 200 = 10 分钟，防止"僵尸 running"导致无限轮询
  const MAX_GRAPH_POLL_TICKS = 200
  let ticks = 0
  graphPollTimer = setInterval(async () => {
    ticks += 1
    await loadGraphStatus()
    if (graphStatus.run && ['completed', 'failed'].includes(graphStatus.run.status)) {
      stopGraphPolling()
    } else if (ticks >= MAX_GRAPH_POLL_TICKS) {
      stopGraphPolling()
      notify('图谱构建超时（10 分钟），可能已被中断。请刷新状态后重新构建。')
    }
  }, 3000)
}

function stopGraphPolling() {
  if (graphPollTimer) {
    clearInterval(graphPollTimer)
    graphPollTimer = null
  }
}

async function resetGraph() {
  if (!activeKb.value) return
  if (!confirm('确定重置该知识库的图谱数据？将清空 Neo4j 子图、Milvus 语义索引、PostgreSQL 图谱记录与内存缓存。')) return
  try {
    await api.delete(`/knowledge/bases/${activeKb.value.id}/graph`)
    graphSubgraph.entities = []
    graphSubgraph.nodes = []
    graphSubgraph.edges = []
    graphCenter.value = ''
    await loadGraphStatus()
    notify('图谱数据已重置。')
  } catch (error) {
    notify(error.response?.data?.detail || '图谱重置失败。')
  }
}

async function searchGraph() {
  const keyword = graphQuery.value.trim()
  if (!activeKb.value || !keyword) return
  try {
    const data = await api.get(`/knowledge/bases/${activeKb.value.id}/graph/search`, {
      q: keyword,
      depth: graphDepth.value,
    })
    graphSubgraph.entities = data.entities || []
    graphSubgraph.nodes = data.nodes || []
    graphSubgraph.edges = data.edges || []
    graphCenter.value = graphSubgraph.entities[0]?.name || ''
  } catch (error) {
    notify(error.response?.data?.detail || '子图搜索失败。')
  }
}

const graphNodePositions = computed(() => {
  const positions = {}
  const nodes = graphSubgraph.nodes
  const centerId = graphCenter.value
  const cx = 340
  const cy = 200
  const others = nodes.filter((node) => node.id !== centerId)
  others.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / Math.max(others.length, 1) - Math.PI / 2
    const radius = 110 + (index % 3) * 45
    positions[node.id] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    }
  })
  if (centerId) positions[centerId] = { x: cx, y: cy }
  return positions
})

function runStatusLabel(status) {
  return { pending: '等待中', running: '构建中', completed: '已完成', failed: '失败' }[status] || status || '未构建'
}

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
  loadGraphPanel()
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
  if (tabId === 'graph') {
    loadGraphPanel()
  }
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
  loadGraphPanel()
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
  stopGraphPolling()
  closeObjectUrl()
  if (noticeTimer) clearTimeout(noticeTimer)
})
</script>

<style scoped src="../styles/knowledge-workspace.css"></style>
