const scenarios = {
  qa: {
    context: '知识库 / 团队知识空间',
    question: '新同事入职，需要准备哪些资料？',
    trace: ['检索入职流程与团队文档', '融合相关片段，整理办理清单'],
    intro: '可以先准备这三类资料：',
    items: [['身份与人事材料', '按入职指南提交所需信息。'], ['账号与开发环境', '申请权限，完成环境配置。'], ['团队与项目文档', '阅读协作规范和项目说明。']],
    ending: '具体要求请以团队当前的入职文档为准。',
    sources: ['入职指南.md', '团队协作手册.pdf'],
  },
  graph: {
    context: '知识图谱 / 关联探索',
    question: '知识库里的文档、实体和关系是怎么关联的？',
    trace: ['提取问题中的实体，检索关联关系', '结合图谱路径，回到原始文档寻找依据'],
    intro: '可以沿着三个层次理解这些关联：',
    items: [['文档片段', '保存原始内容，为检索和引用提供依据。'], ['知识实体', '提取内容中的人、组织、概念等对象。'], ['实体关系', '连接相关实体，辅助发现跨文档的知识线索。']],
    ending: '图谱帮助发现关联，原文为回答提供可核查的上下文。',
    sources: ['知识库建设方案.pdf', '项目术语表.md'],
  },
  research: {
    context: 'Deep Research / 任务协作',
    question: '帮我整理一个知识库落地方案，需要考虑哪些方面？',
    trace: ['主 Agent 拆解问题，委派研究子任务', '子 Agent 按需使用知识库与工具，汇总研究结果'],
    intro: '可以从以下三个方面推进调研：',
    items: [['数据准备', '明确文档来源、解析方式和更新流程。'], ['检索设计', '比较语义、关键词与图谱检索的适用场景。'], ['验证与迭代', '建立问题集，结合检索指标与人工审查评估效果。']],
    ending: '研究过程可通过任务进度与结构化黑板协同推进。',
    sources: ['数据准备子任务', '检索设计子任务', '评估方案子任务'],
  },
};

const tabs = [...document.querySelectorAll('[data-scenario]')];
function selectScenario(tab) {
  const scenario = scenarios[tab.dataset.scenario];
  tabs.forEach((item) => {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
  });
  document.querySelector('#demo-panel').setAttribute('aria-labelledby', tab.id);
  document.querySelector('#demo-context').textContent = scenario.context;
  document.querySelector('#demo-question').textContent = scenario.question;
  document.querySelector('#demo-trace').replaceChildren(...scenario.trace.map((text) => {
    const item = document.createElement('li');
    item.textContent = text;
    return item;
  }));
  const intro = document.createElement('p');
  intro.textContent = scenario.intro;
  const list = document.createElement('ul');
  scenario.items.forEach(([title, text]) => {
    const item = document.createElement('li');
    const strong = document.createElement('strong');
    strong.textContent = title;
    item.append(strong, `：${text}`);
    list.append(item);
  });
  const ending = document.createElement('p');
  ending.textContent = scenario.ending;
  document.querySelector('#demo-answer').replaceChildren(intro, list, ending);
  document.querySelector('#demo-sources').replaceChildren(...scenario.sources.map((text) => {
    const source = document.createElement('span');
    source.textContent = `↗ ${text}`;
    return source;
  }));
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectScenario(tab));
  tab.addEventListener('keydown', (event) => {
    let target;
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') target = (index + 1) % tabs.length;
    if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') target = 0;
    if (event.key === 'End') target = tabs.length - 1;
    if (target === undefined) return;
    event.preventDefault();
    selectScenario(tabs[target]);
    tabs[target].focus();
  });
});

const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
const narrowViewport = window.matchMedia('(max-width: 760px)');
function setTabOrientation() {
  document.querySelector('[role="tablist"]').setAttribute('aria-orientation', narrowViewport.matches ? 'horizontal' : 'vertical');
}
setTabOrientation();
narrowViewport.addEventListener('change', setTabOrientation);

const motionToggle = document.querySelector('#motion-toggle');
const workflowImage = document.querySelector('#workflow-image');
const workflowSource = document.querySelector('#workflow-picture source');
let motionPaused = motionPreference.matches;
function updateMotion() {
  document.body.classList.toggle('motion-paused', motionPaused);
  workflowImage.src = motionPaused ? 'assets/overview-poster.png' : 'assets/knowledge-flow.gif';
  // Match picture's source too, so an explicit user choice can override the OS default.
  workflowSource.srcset = workflowImage.getAttribute('src');
  motionToggle.setAttribute('aria-pressed', String(motionPaused));
  motionToggle.replaceChildren(document.createTextNode(motionPaused ? '播放动效 ' : '暂停动效 '));
  const symbol = document.createElement('span');
  symbol.setAttribute('aria-hidden', 'true');
  symbol.textContent = motionPaused ? '▷' : 'Ⅱ';
  motionToggle.append(symbol);
}
motionToggle.hidden = false;
updateMotion();
motionToggle.addEventListener('click', () => {
  motionPaused = !motionPaused;
  updateMotion();
});
motionPreference.addEventListener('change', (event) => {
  motionPaused = event.matches;
  updateMotion();
});

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy).textContent;
    const status = document.querySelector('.copy-status');
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable');
      await navigator.clipboard.writeText(text);
      button.textContent = '已复制 ✓';
      status.textContent = '启动命令已复制到剪贴板。请先在 .env 中填写模型 API Key。';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(button.dataset.copy));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = '已选中，请手动复制';
      status.textContent = '无法访问剪贴板，命令已选中，请按 Ctrl+C 或 Command+C 复制。';
    }
  });
});
