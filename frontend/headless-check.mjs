// 临时诊断：无头浏览器打开 RAG 评估 tab，dump 控制台错误 + DOM 实况。
import puppeteer from 'puppeteer-core'

const BASE = 'http://localhost:5173'
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'

const token = process.argv[2]
const kbId = process.argv[3]

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu'],
})
const page = await browser.newPage()
const logs = []
page.on('pageerror', (e) => {
  let inner = ''
  try { inner = String(e.stack || '') } catch {}
  logs.push(`[pageerror] ${e.message}\nSTACK: ${inner.split('\n').slice(0, 10).join(' || ')}`)
})
page.on('console', async (m) => {
  try {
    const parts = await Promise.all(m.args().map((a) => a.evaluate((x) => {
      if (x instanceof Error) return String(x.stack).split('\n').slice(0, 8).join(' || ')
      try { return typeof x === 'object' ? JSON.stringify(x).slice(0, 400) : String(x) } catch { return String(x) }
    })))
    logs.push(`[console.${m.type()}] ${parts.join(' ~ ')}`)
  } catch {
    logs.push(`[console.${m.type()}] ${m.text()}`)
  }
})
page.on('requestfailed', (r) => logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`))

await page.goto(BASE + '/login', { waitUntil: 'networkidle0' })
await page.evaluate((t) => localStorage.setItem('token', t), token)
await page.evaluate((u) => localStorage.setItem('user', JSON.stringify(u)), { id: 'probe', username: 'probe' })

await page.goto(`${BASE}/knowledge?kb=${kbId}&tab=evaluation`, { waitUntil: 'networkidle0' })
await new Promise((r) => setTimeout(r, 3000))

const info = await page.evaluate(() => {
  const section = document.querySelector('main.kbw-detail-scroll > section')
  const shell = document.querySelector('.ev-shell')
  const cards = [...document.querySelectorAll('.ev-card')]
  const cs = section ? getComputedStyle(section) : null
  const shellCs = shell ? getComputedStyle(shell) : null
  return {
    sectionFound: !!section,
    sectionHTMLLen: section ? section.innerHTML.length : 0,
    sectionChildren: section ? [...section.children].map(c => c.className) : [],
    sectionDisplay: cs && cs.display, sectionHeight: cs && cs.height,
    shellFound: !!shell,
    shellDisplay: shellCs && shellCs.display, shellVis: shellCs && shellCs.visibility,
    shellH: shellCs && shellCs.height, shellOF: shellCs && (shellCs.overflow + '/' + shellCs.overflowY),
    cardCount: cards.length,
    cardSummary: cards.map(c => {
      const s = getComputedStyle(c)
      return { cls: c.className.slice(0,40), h: c.offsetHeight, disp: s.display, vis: s.visibility, txt: (c.innerText||'').slice(0,40).replace(/\n/g,'|') }
    }),
    hasImportBtn: !!([...document.querySelectorAll('button')].find(b => b.innerText.includes('导入评测集'))),
    activeTabLabel: document.querySelector('.kbw-tabs button.active')?.innerText,
  }
})
console.log(JSON.stringify(info, null, 2))
console.log('--- console/page logs ---')
console.log(logs.slice(0, 40).join('\n'))
await browser.close()
