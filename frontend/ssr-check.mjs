// 临时诊断脚本：SSR 渲染嵌入模式 EvaluationView，定位空白原因。
import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { createServer } from 'vite'

const errors = []
const server = await createServer({
  root: process.cwd(),
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})
const mod = await server.ssrLoadModule('/src/views/EvaluationView.vue')
const EvalView = mod.default

const app = createSSRApp(EvalView, { kbId: 'fd19ec7d-32af-4460-9d24-31536a133187', kbName: 'ZX Bank' })
app.config.warnHandler = (msg) => errors.push('WARN: ' + msg)
app.config.errorHandler = (err) => errors.push('ERR: ' + (err?.stack || err))

try {
  const html = await renderToString(app)
  console.log('--- RENDER OK, length:', html.length)
  console.log(html.slice(0, 3000))
  console.log('...')
  console.log('--- contains markers:')
  for (const m of ['导入评测集', '评估基准', 'ev-dataset-bar', '测试用例', 'ev-empty', '请先选择知识库']) {
    console.log(`  ${m}: ${html.includes(m)}`)
  }
} catch (err) {
  console.log('--- RENDER THREW:', err?.stack || err)
}
console.log('--- collected warnings/errors:')
for (const e of errors) console.log(e)
await server.close()
process.exit(0)
