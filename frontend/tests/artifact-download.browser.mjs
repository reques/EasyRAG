// Run with: node tests/artifact-download.browser.mjs
import assert from 'node:assert/strict'
import { createServer } from 'vite'
import puppeteer from 'puppeteer-core'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const server = await createServer({ server: { host: '127.0.0.1', port: 5187, strictPort: true } })
let browser
try {
  await server.listen()
  browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  })
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.setViewport({ width: 1440, height: 1000 })
  await page.goto('http://127.0.0.1:5187/tests/session-status.html')
  await page.waitForSelector('textarea')
  await page.waitForFunction(() => document.body.textContent.includes('材料已整理。'))
  await page.type('textarea', 'Generate a downloadable report')
  await page.click('.btn-send')
  await page.waitForSelector('.btn-stop')
  await page.click('[data-action="file"]')
  await page.waitForSelector('.download-artifact button', { visible: true })
  await page.waitForFunction(() => !document.querySelector('.task-panel-drawer-enter-active'))
  assert.match(await page.$eval('.download-artifact', el => el.textContent), /分析报告.md/)
  await page.click('.artifact-download-button')
  await page.waitForFunction(() => document.body.dataset.downloaded === '分析报告.md')
  await page.click('[data-action="done"]')
  await page.click('[data-action="history"]')
  await page.waitForSelector('.download-artifact button', { visible: true })
  assert.equal(await page.$$eval('.download-artifact', items => items.length), 1)
  await page.click('.download-artifact button')
  await page.waitForSelector('.artifact-preview iframe')
  assert.equal(await page.$eval('.artifact-preview iframe', el => el.getAttribute('sandbox')), '')
  await page.click('[aria-label="关闭预览"]')
  await page.waitForSelector('.artifact-preview', { hidden: true })
  await page.waitForFunction(() => !document.querySelector('.task-panel-drawer-enter-active'))
  await page.screenshot({ path: join(tmpdir(), 'easyrag-artifact-download.png') })

  await page.goto('http://127.0.0.1:5187/tests/artifact-preview.html')
  const external = []
  page.on('request', req => { if (req.url().includes('preview-unsafe.invalid')) external.push(req.url()) })
  await page.click('[data-format="docx"]')
  await page.waitForSelector('.artifact-preview iframe')
  const frame = await (await page.$('.artifact-preview iframe')).contentFrame()
  await frame.waitForSelector('h1')
  assert.match(await frame.$eval('body', el => el.textContent), /项目分析报告/)
  assert.equal(await frame.$$eval('[onerror], script, a[href^="javascript:"]', els => els.length), 0)
  assert.deepEqual(external, [])
  await page.screenshot({ path: join(tmpdir(), 'easyrag-word-preview.png') })
  await page.click('[aria-label="关闭预览"]')
  await page.click('[data-format="xlsx"]')
  await page.waitForSelector('.sheet-scroll table')
  assert.match(await page.$eval('.sheet-scroll', el => el.textContent), /产品乙0/)
  await page.click('.sheet-tabs button:nth-child(2)')
  assert.match(await page.$eval('.sheet-scroll', el => el.textContent), /100false/)
  await page.screenshot({ path: join(tmpdir(), 'easyrag-excel-preview.png') })
  await page.keyboard.press('Escape')
  await page.waitForSelector('.artifact-preview', { hidden: true })
  await page.click('[data-error]')
  await page.waitForSelector('[role="alert"]')
  await page.click('[aria-label="关闭预览"]')
  await page.setViewport({ width: 390, height: 844 })
  await page.click('[data-format="xlsx"]')
  await page.waitForSelector('.sheet-scroll table')
  assert.equal(await page.$eval('.artifact-preview', el => Math.round(el.getBoundingClientRect().width)), 390)
  assert.deepEqual(errors, [])
  console.log('PASS: streaming/download/history; Word preview isolation; Excel tabs, errors, keyboard and mobile')
} catch (error) {
  const pages = await browser?.pages()
  await pages?.at(-1)?.screenshot({ path: join(tmpdir(), 'easyrag-artifact-error.png') })
  throw error
} finally {
  await browser?.close()
  await server.close()
}
