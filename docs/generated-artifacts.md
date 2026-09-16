# 可下载文件产物

对话中可以要求“生成 Word 报告”“导出包含明细和汇总两张表的 Excel”或“生成 PDF”。
Agent 调用 `create_file(filename, content)`，保存成功后，右侧“状态 → 产物”显示文件名、大小、预览和下载按钮。普通 Agent 和深度研究主 Agent 均可调用，无需选择 Skill。

## 支持范围

- Word（DOCX）、Excel（XLSX）、PDF，以及 Markdown、TXT、CSV、JSON、HTML；单文件最大 5 MiB（生成前输入、生成后文件均检查）。
- `.doc`、`.xls` 文件名自动转换为 `.docx`、`.xlsx`，保存真正的 Office 文件，不生成旧版二进制格式。
- Word/PDF 接收 Markdown，支持标题、正文、列表、表格。PDF 使用 ReportLab，Docker 镜像包含文泉驿中文字体并嵌入 PDF；本地 Windows 尝试微软雅黑/宋体。
- Excel 接收工作表 JSON 或单表 CSV，支持多个工作表、表头样式、冻结首行、筛选和自动列宽。最多 20 张工作表，每表 10000 行、每行 100 列、总计 100000 单元格。字符串按文本保存，不执行公式；需要计算时传入计算后的数值。
- JSON 会先检查语法；文件名不能包含路径或控制字符。
- 不提供任意代码执行或公开分享链接。

Excel 工具输入示例（`content` 为以下 JSON 的字符串）：

```json
{"sheets":[{"name":"明细","rows":[["产品","金额"],["甲",100],["乙",200]]},{"name":"汇总","rows":[["合计"],[300]]}]}
```

## 预览

| 格式 | 展示方式 |
| --- | --- |
| DOCX | 标题、正文、强调和表格内容；不是 Word 的精确分页排版 |
| XLSX、CSV | 表格；Excel 可切换工作表；每表最多展示 200 行、50 列，超过时提示下载完整数据 |
| PDF | 浏览器 PDF 页面查看器 |
| Markdown、HTML | 隔离的文档预览，禁用脚本和外部资源 |
| TXT、JSON | 纯文本，JSON 格式化展示 |

预览弹窗包含下载入口，支持错误重试、关闭和移动端布局。Word/HTML 内容不会插入应用 DOM，使用无脚本的 sandbox iframe 和 CSP。Office 预览检查解压后的体积，避免无界解压。

## 存储与访问

复用 `MINIO_*` 配置。文件保存在已有 bucket 的 `artifacts/<user UUID>/<conversation UUID>/<artifact UUID>` 下，原文件名存于对象元数据。元数据和正文通过一次对象写入保存，不需要数据库迁移。

- `GET /api/v1/artifacts/{conversation_id}`：会话的文件列表。
- `GET /api/v1/artifacts/{conversation_id}/{artifact_id}`：附件下载。
- `GET /api/v1/artifacts/{conversation_id}/{artifact_id}/preview`：预览数据；PDF 返回可内联显示的二进制，其他格式返回 JSON。

所有接口均要求登录，并检查会话归属。前端通过带 JWT 的请求取得数据，不暴露 MinIO 凭据或依赖公开 bucket。会话切换时加载持久化文件列表，文件创建事件即时更新 UI；中断回答不影响已成功保存的文件。

删除会话后，鉴权接口不再允许访问其文件；存储对象目前保留，部署方可按保留需求配置 MinIO 生命周期清理。

## 验证与部署

后端：`python -m pytest tests/test_artifacts.py tests/test_artifact_formats.py tests/test_execution_trace.py`

前端：`npm run build`。交互回归：在 frontend 下运行 `node tests/artifact-download.browser.mjs`（需要可用的 `puppeteer-core` 和 Chrome；可通过 `CHROME_PATH` 指定浏览器路径）。测试使用隔离的模拟接口，不调用真实模型。

Docker 更新：

```sh
docker compose up -d --build --no-deps backend frontend
```
