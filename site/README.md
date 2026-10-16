# EasyRAG 项目主页

中文静态主页，独立于 `frontend/` 中的实际业务应用。使用原生 HTML、CSS 和 JavaScript，无需 npm 安装或构建，不调用模型、不上传数据。页面中的问答场景和 GIF 均为能力示意。

## 本地预览

在仓库根目录运行：

```bash
python -m http.server 4173 --directory site --bind 127.0.0.1
```

访问 `http://127.0.0.1:4173`。也可以直接打开 `index.html`；剪贴板功能在 HTTPS 或 localhost 环境下更可靠，失败时页面会选中命令供手动复制。

## 发布到 GitHub Pages

预期地址：**https://reques.github.io/EasyRAG/**。

1. 将主页、素材和 `.github/workflows/pages.yml` 提交并推送到 GitHub。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。
3. `main` 或 `home` 分支中涉及主页的提交会自动运行 **Deploy project homepage** 工作流。默认分支包含该工作流后，也可在 **Actions → Deploy project homepage → Run workflow** 手动运行。
4. 等待 `build` 与 `deploy` 成功后，访问工作流 `github-pages` 环境返回的实际地址。

当前主页使用 `home` 分支发布，推送主页变更即可触发部署，无需将其他分支的业务代码合并到 `main`。工作流也支持 `main`，未来可合并主页文件后从默认分支发布。若 Pages 环境有分支保护限制，需允许实际发布分支。

工作流只上传 `site/`，不会发布仓库中的后端配置、业务数据或 `.env`。将来更换 GitHub 用户名或仓库名时，更新 HTML 和 README 中的仓库/主页地址及 Open Graph URL。

## 内容与交互

- 图标与 3D 猫头鹰主视觉；响应式深色页面。
- 混合检索、知识图谱、Agent / Skills / MCP 能力介绍。
- 三种可切换示意场景：问答、关联探索、深度研究。支持方向键、Home / End 操作。
- 知识流转 GIF、架构概览、Docker 启动命令与复制反馈。
- 跳转链接、焦点样式、图片替代文本、`prefers-reduced-motion` 与暂停动画按钮。
- JavaScript 不可用时，导航、首个场景、部署指引仍可阅读；系统要求减少动态效果时使用静态流程图。

## 品牌素材

| 文件 | 用途 |
|------|------|
| `assets/owl-icon.png` | 基于用户原图改进的高分辨率图标 |
| `assets/owl-icon.webp` | 导航、README 中使用的小体积图标 |
| `assets/owl-mascot.png` | 透明背景 3D 吉祥物原始输出 |
| `assets/owl-mascot.webp` | 首页主视觉的优化版本 |
| `assets/favicon.png` / `assets/apple-touch-icon.png` | 浏览器及移动设备图标 |
| `assets/knowledge-flow.gif` | 64 帧、6.4 秒循环的概念流程动画 |
| `assets/overview-poster.png` | 完整流程静态图与分享预览图 |

图标与吉祥物由内置 **ImageGen** 基于用户提供的猫头鹰/文档原图生成。原有 `assets/` 中的备选图标保留。本次生成未使用 CLI 或用户 API Key。WebP 和小尺寸图标为生成结果的格式/尺寸转换；GIF 将生成的吉祥物与程序绘制的流程图组合并添加悬浮、进度与高亮效果。

实际 Vue 应用的标签页图标也已替换：`frontend/public/easyrag-icon.png` 与主页 favicon 使用同一份图标。更换品牌图标时请同步这两处。

### 生成提示词

**图标：** Preserve the supplied symmetric white owl-with-document silhouette, pointed ears, almond eye cutouts and beak. Refine into a premium EasyRAG icon with subtle porcelain depth, cyan rim light, midnight indigo rounded-square background, a document inset with three cyan retrieval lines and connected knowledge nodes. Keep generous padding and small-size legibility. No text, letters, scenery or extra limbs.

**吉祥物（用户选择活泼 3D 风格）：** A lively, friendly 3D white porcelain owl derived from the supplied EasyRAG owl/document identity, with pointed ear tufts, large expressive indigo eyes and cyan glints, a tiny indigo beak, short rounded wings and a playful slight tilt. An indigo folded-corner document panel on its belly with cyan lines and connected nodes; holds a glowing document, with small floating page tiles. Matte sculpted material, soft lavender shadows, cyan highlights, studio lighting, whole-body centered, true transparent background, no text or wordmark.

重新生成动图需要 Python 与 Pillow：

```bash
python -m pip install Pillow
python scripts/generate_homepage_animation.py
```

脚本优先使用 Windows 微软雅黑，也支持 Linux Noto CJK 字体路径。重新生成后请目视检查中文字体，再提交 GIF 与静态图。

## 验证

在仓库根目录执行：

```bash
python scripts/check_homepage.py
node --check site/main.js
```

检查 HTML 锚点、页面/脚本/README 本地素材以及脚本语法。视觉检查建议覆盖 1440px 桌面、390px 手机、320px 窄屏、无 JavaScript 和减少动态效果模式。
