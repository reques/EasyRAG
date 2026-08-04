"""hermes-verify: 知识库拖拽上传验证（非测试套件）。

针对 2026-08-04 拖拽上传改动:
1. 模板: label 拖拽事件绑定 + dragover class + UploadCloud 图标
2. 脚本: 拖拽处理函数存在, ACCEPT_EXTS 与后端 ALLOWED_EXTENSIONS 一致, 类型校验
3. 样式: dragover 态 class 定义
4. dist: 构建产物同步

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_drag_upload.py
前置: frontend 已 npm run build（dist 检查依赖构建产物）
"""
import os
import re
import sys
import glob

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
vue = open(os.path.join(ROOT, "frontend/src/views/KnowledgeView.vue"), encoding="utf-8").read()
tmpl = vue.split("<script")[0]
css = open(os.path.join(ROOT, "frontend/src/style.css"), encoding="utf-8").read()

print("== 1. 模板 ==")
check("@dragover.prevent bound", "@dragover.prevent" in tmpl)
check("@dragleave bound", "@dragleave" in tmpl)
check("@drop.prevent bound", "@drop.prevent" in tmpl)
check("dragover class binding", "file-label--dragover" in tmpl)
check("UploadCloud icon used", "UploadCloud" in tmpl)
check("hint text mentions drag", "拖拽" in tmpl)

print("== 2. 脚本 ==")
check("onDragOver defined", "function onDragOver" in vue)
check("onDragLeave defined", "function onDragLeave" in vue)
check("onDrop defined", "function onDrop" in vue)
check("UploadCloud imported", "UploadCloud" in vue.split("<script")[1])

m = re.search(r"ACCEPT_EXTS = \[([^\]]+)\]", vue)
front_exts = set(re.findall(r"'(\.\w+)'", m.group(1))) if m else set()
kr = open(os.path.join(ROOT, "backend/server/routers/knowledge_router.py"), encoding="utf-8").read()
bm = re.search(r"ALLOWED_EXTENSIONS = \{([^}]+)\}", kr)
back_exts = set(re.findall(r'"(\.\w+)"', bm.group(1))) if bm else set()
check("ACCEPT_EXTS matches backend", front_exts == back_exts,
      f"front={sorted(front_exts)} back={sorted(back_exts)}")

check("drop validates extension", "不支持的文件类型" in vue)
check("drop ignores while uploading", "if (uploading.value) return" in vue)
check("closeUpload resets dragOver", "dragOver.value = false" in vue)

print("== 3. 样式 ==")
check(".file-label--dragover styled", ".file-label--dragover" in css)
check("hint layout styled", ".file-label-hint" in css)

print("== 4. dist ==")
chunks = glob.glob(os.path.join(ROOT, "frontend/dist/assets/KnowledgeView-*.js"))
check("KnowledgeView chunk exists", len(chunks) == 1, os.path.basename(chunks[0]) if chunks else "none")
if chunks:
    check("dist has drag hint", "拖拽" in open(chunks[0], encoding="utf-8", errors="replace").read())

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed (ad-hoc) ==")
sys.exit(0 if passed == len(results) else 1)
