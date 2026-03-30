# -*- coding: utf-8 -*-
import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'e:/pycode/all-in-rag/gradio_app.py'

with open(SRC, encoding='utf-8') as fh:
    text = fh.read()

# ── 1. Replace kb_upload function ────────────────────────────────────────────
OLD_FUNC = re.compile(
    r'def kb_upload\(file_obj, chunk_size: int, chunk_overlap: int\) -> str:.*?'
    r'return f"\[错误\] \{e\}"',
    re.DOTALL,
)

NEW_FUNC = ('def kb_upload(file_objs, chunk_size: int, chunk_overlap: int) -> str:\n'
    '    """支持一次上传多个文件。file_objs 可能是单个文件对象或列表。"""\n'
    '    if file_objs is None:\n'
    '        return "请先选择至少一个文件。"\n'
    '    if not isinstance(file_objs, list):\n'
    '        file_objs = [file_objs]\n'
    '    if len(file_objs) == 0:\n'
    '        return "请先选择至少一个文件。"\n'
    '\n'
    '    results = []\n'
    '    total_indexed = 0\n'
    '    for file_obj in file_objs:\n'
    '        p = Path(file_obj.name)\n'
    '        try:\n'
    '            with p.open("rb") as f:\n'
    '                with _client(timeout=120) as c:\n'
    '                    r = c.post(\n'
    '                        f"{API_BASE}/kb/upload",\n'
    '                        files={"file": (p.name, f, "application/octet-stream")},\n'
    '                        data={"chunk_size": str(chunk_size), "chunk_overlap": str(chunk_overlap)},\n'
    '                    )\n'
    '                    r.raise_for_status()\n'
    '                    d = r.json()\n'
    '            total_indexed += d.get("indexed", 0)\n'
    '            results.append(f"OK `{p.name}` - indexed {d.get(\'indexed\', 0)} chunks")\n'
    '        except httpx.HTTPStatusError as e:\n'
    '            results.append(f"ERR `{p.name}` - [HTTP {e.response.status_code}] {e.response.text}")\n'
    '        except Exception as e:\n'
    '            results.append(f"ERR `{p.name}` - {e}")\n'
    '\n'
    '    summary = f"共处理 {len(file_objs)} 个文件，合计索引 {total_indexed} 个块。\\n\\n"\n'
    '    return summary + "\\n".join(results)')

m = OLD_FUNC.search(text)
if not m:
    print('ERROR: kb_upload function not found')
    sys.exit(1)
text = OLD_FUNC.sub(NEW_FUNC, text, count=1)
print('OK: kb_upload replaced')

# ── 2. Replace upload tab UI + insert kb-info tab ────────────────────────────
OLD_TAB = re.compile(
    r'        with gr\.Tab\("知识库上传"\):.*?'
    r'up_btn\.click\(kb_upload, \[up_file, up_cs, up_co\], up_result\)',
    re.DOTALL,
)

NEW_TAB = ('        with gr.Tab("知识库上传"):\n'
    '            gr.Markdown("### 上传文档到知识库\\n支持：`.txt`  `.md`  `.pdf`  `.docx`\\n> 可一次选择多个文件批量上传")\n'
    '            with gr.Row():\n'
    '                up_file = gr.File(\n'
    '                    label="选择文件（支持多选）",\n'
    '                    file_types=[".txt", ".md", ".pdf", ".docx"],\n'
    '                    file_count="multiple",\n'
    '                    scale=3,\n'
    '                )\n'
    '                with gr.Column(scale=2):\n'
    '                    up_cs  = gr.Slider(0, 2000, value=0, step=50, label="Chunk Size (0=默认)")\n'
    '                    up_co  = gr.Slider(0, 500,  value=0, step=10, label="Chunk Overlap (0=默认)")\n'
    '                    up_btn = gr.Button("上传并索引", variant="primary")\n'
    '            up_result = gr.Textbox(label="结果", lines=6, interactive=False)\n'
    '            up_btn.click(kb_upload, [up_file, up_cs, up_co], up_result)\n'
    '\n'
    '        with gr.Tab("知识库详情"):\n'
    '            gr.Markdown("### 知识库文档概览")\n'
    '            info_btn = gr.Button("刷新详情", variant="secondary")\n'
    '            info_out = gr.Markdown()\n'
    '            info_btn.click(kb_info, outputs=info_out)\n'
    '            demo.load(kb_info, outputs=info_out)')

m = OLD_TAB.search(text)
if not m:
    print('ERROR: upload tab UI not found')
    sys.exit(1)
text = OLD_TAB.sub(NEW_TAB, text, count=1)
print('OK: upload tab + kb-info tab replaced')

with open(SRC, 'w', encoding='utf-8') as fh:
    fh.write(text)
print('OK: gradio_app.py written successfully')
