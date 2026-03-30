"""Gradio frontend for All-in-RAG (port 7860 -> FastAPI port 8000).

Usage:
    python gradio_app.py
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List
import httpx
import gradio as gr

# -- proxy fix: must run at import time, before any httpx request ------------
for _pv in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "ALL_PROXY", "all_proxy"):
    os.environ.pop(_pv, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,::1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# ---------------------------------------------------------------------------

API_BASE    = os.getenv("API_BASE",    "http://127.0.0.1:8000/api/v1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

_TRANSPORT = httpx.HTTPTransport(retries=1)


def _client(timeout: float = 30) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=False, transport=_TRANSPORT)


def _post(path: str, **kwargs) -> dict:
    with _client(timeout=120) as c:
        r = c.post(f"{API_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json()


def _get(path: str) -> dict:
    with _client(timeout=30) as c:
        r = c.get(f"{API_BASE}{path}")
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Agent chat
# ---------------------------------------------------------------------------

def agent_chat(msg: str, history: List[dict], sid: str):
    if not msg.strip():
        return history, ""
    try:
        d = _post("/chat", json={"query": msg, "session_id": sid or "gradio"})
        answer = d.get("final_answer", "(no answer)")
        detail = (
            f"**\u610f\u56fe**: {d.get('intent','-')} ({d.get('intent_confidence',0):.0%})  |  "
            f"**\u68c0\u7d22**: {'\u662f' if d.get('retrieval_triggered') else '\u5426'}  |  "
            f"**\u5de5\u5177**: {d.get('tool_name') or '-'}  |  "
            f"**\u8017\u65f6**: {d.get('elapsed_seconds',0)}s"
        )
        reply = f"{answer}\n\n---\n{detail}"
    except httpx.HTTPStatusError as e:
        reply = f"[HTTP {e.response.status_code}] {e.response.text}"
    except Exception as e:
        reply = f"[\u9519\u8bef] {e}"
    history = history + [
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


# ---------------------------------------------------------------------------
# Knowledge-base helpers
# ---------------------------------------------------------------------------

def kb_upload(file_objs, chunk_size: int, chunk_overlap: int) -> str:
    """\u652f\u6301\u4e00\u6b21\u4e0a\u4f20\u591a\u4e2a\u6587\u4ef6\u3002"""
    if file_objs is None:
        return "\u8bf7\u5148\u9009\u62e9\u81f3\u5c11\u4e00\u4e2a\u6587\u4ef6\u3002"
    if not isinstance(file_objs, list):
        file_objs = [file_objs]
    if len(file_objs) == 0:
        return "\u8bf7\u5148\u9009\u62e9\u81f3\u5c11\u4e00\u4e2a\u6587\u4ef6\u3002"

    results = []
    total_indexed = 0
    for file_obj in file_objs:
        p = Path(file_obj.name)
        try:
            with p.open("rb") as f:
                with _client(timeout=120) as c:
                    r = c.post(
                        f"{API_BASE}/kb/upload",
                        files={"file": (p.name, f, "application/octet-stream")},
                        data={
                            "chunk_size":    str(chunk_size),
                            "chunk_overlap": str(chunk_overlap),
                        },
                    )
                    r.raise_for_status()
                    d = r.json()
            total_indexed += d.get("indexed", 0)
            results.append(f"OK `{p.name}` -- \u6210\u529f\u7d22\u5f15 {d.get('indexed', 0)} \u4e2a\u5757")
        except httpx.HTTPStatusError as e:
            results.append(f"ERR `{p.name}` -- [HTTP {e.response.status_code}] {e.response.text}")
        except Exception as e:
            results.append(f"ERR `{p.name}` -- {e}")

    summary = (
        f"\u5171\u5904\u7406 {len(file_objs)} \u4e2a\u6587\u4ef6\uff0c"
        f"\u5408\u8ba1\u7d22\u5f15 {total_indexed} \u4e2a\u5757\u3002\n\n"
    )
    return summary + "\n".join(results)


def kb_info() -> str:
    """\u83b7\u53d6\u77e5\u8bc6\u5e93\u8be6\u7ec6\u4fe1\u606f\u3002"""
    try:
        d = _get("/kb/info")
        files        = d.get("files", [])
        total_files  = d.get("total_files",  0)
        total_chunks = d.get("total_chunks", 0)
        total_chars  = d.get("total_chars",  0)
        lines = [
            "## \u77e5\u8bc6\u5e93\u6982\u89c8",
            f"- **\u6587\u4ef6\u6570**: {total_files}",
            f"- **\u603b\u5757\u6570**: {total_chunks}",
            f"- **\u603b\u5b57\u7b26\u6570**: {total_chars}",
            "",
            "## \u6587\u4ef6\u8be6\u60c5",
        ]
        if not files:
            lines.append("_\u77e5\u8bc6\u5e93\u4e3a\u7a7a\uff0c\u5c1a\u672a\u4e0a\u4f20\u4efb\u4f55\u6587\u6863\u3002_")
        else:
            lines.append("| # | \u6587\u4ef6\u540d | \u5757\u6570 | \u5b57\u7b26\u6570 |")
            lines.append("|---|--------|------|--------|")
            for i, fi in enumerate(files, 1):
                src    = fi.get("source", "-")
                chunks = fi.get("chunk_count", 0)
                chars  = fi.get("char_count",  0)
                lines.append(f"| {i} | `{src}` | {chunks} | {chars} |")
        return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        return f"[HTTP {e.response.status_code}] {e.response.text}"
    except Exception as e:
        return f"[\u9519\u8bef] {e}"


def kb_search(query: str, top_k: int) -> str:
    if not query.strip():
        return "\u8bf7\u8f93\u5165\u67e5\u8be2\u5185\u5bb9\u3002"
    try:
        d = _post("/kb/search", json={"query": query, "top_k": int(top_k)})
        results = d.get("results", [])
        if not results:
            return "\u672a\u627e\u5230\u76f8\u5173\u6587\u6863\u3002"
        lines = [f"\u5171\u627e\u5230 **{d['total']}** \u6761\u7ed3\u679c:\n"]
        for i, r in enumerate(results, 1):
            src     = r.get("metadata", {}).get("source", "\u672a\u77e5")
            content = r["content"][:300].replace("\n", " ")
            lines.append(
                f"**[{i}]** \u76f8\u5173\u5ea6: `{r.get('score',0):.4f}` "
                f"| \u6765\u6e90: `{src}`\n\n{content}\n"
            )
        return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        return f"[HTTP {e.response.status_code}] {e.response.text}"
    except Exception as e:
        return f"[\u9519\u8bef] {e}"


def kb_ask(query: str, top_k: int, sid: str) -> str:
    if not query.strip():
        return "\u8bf7\u8f93\u5165\u95ee\u9898\u3002"
    try:
        d = _post("/kb/ask", json={"query": query, "top_k": int(top_k), "session_id": sid or "gradio"})
        src_str = ", ".join(d.get("sources", [])) or "\u65e0"
        return (
            f"{d.get('answer','(\u65e0\u56de\u7b54)')}\n\n---\n"
            f"**\u68c0\u7d22\u6587\u6863\u6570**: {d.get('retrieved_docs_count',0)}  |  "
            f"**\u6765\u6e90**: {src_str}  |  **\u8017\u65f6**: {d.get('elapsed_seconds',0)}s"
        )
    except httpx.HTTPStatusError as e:
        return f"[HTTP {e.response.status_code}] {e.response.text}"
    except Exception as e:
        return f"[\u9519\u8bef] {e}"


def system_health() -> str:
    lines = []
    try:
        d = _get("/health")
        lines += [
            f"**\u4e3b\u670d\u52a1**: {d.get('status','?').upper()}",
            f"- \u7248\u672c: {d.get('version','-')}",
            f"- LLM: {d.get('llm_model','-')}",
            f"- \u5411\u91cf\u5e93: {d.get('vector_store_type','-')}",
            f"- Embedding: {d.get('embedding_type','-')}",
            "",
        ]
    except Exception as e:
        lines.append(f"**\u4e3b\u670d\u52a1**: \u5f02\u5e38 \u2014 {e}\n")
    try:
        d = _get("/kb/health")
        lines += [
            f"**\u77e5\u8bc6\u5e93**: {d.get('status','?').upper()}",
            f"- \u5411\u91cf\u5e93: {d.get('vector_store_type','-')}",
            f"- Embedding \u7c7b\u578b: {d.get('embedding_type','-')}",
            f"- Embedding \u6a21\u578b: {d.get('embedding_model','-')}",
        ]
    except Exception as e:
        lines.append(f"**\u77e5\u8bc6\u5e93**: \u5f02\u5e38 \u2014 {e}")
    lines.append(f"\n_\u540e\u7aef: {API_BASE}_")
    return "\n".join(lines)


def kb_clear() -> str:
    try:
        with _client(timeout=30) as c:
            r = c.delete(f"{API_BASE}/kb/collection")
            r.raise_for_status()
            return f"\u5df2\u6e05\u7a7a\u77e5\u8bc6\u5e93\u3002\n{r.json().get('message','')}"
    except httpx.HTTPStatusError as e:
        return f"[HTTP {e.response.status_code}] {e.response.text}"
    except Exception as e:
        return f"[\u9519\u8bef] {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container{font-family:'Noto Sans SC','PingFang SC',sans-serif;}
#title{text-align:center;margin-bottom:0.1em;}
#subtitle{text-align:center;color:#888;font-size:0.9em;margin-top:0;}
"""

with gr.Blocks(title="All-in-RAG \u8c03\u8bd5\u53f0", css=CSS) as demo:
    gr.Markdown("# All-in-RAG \u8c03\u8bd5\u53f0", elem_id="title")
    gr.Markdown(
        f"Gradio \u524d\u7aef \u00b7 \u7aef\u53e3 **{GRADIO_PORT}**  \u2192  FastAPI \u540e\u7aef \u00b7 \u7aef\u53e3 **8000**",
        elem_id="subtitle",
    )

    with gr.Tabs():

        # ── Agent \u5bf9\u8bdd ──────────────────────────────────────────────────
        with gr.Tab("Agent \u5bf9\u8bdd"):
            gr.Markdown("### LangGraph \u5168\u6d41\u7a0b Agent \u5bf9\u8bdd")
            chatbot = gr.Chatbot(height=460, label="\u5bf9\u8bdd\u5386\u53f2")
            with gr.Row():
                chat_in  = gr.Textbox(
                    placeholder="\u8f93\u5165\u95ee\u9898\uff0c\u56de\u8f66\u53d1\u9001\u2026",
                    show_label=False,
                    scale=8,
                )
                chat_btn = gr.Button("\u53d1\u9001", variant="primary", scale=1)
            with gr.Row():
                chat_sid   = gr.Textbox(value="gradio_session", label="Session ID", scale=3)
                chat_clear = gr.Button("\u6e05\u7a7a\u5bf9\u8bdd", scale=1)
            chat_btn.click(agent_chat, [chat_in, chatbot, chat_sid], [chatbot, chat_in])
            chat_in.submit(agent_chat, [chat_in, chatbot, chat_sid], [chatbot, chat_in])
            chat_clear.click(lambda: ([], ""), outputs=[chatbot, chat_in])

                # ── 知识库上传（多文件）
        with gr.Tab("知识库上传"):
            gr.Markdown(
                "### 上传文档到知识库\n"
                "支持格式：`.txt`  `.md`  `.pdf`  `.docx`\n"
                "> 可一次选择多个文件批量上传"
            )
            with gr.Row():
                up_file = gr.File(
                    label="选择文件（支持多选）",
                    file_types=[".txt", ".md", ".pdf", ".docx"],
                    file_count="multiple",
                    scale=3,
                )
                with gr.Column(scale=2):
                    up_cs  = gr.Slider(0, 2000, value=0, step=50,  label="Chunk Size（0=使用默认）")
                    up_co  = gr.Slider(0, 500,  value=0, step=10,  label="Chunk Overlap（0=使用默认）")
                    up_btn = gr.Button("上传并索引", variant="primary")
            up_result = gr.Textbox(label="上传结果", lines=8, interactive=False)
            up_btn.click(kb_upload, [up_file, up_cs, up_co], up_result)

        # ── 知识库详情
        with gr.Tab("知识库详情"):
            gr.Markdown("### 知识库文档概览")
            with gr.Row():
                info_btn = gr.Button("刷新详情", variant="secondary")
            info_out = gr.Markdown()
            info_btn.click(kb_info, outputs=info_out)
            demo.load(kb_info, outputs=info_out)

        # ── 语义检索
        with gr.Tab("语义检索"):
            gr.Markdown("### 语义检索（不调用 LLM）")
            with gr.Row():
                sr_q   = gr.Textbox(label="查询内容", placeholder="输入关键词…", scale=5)
                sr_k   = gr.Slider(1, 20, value=4, step=1, label="Top-K", scale=1)
                sr_btn = gr.Button("检索", variant="primary", scale=1)
            sr_out = gr.Markdown()
            sr_btn.click(kb_search, [sr_q, sr_k], sr_out)
            sr_q.submit(kb_search, [sr_q, sr_k], sr_out)

        # ── RAG 问答
        with gr.Tab("RAG 问答"):
            gr.Markdown("### RAG 问答（检索 + LLM 生成）")
            with gr.Row():
                ask_q   = gr.Textbox(label="问题", placeholder="输入问题…", scale=4)
                ask_k   = gr.Slider(1, 20, value=4, step=1, label="Top-K", scale=1)
                ask_sid = gr.Textbox(value="gradio", label="Session ID", scale=2)
                ask_btn = gr.Button("提问", variant="primary", scale=1)
            ask_out = gr.Markdown()
            ask_btn.click(kb_ask, [ask_q, ask_k, ask_sid], ask_out)
            ask_q.submit(kb_ask, [ask_q, ask_k, ask_sid], ask_out)

        # ── 系统状态
        with gr.Tab("系统状态"):
            gr.Markdown("### 系统健康检查")
            h_btn = gr.Button("刷新状态", variant="secondary")
            h_out = gr.Markdown()
            h_btn.click(system_health, outputs=h_out)
            demo.load(system_health, outputs=h_out)

        # ── 危险操作
        with gr.Tab("危险操作"):
            gr.Markdown(
                "### 清空知识库\n"
                "> **警告**: 永久删除所有已索引文档，不可恢复！"
            )
            with gr.Row():
                cl_btn = gr.Button("清空全部文档", variant="stop")
                cl_out = gr.Textbox(label="结果", interactive=False)
            cl_btn.click(kb_clear, outputs=cl_out)


if __name__ == "__main__":
    print("\nGradio 调试台启动中…")
    print(f"  前端: http://127.0.0.1:{GRADIO_PORT}")
    print(f"  后端: {API_BASE}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        inbrowser=False,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
