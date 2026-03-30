import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'e:/pycode/all-in-rag/gradio_app.py'

with open(SRC, encoding='utf-8') as fh:
    text = fh.read()

# Find the truncation point and replace from there
CUT = 'chat_in.submit(agent_chat, [chat_in,'
idx = text.rfind(CUT)
assert idx != -1, 'cut point not found'

RESTORED = '''chat_in.submit(agent_chat, [chat_in, chatbot, chat_sid], [chatbot, chat_in])
            chat_clear.click(lambda: ([], ""), outputs=[chatbot, chat_in])

        # -- KB upload (multi-file) ------------------------------------------
        with gr.Tab("\u77e5\u8bc6\u5e93\u4e0a\u4f20"):
            gr.Markdown(
                "### \u4e0a\u4f20\u6587\u6863\u5230\u77e5\u8bc6\u5e93\n"
                "\u652f\u6301\uff1a`.txt`  `.md`  `.pdf`  `.docx`\n"
                "> \u53ef\u4e00\u6b21\u9009\u62e9\u591a\u4e2a\u6587\u4ef6\u6279\u91cf\u4e0a\u4f20"
            )
            with gr.Row():
                up_file = gr.File(
                    label="\u9009\u62e9\u6587\u4ef6\uff08\u652f\u6301\u591a\u9009\uff09",
                    file_types=[".txt", ".md", ".pdf", ".docx"],
                    file_count="multiple",
                    scale=3,
                )
                with gr.Column(scale=2):
                    up_cs  = gr.Slider(0, 2000, value=0, step=50, label="Chunk Size (0=\u9ed8\u8ba4)")
                    up_co  = gr.Slider(0, 500,  value=0, step=10, label="Chunk Overlap (0=\u9ed8\u8ba4)")
                    up_btn = gr.Button("\u4e0a\u4f20\u5e76\u7d22\u5f15", variant="primary")
            up_result = gr.Textbox(label="\u7ed3\u679c", lines=6, interactive=False)
            up_btn.click(kb_upload, [up_file, up_cs, up_co], up_result)

        # -- KB info ----------------------------------------------------------
        with gr.Tab("\u77e5\u8bc6\u5e93\u8be6\u60c5"):
            gr.Markdown("### \u77e5\u8bc6\u5e93\u6587\u6863\u6982\u89c8")
            info_btn = gr.Button("\u5237\u65b0\u8be6\u60c5", variant="secondary")
            info_out = gr.Markdown()
            info_btn.click(kb_info, outputs=info_out)
            demo.load(kb_info, outputs=info_out)

        # -- semantic search --------------------------------------------------
        with gr.Tab("\u8bed\u4e49\u68c0\u7d22"):
            gr.Markdown("### \u8bed\u4e49\u68c0\u7d22\uff08\u4e0d\u8c03\u7528 LLM\uff09")
            with gr.Row():
                sr_q   = gr.Textbox(label="\u67e5\u8be2\u5185\u5bb9", placeholder="\u8f93\u5165\u5173\u952e\u8bcd\u2026", scale=5)
                sr_k   = gr.Slider(1, 20, value=4, step=1, label="Top-K", scale=1)
                sr_btn = gr.Button("\u68c0\u7d22", variant="primary", scale=1)
            sr_out = gr.Markdown()
            sr_btn.click(kb_search, [sr_q, sr_k], sr_out)
            sr_q.submit(kb_search, [sr_q, sr_k], sr_out)

        # -- RAG QA -----------------------------------------------------------
        with gr.Tab("RAG \u95ee\u7b54"):
            gr.Markdown("### RAG \u95ee\u7b54\uff08\u68c0\u7d22 + LLM \u751f\u6210\uff09")
            with gr.Row():
                ask_q   = gr.Textbox(label="\u95ee\u9898", placeholder="\u8f93\u5165\u95ee\u9898\u2026", scale=4)
                ask_k   = gr.Slider(1, 20, value=4, step=1, label="Top-K", scale=1)
                ask_sid = gr.Textbox(value="gradio", label="Session ID", scale=2)
                ask_btn = gr.Button("\u63d0\u95ee", variant="primary", scale=1)
            ask_out = gr.Markdown()
            ask_btn.click(kb_ask, [ask_q, ask_k, ask_sid], ask_out)
            ask_q.submit(kb_ask, [ask_q, ask_k, ask_sid], ask_out)

        # -- system health ----------------------------------------------------
        with gr.Tab("\u7cfb\u7edf\u72b6\u6001"):
            gr.Markdown("### \u7cfb\u7edf\u5065\u5eb7\u68c0\u67e5")
            h_btn = gr.Button("\u5237\u65b0\u72b6\u6001", variant="secondary")
            h_out = gr.Markdown()
            h_btn.click(system_health, outputs=h_out)
            demo.load(system_health, outputs=h_out)

        # -- danger zone ------------------------------------------------------
        with gr.Tab("\u5371\u9669\u64cd\u4f5c"):
            gr.Markdown("### \u6e05\u7a7a\u77e5\u8bc6\u5e93\n> **\u8b66\u544a**: \u6c38\u4e45\u5220\u9664\u6240\u6709\u5df2\u7d22\u5f15\u6587\u6863\uff0c\u4e0d\u53ef\u6062\u590d\uff01")
            with gr.Row():
                cl_btn = gr.Button("\u6e05\u7a7a\u5168\u90e8\u6587\u6863", variant="stop")
                cl_out = gr.Textbox(label="\u7ed3\u679c", interactive=False)
            cl_btn.click(kb_clear, outputs=cl_out)


if __name__ == "__main__":
    print("\nGradio \u8c03\u8bd5\u53f0\u542f\u52a8\u4e2d\u2026")
    print(f"  \u524d\u7aef: http://127.0.0.1:{GRADIO_PORT}")
    print(f"  \u540e\u7aef: {API_BASE}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        inbrowser=False,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
'''

text = text[:idx] + RESTORED

with open(SRC, 'w', encoding='utf-8') as fh:
    fh.write(text)
print('OK: file completed')

# verify syntax
import ast
ast.parse(open(SRC, encoding='utf-8').read())
print('OK: syntax valid')
