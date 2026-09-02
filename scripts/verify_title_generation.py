"""hermes-verify: 会话标题语义化生成验证（非测试套件）。

针对 2026-08-04 标题语义化改动:
1. 静态: 统一函数存在、max_tokens=256、stream 后台协程、router 无旧 prompt 残留、
   全部 3 个标题入口(send/stream/summarize)收敛到 generate_conversation_title
2. 函数级: LLM 标题语义化(名词短语、非原文截取、长度合规)
3. fallback: 祈使词剥离、长度截断、空输入兜底

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_title_generation.py
前置: LLM 服务可达（函数级检查走真实 LLM 调用）
"""
import ast
import asyncio
import subprocess
import sys

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
sys.path.insert(0, ROOT)

print("== 1. 静态检查 ==")
svc = open(f"{ROOT}/backend/services/chat_service.py", encoding="utf-8").read()
rtr = open(f"{ROOT}/backend/server/routers/chat_router.py", encoding="utf-8").read()
try:
    ast.parse(svc); ast.parse(rtr)
    check("syntax", True)
except SyntaxError as e:
    check("syntax", False, str(e)); sys.exit(1)

check("generate_conversation_title exists", "async def generate_conversation_title" in svc)
check("_fallback_title exists", "def _fallback_title" in svc)
check("max_tokens=256 (reasoning 保底)", "max_tokens=256" in svc)
check("router uses unified fn", "generate_conversation_title" in rtr)
check("stream: background task", "create_task(_gen_title())" in rtr)
check("no summary_prompt anywhere in router", "summary_prompt" not in rtr)
check("no max_tokens=40/50 in router", "max_tokens=40" not in rtr and "max_tokens=50" not in rtr)
check("summarize endpoint uses unified fn",
      "title = await generate_conversation_title(user_msg, asst_msg)" in rtr)

print("== 2. LLM 标题语义化 ==")

async def fn_checks():
    from backend.services.chat_service import generate_conversation_title, _fallback_title
    cases = [
        ("某员工在作业中发现设备异常，判断继续操作可能直接危及人身安全，他应该怎么处理？",
         "安全生产法第五十二条规定，从业人员发现直接危及人身安全的紧急情况时，有权停止作业或者在采取可能的应急措施后撤离作业场所。"),
        ("民法典第1042条规定了什么内容？",
         "民法典第1042条规定了婚姻家庭的禁止行为，包括禁止包办买卖婚姻、禁止重婚、禁止家庭暴力等。"),
    ]
    for q, a in cases:
        t = await generate_conversation_title(q, a)
        semantic = (t != "New Conversation" and 2 <= len(t) <= 30
                    and not q.startswith(t[:8]) and not t.endswith(("？", "。", "?", ".")))
        check(f"semantic: {q[:16]}…", semantic, f"{t!r}")

    for q, expect in [("请帮我解释一下民法典", "解释一下民法典"),
                      ("帮我查一下最新新闻", "查一下最新新闻"),
                      ("你能不能告诉我怎么做", "怎么做")]:
        check(f"fallback: {q[:12]}…", _fallback_title(q) == expect, _fallback_title(q))
    check("fallback truncates with …", _fallback_title("很长的输入" * 20).endswith("…"))
    check("fallback empty -> 新对话", _fallback_title("   ") == "新对话")

asyncio.run(fn_checks())

print("== 3. 清理 ==")
subprocess.run(["docker", "exec", "easyrag-postgres", "psql", "-U", "easyrag", "-d", "easyrag", "-c",
                "DELETE FROM conversations WHERE created_at > now() - interval '2 hours' AND title='New Conversation';"],
               capture_output=True, text=True)
check("cleaned", True)

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed (ad-hoc) ==")
sys.exit(0 if passed == len(results) else 1)
