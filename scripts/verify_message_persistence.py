"""hermes-verify: nodes.py executor 线程 DB 连接池隔离修复验证（非测试套件）。

针对 2026-08-04 消息丢失根因修复:
1. nodes.py 语法 + 无残留「线程内 asyncio.run 复用全局连接池」模式
2. /chat/send 带检索路径: user+assistant 均落库（executor DB 访问后连接池未中毒）
3. 同会话第二轮历史累积（4 条）+ 模型上下文回忆
4. /chat/stream SSE 落库完整
5. stream 后再 send（污染回归）
6. 清理测试数据

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_message_persistence.py
前置: 后端 :8000 运行中, easyrag-postgres 容器运行中, 存在 testuser/test123456 账户
"""
import ast
import json
import subprocess
import sys
import urllib.request
import urllib.error
import http.client

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
BASE = "http://localhost:8000/api/v1"

# ── 1. 静态检查 ─────────────────────────────────────────────────────
print("== 1. nodes.py 静态检查 ==")
src = open(f"{ROOT}/app/graph/nodes.py", encoding="utf-8").read()
try:
    ast.parse(src)
    check("syntax", True)
except SyntaxError as e:
    check("syntax", False, str(e))
    sys.exit(1)

check("has _run_in_thread_isolated helper", "_run_in_thread_isolated" in src)
check("has isolated engine dispose", "engine.dispose()" in src)
# 危险模式精确匹配: asyncio.run(X) 其中 X 不是 _run_with_isolated_engine
# 逐行检查，排除注释行和隔离封装自身
import re
bad_lines = [
    i + 1 for i, line in enumerate(src.splitlines())
    if "asyncio.run(" in line
    and "_run_with_isolated_engine" not in line
    and not line.strip().startswith("#")
]
check("no raw asyncio.run outside isolated wrapper", not bad_lines, f"lines={bad_lines}")

# ── 2. HTTP 行为验证 ────────────────────────────────────────────────
def req(method, path, data=None, token=None):
    headers, body = {}, None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            payload = resp.read()
            return resp.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:200]}

def db(sql):
    out = subprocess.run(["docker", "exec", "easyrag-postgres", "psql", "-U", "easyrag",
                          "-d", "easyrag", "-t", "-c", sql], capture_output=True, text=True)
    return out.stdout.strip()

print("== 2. 消息落库行为 ==")
try:
    s, tok = req("POST", "/auth/login", {"username": "testuser", "password": "test123456"})
    check("login", s == 200)
    token = tok["access_token"]
except Exception as e:
    check("backend reachable", False, str(e))
    sys.exit(1)

# 2a. send 带检索（触发 executor 线程内 DB 查询）
s, resp = req("POST", "/chat/send", {"query": "民法典里讲了什么内容？"}, token=token)
check("send with retrieval: 200", s == 200, str(resp)[:80] if s != 200 else "")
conv_id = resp.get("conversation_id")
n = db(f"SELECT string_agg(role, ',' ORDER BY id) FROM messages WHERE conversation_id='{conv_id}';")
check("user+assistant both persisted", n == "user,assistant", f"roles={n}")

# 2b. 第二轮历史累积
s, resp2 = req("POST", "/chat/send", {"query": "我刚才问的是什么？", "conversation_id": conv_id}, token=token)
check("second turn: 200", s == 200)
n = db(f"SELECT count(*) FROM messages WHERE conversation_id='{conv_id}';")
check("history accumulated to 4 msgs", n == "4", f"count={n}")
check("model recalls context", "民法典" in resp2.get("answer", ""), resp2.get("answer", "")[:50])

# 2c. stream SSE 落库
conn = http.client.HTTPConnection("localhost", 8000)
conn.request("POST", "/api/v1/chat/stream", body=json.dumps({"query": "1+1=?"}),
             headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
r = conn.getresponse()
sse = r.read().decode("utf-8", errors="replace")
conn.close()
check("stream: 200", r.status == 200)
events = [json.loads(l[6:]) for l in sse.split("\n") if l.startswith("data: ")]
types = [e["type"] for e in events]
check("stream: has delta + done, no error", "done" in types and "error" not in types,
      f"deltas={types.count('delta')}")
sconv = next((e["conversation_id"] for e in events if e["type"] == "conversation_id"), None)
if sconv:
    n = db(f"SELECT string_agg(role, ',' ORDER BY id) FROM messages WHERE conversation_id='{sconv}';")
    check("stream conv persisted user+assistant", n == "user,assistant", f"roles={n}")

# 2d. 污染回归: stream 后再 send
s, resp3 = req("POST", "/chat/send", {"query": "再说一遍 1+1"}, token=token)
check("post-stream send: 200 (pool not poisoned)", s == 200, str(resp3)[:80] if s != 200 else "")
if s == 200:
    n = db(f"SELECT count(*) FROM messages WHERE conversation_id='{resp3['conversation_id']}';")
    check("post-stream send persisted", n == "2", f"count={n}")

# ── 3. 清理测试数据 ─────────────────────────────────────────────────
print("== 3. 清理 ==")
subprocess.run(["docker", "exec", "easyrag-postgres", "psql", "-U", "easyrag", "-d", "easyrag", "-c",
                "DELETE FROM conversations WHERE created_at > now() - interval '2 hours';"],
               capture_output=True, text=True)
n = db("SELECT count(*) FROM conversations;")
check("test data cleaned", n == "0", f"remaining={n}")

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed ==")
sys.exit(0 if passed == len(results) else 1)
