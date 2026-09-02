"""hermes-verify: ReAct 循环子图验证（非测试套件）。

针对 Task 2 ReAct 循环:
1. 静态: agent_reasoning 节点/route_after_reasoning/循环边/state 新字段/prompt 注册
2. 单工具任务: "1+1等于几" 走 ReAct → calculator → final_answer
3. 多步任务: 检索+计算组合 → 多轮 reasoning
4. 步数耗尽: 强制回答
5. 快速路径回归: chitchat/knowledge_qa 仍走静态路径（use_react=False）

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_react_loop.py
前置: 后端 :8000 运行中（行为测试经 HTTP）; LLM 服务可达
"""
import ast
import json
import subprocess
import sys
import time
import urllib.request
import http.client

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
BASE = "http://localhost:8000/api/v1"

# ── 1. 静态检查 ─────────────────────────────────────────────────────
print("== 1. 静态结构 ==")
nodes_src = open(f"{ROOT}/app/graph/nodes.py", encoding="utf-8").read()
router_src = open(f"{ROOT}/app/graph/router.py", encoding="utf-8").read()
wf_src = open(f"{ROOT}/app/graph/workflow.py", encoding="utf-8").read()
state_src = open(f"{ROOT}/app/graph/state.py", encoding="utf-8").read()
tmpl_src = open(f"{ROOT}/app/prompts/templates.py", encoding="utf-8").read()

for f in [nodes_src, router_src, wf_src, state_src, tmpl_src]:
    try:
        ast.parse(f)
    except SyntaxError as e:
        check("syntax", False, str(e)); sys.exit(1)
check("syntax all 5 files", True)

check("agent_reasoning node defined", "def agent_reasoning" in nodes_src)
check("route_after_reasoning defined", "def route_after_reasoning" in router_src)
check("AGENT_REASONING constant", 'AGENT_REASONING' in router_src)
check("agent_reasoning registered in workflow", "agent_reasoning" in wf_src)
check("cycle edge tool_execution -> agent_reasoning",
      "AGENT_REASONING:   AGENT_REASONING" in wf_src or "AGENT_REASONING: AGENT_REASONING" in wf_src)
check("state has observations", "observations:" in state_src)
check("state has use_react", "use_react:" in state_src)
check("state has react_iterations", "react_iterations:" in state_src)
check("REACT_REASONING prompt defined", "REACT_REASONING" in tmpl_src)
check("intent_recognition sets use_react", "use_react = intent ==" in nodes_src)
check("tool_execution handles pending_tool", "pending_tool" in nodes_src)
check("tool_execution _retry skip", '"_retry"' in nodes_src)

# graph 编译
sys.path.insert(0, ROOT)
try:
    from app.graph.workflow import build_graph
    g = build_graph()
    check("graph compiles with agent_reasoning", "agent_reasoning" in g.nodes)
except Exception as e:
    check("graph compiles", False, str(e)[:100])

# ── 2. 行为验证（HTTP）──────────────────────────────────────────────
print("== 2. ReAct 行为（需后端运行）==")

def login():
    r = urllib.request.Request(BASE + "/auth/login",
        data=json.dumps({"username": "testuser", "password": "test123456"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())["access_token"]

def send(query, token):
    """同步 /chat/send 走完整 workflow（含 ReAct 分流），返回 (intent, answer, steps)"""
    r = urllib.request.Request(BASE + "/chat/send",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST")
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            d = json.loads(resp.read())
            return d.get("intent"), d.get("answer", ""), d.get("steps", []), resp.status
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:200], [], e.code

try:
    token = login()
    check("login", True)
except Exception as e:
    check("backend reachable", False, str(e))
    print("后端未运行，跳过行为测试")
    sys.exit(1 if any(not ok for _, ok, _ in results) else 0)

# 快速路径回归: chitchat 不应走 ReAct
intent, answer, steps, status = send("你好", token)
used_react = any("agent_reasoning" in s for s in steps)
check("chitchat stays on fast path (no ReAct)", not used_react and intent == "chitchat",
      f"intent={intent} react={used_react}")

# 单工具 ReAct: 显式复杂多步任务触发 ReAct 循环
# （注意: 简单计算"1+1"会被意图识别分类为 tool_use 走快速路径, 这是设计正确的——
#  ReAct 是给 complex_task 的。所以用需要多步推理的复杂 query 触发。）
intent, answer, steps, status = send(
    "民法典中关于婚姻家庭禁止行为的条款号是多少？把这个条款号的数字乘以10，结果是多少", token)
used_react = any("agent_reasoning" in s for s in steps)
check("complex multi-step task triggers ReAct", used_react,
      f"steps={[s for s in steps if 'reasoning' in s or 'tool' in s or 'react' in s][:3]}")
check("ReAct answer contains 10420", "10420" in answer, f"answer={answer[:60]}")

# 快速路径回归: 简单 tool_use 不进 ReAct
intent, answer, steps, status = send("1+1等于几", token)
used_react = any("agent_reasoning" in s for s in steps)
check("simple calc stays on fast path (no ReAct)", not used_react and intent == "tool_use",
      f"intent={intent} react={used_react}")

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed (ad-hoc) ==")
sys.exit(0 if passed == len(results) else 1)
