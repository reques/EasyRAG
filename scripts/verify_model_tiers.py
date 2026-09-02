"""hermes-verify: 模型分级接口验证（非测试套件）。

针对 Task 4 模型分级:
1. 配置: LLM_FAST_BASE_URL/LLM_FAST_API_KEY/LLM_FAST_MODEL 存在且默认 None
2. get_llm_client() 默认 main 不变（向后兼容）
3. tier="fast" 未配置 LLM_FAST_MODEL 时回退主模型
4. tier="fast" 配置后返回 fast client（不同 model 名）
5. 同一 tier 单例缓存

用法: D:/Anaconda3/envs/stage1-agent/python.exe scripts/verify_model_tiers.py
"""
import ast
import sys

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

ROOT = r"E:\Learn_Agent\develop\EasyRAG"
sys.path.insert(0, ROOT)

print("== 1. 静态 ==")
cfg_src = open(f"{ROOT}/app/core/config.py", encoding="utf-8").read()
cli_src = open(f"{ROOT}/app/llm/client.py", encoding="utf-8").read()
try:
    ast.parse(cfg_src); ast.parse(cli_src)
    check("syntax", True)
except SyntaxError as e:
    check("syntax", False, str(e)); sys.exit(1)

check("LLM_FAST_BASE_URL config", "LLM_FAST_BASE_URL" in cfg_src)
check("LLM_FAST_API_KEY config", "LLM_FAST_API_KEY" in cfg_src)
check("LLM_FAST_MODEL config", "LLM_FAST_MODEL" in cfg_src)
check("get_llm_client has tier param", 'def get_llm_client(tier: str = "main")' in cli_src)
check("fast falls back when unconfigured", 'tier = "main"' in cli_src)

print("== 2. 行为 ==")
from app.core.config import get_settings
from app.llm.client import get_llm_client

cfg = get_settings()
check("LLM_FAST_MODEL default None", cfg.LLM_FAST_MODEL is None, f"got {cfg.LLM_FAST_MODEL!r}")

main_client = get_llm_client()
check("default is main tier", main_client.model == cfg.LLM_MODEL, f"model={main_client.model}")

# fast 未配置 → 回退主模型
fast_client = get_llm_client(tier="fast")
check("fast unconfigured falls back to main model",
      fast_client.model == cfg.LLM_MODEL, f"fast model={fast_client.model}")
check("fast fallback returns same singleton as main", fast_client is get_llm_client())

# 配置 fast 后 → 独立 client
import app.llm.client as client_mod
cfg.LLM_FAST_MODEL = "deepseek-chat-fast-test"  # 模拟配置
cfg.LLM_FAST_BASE_URL = None
cfg.LLM_FAST_API_KEY = None
client_mod._tier_clients.clear()  # 清缓存强制重建
fast2 = get_llm_client(tier="fast")
check("fast configured returns fast model", fast2.model == "deepseek-chat-fast-test",
      f"model={fast2.model}")
check("fast and main are different clients", fast2 is not get_llm_client(tier="main"))
check("fast singleton cached", get_llm_client(tier="fast") is fast2)
# 还原
cfg.LLM_FAST_MODEL = None
client_mod._tier_clients.clear()

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n== {passed}/{len(results)} checks passed (ad-hoc) ==")
sys.exit(0 if passed == len(results) else 1)
