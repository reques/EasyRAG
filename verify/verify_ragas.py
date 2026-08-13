"""RAGAS 评委兼容性验证（一次性脚本）。

目的：验证 RAGAS 0.4.3 能否用项目现有的 DeepSeek（网关）/ Qwen（DashScope）
作为「LLM 评委」跑通检索类评估指标，为后端接入排除兼容性风险。

运行（用隔离 venv，不动项目环境）：
    <venv>/python verify/verify_ragas.py

说明：
  - 不 import app.*（避免 PYTHONPATH 污染），直接用 dotenv 读 .env 拿 LLM 配置。
  - 只测检索类三个指标：ContextPrecision / ContextRecall / ContextRelevance。
  - 每个评委（DeepSeek / Qwen）各跑一轮，打印逐指标得分；失败则打印原始异常。
"""
from __future__ import annotations

import os
import sys
import warnings

from dotenv import load_dotenv

# 屏蔽 ragas 旧式指标/旧式 LLM wrapper 的弃用告警（弃用事实已在代码注释中说明）。
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def judge_configs():
    """从 .env 收集可用的评委 LLM 配置（DeepSeek 走网关，Qwen 走 DashScope）。"""
    cfg = []
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        cfg.append({
            "label": "DeepSeek(网关 open.aiservetech.com.cn)",
            "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": ds_key,
            "model": os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        })
    qwen_key = os.getenv("DASHSCOPE_API_KEY")
    qwen_model = os.getenv("QWEN_MODEL", "qwen3.6-flash")
    if qwen_key:
        cfg.append({
            "label": "Qwen(DashScope)",
            "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "api_key": qwen_key,
            "model": qwen_model,
        })
    elif ds_key:
        # 本工程 .env 未填 DASHSCOPE_API_KEY：试经网关用 qwen 模型（复用网关 key，
        # 同 app/llm/models.py 里 provider_key 的回退思路），验证网关是否支持 qwen。
        cfg.append({
            "label": "Qwen(经网关，复用 DeepSeek key)",
            "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": ds_key,
            "model": qwen_model,
        })
    return cfg


# 迷你评估集：每条含问题 + 检索到的上下文（2 条，一条相关一条干扰）+ 参考答案 + 生成回答。
# 用于验证 RAGAS 能否出分，不代表真实知识库内容。
SAMPLE_DATASET = [
    {
        "user_input": "相对论是谁提出的？",
        "retrieved_contexts": [
            "阿尔伯特·爱因斯坦在1905年提出狭义相对论，1915年提出广义相对论。",
            "艾萨克·牛顿提出了万有引力定律和三大运动定律。",
        ],
        "response": "相对论是由阿尔伯特·爱因斯坦提出的。",
        "reference": "相对论由爱因斯坦提出。",
    },
    {
        "user_input": "法国的首都是哪座城市？",
        "retrieved_contexts": [
            "巴黎是法国的首都和最大城市，位于塞纳河畔。",
            "柏林是德国的首都。",
        ],
        "response": "法国的首都是巴黎。",
        "reference": "法国的首都是巴黎。",
    },
    {
        "user_input": "光合作用会释放什么气体？",
        "retrieved_contexts": [
            "光合作用过程中，植物吸收二氧化碳并释放氧气。",
            "呼吸作用消耗氧气并释放二氧化碳。",
        ],
        "response": "光合作用会释放氧气。",
        "reference": "光合作用释放氧气。",
    },
]


def run_one(cfg) -> None:
    import httpx
    from langchain_openai import ChatOpenAI

    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextPrecision, ContextRecall, ContextRelevance

    # 0.4.3 的旧式指标走 agenerate_text 通道，评委 LLM 必须是 LangchainLLMWrapper
    # （llm_factory 返回的 InstructorLLM 没有 agenerate_text，会报错）。
    # trust_env=False 与项目 LLMClient 一致，绕过本机代理环境变量。
    langchain_llm = ChatOpenAI(
        model=cfg["model"],          # alias model_name
        api_key=cfg["api_key"],      # alias openai_api_key
        base_url=cfg["base_url"],    # alias openai_api_base
        temperature=0,
        max_retries=1,
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )
    judge = LangchainLLMWrapper(langchain_llm)

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        ContextRelevance(),
    ]
    ds = EvaluationDataset.from_list(SAMPLE_DATASET)

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge,
        show_progress=False,
        raise_exceptions=True,
    )

    df = result.to_pandas()
    input_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    metric_cols = [c for c in df.columns if c not in input_cols]
    for c in metric_cols:
        per_sample = [round(float(x), 3) if x == x else None for x in df[c]]
        print(f"    {c:24s} = 逐条 {per_sample}  均值 = {df[c].mean():.3f}")



def main() -> int:
    import importlib.metadata as m

    print("=" * 64)
    print("RAGAS 评委兼容性验证")
    print("=" * 64)
    print(f"ragas={m.version('ragas')}  openai={m.version('openai')}  "
          f"langchain={m.version('langchain')}  langchain-community={m.version('langchain-community')}")
    print()

    configs = judge_configs()
    if not configs:
        print("[失败] .env 里 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 均缺失，无从测试。")
        return 2

    ok = True
    for cfg in configs:
        print(f"── 评委：{cfg['label']}  (model={cfg['model']}) ──")
        try:
            run_one(cfg)
            print("    >>> 该评委出分成功 ✅\n")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"    >>> 该评委失败 ❌")
            print(f"        异常类型: {type(exc).__name__}")
            print(f"        异常信息: {exc}\n")

    print("=" * 64)
    print("结论：" + ("至少一个评委可正常当 RAGAS 评委，兼容性通过 ✅" if ok else "全部评委失败，需排查 ❌"))
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
