"""智能路由（auto 模式）验证。

用法：D:/Anaconda3/envs/stage1-agent/python.exe verify/verify_auto_route.py
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and __package__ is None:
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.agent_service import AgentService
from app.core.config import get_settings

cfg = get_settings()

results = []


def check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    results.append((name, ok, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def main():
    print("=" * 60)
    print("智能路由（auto 模式）验证")
    print("=" * 60)

    # 1. 配置项存在
    print("\n[1] 配置")
    check("AGENT_MODE 配置存在", hasattr(cfg, "AGENT_MODE"))
    check("AGENT_MODE 默认 auto", cfg.AGENT_MODE == "auto", f"实际: {cfg.AGENT_MODE}")

    # 2. _should_use_multi 判断逻辑
    print("\n[2] 智能判断")

    # 单一意图 → single
    check("问候 → single", not AgentService._should_use_multi("你好"))
    check("简单问答 → single", not AgentService._should_use_multi("今天天气怎么样"))
    check("短查询 → single", not AgentService._should_use_multi("1+1等于几"))

    # 多领域组合 → multi
    check(
        "法律+代码 → multi",
        AgentService._should_use_multi("帮我查一下劳动合同法，然后写一个 Python 脚本计算补偿金额"),
    )
    check(
        "检索+生成 → multi",
        AgentService._should_use_multi("搜索最新新闻并写一篇摘要"),
    )
    check(
        "分析+代码 → multi",
        AgentService._should_use_multi("分析这个算法并写个实现"),
    )

    # 长查询 + 连词 → multi
    long_query = "请详细分析公司裁员涉及的法律风险，包括经济补偿计算方式、社保缴纳义务、以及可能需要准备的法律文书，然后给我一个完整的操作 checklist"
    check("长查询+连词 → multi", AgentService._should_use_multi(long_query))

    # 3. 边界情况
    print("\n[3] 边界")
    check("空字符串 → single", not AgentService._should_use_multi(""))
    check("纯英文 → single", not AgentService._should_use_multi("hello world"))
    check(
        "仅单领域 → single",
        not AgentService._should_use_multi("帮我查一下劳动合同法第47条"),
    )

    # 汇总
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"结果: {passed}/{total} 通过")
    if passed < total:
        print("失败项:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
