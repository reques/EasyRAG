"""GraphRAG 图谱模块离线验证（不依赖 Neo4j / Milvus / LLM）。

验证内容：
1. RRF 融合：跨路命中的 item 排第一、归一化正确
2. 实体/三元组索引 key 生成
3. 抽取结果数据类与批量抽取的短文本跳过逻辑
4. neo4j client 模块可导入（需安装 neo4j-graphrag / neo4j 包；未装则跳过）

用法：PYTHONPATH= python verify/verify_graphrag.py
"""

from __future__ import annotations

import sys


def main() -> int:
    failures = 0

    # ── 1. RRF 融合 ──────────────────────────────────────────────────────
    from app.rag.rrf import rrf_fuse, rrf_normalize

    fused = rrf_fuse([["c1", "c2", "c3"], ["c4", "c1", "c5"]], k=60)
    assert fused[0][0] == "c1", "跨路命中的 c1 应排第一"
    norm = rrf_normalize(fused)
    assert abs(norm["c1"] - 1.0) < 1e-6, "第一名归一化为 1.0"
    print("[1/4] RRF 融合 OK:", [i for i, _ in fused])

    # ── 2. 索引 key ──────────────────────────────────────────────────────
    from app.rag.graph_vector_index import entity_key, triple_key

    assert entity_key("kb1", "实体A") == "e:kb1:实体A"
    assert triple_key("kb1", "A", "属于", "B") == "t:kb1:A|属于|B"
    print("[2/4] 实体/三元组 key OK")

    # ── 3. 抽取结果与批量抽取 ────────────────────────────────────────────
    from app.rag.extractors import ExtractionResult, GraphExtractor

    class _FakeExtractor(GraphExtractor):
        name = "fake"

        async def extract(self, text, meta=None):
            from app.rag.extractors import EntityExtraction, RelationExtraction

            return ExtractionResult(
                entities=[EntityExtraction(name=text.strip(), entity_type="concept")],
                relations=[RelationExtraction(source="A", target="B", relation="关联")],
            )

    import asyncio

    async def _run():
        fake = _FakeExtractor()
        # 短文本应被跳过（<50 字符）
        results = await fake.extract_batch([
            ("短文本", {}),
            ("这是一段足够长的测试文本，包含多个实体与它们之间的关系，"
             "用于验证批量抽取流程在真实 chunk 长度下能够正常工作并返回结构化结果。", {}),
        ])
        return results

    results = asyncio.run(_run())
    assert len(results) == 2
    assert results[0].empty, "短文本应被跳过"
    assert len(results[1].entities) == 1, "长文本应抽取到实体"
    assert results[1].relations[0].relation == "关联"
    print("[3/4] 抽取器批量逻辑 OK")

    # ── 4. neo4j client 可导入（依赖缺失时跳过）────────────────────────
    try:
        from backend.storage.neo4j.client import (
            ENTITY_LABEL,
            Neo4jUnavailableError,
            get_neo4j_client,
        )

        client = get_neo4j_client()
        # 不应在未连接时抛 import 错误
        print(f"[4/4] neo4j client 导入 OK (label={ENTITY_LABEL}, 连接探测={client.available})")
    except ImportError as exc:
        print(f"[4/4] neo4j client 跳过（未安装 neo4j 依赖: {exc}）")

    if failures:
        print(f"\n❌ {failures} 项失败")
        return 1
    print("\n✅ GraphRAG 图谱模块离线验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
