"""Reciprocal Rank Fusion (RRF) 融合（GraphRAG 阶段 5）。

多路检索结果（图谱召回 / 向量 / BM25）按排名融合：每个 item 的贡献为
1/(k + rank)，k 默认 60。RRF 对分数尺度不敏感，适合融合异质检索路。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """融合多个排名列表，返回按 RRF 分数降序的 [(item, score)]。

    每个列表内部去重保留首次出现位置；item 的最终分数为其在各路
    1/(k + rank) 之和。
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        seen = set()
        for rank, item in enumerate(ranked):
            if item in seen:
                continue
            seen.add(item)
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rrf_normalize(fused: List[Tuple[str, float]]) -> Dict[str, float]:
    """把 RRF 分数归一化到 [0, 1]（供融合评分使用）。

    第一名固定为 1.0，其余按与第一名的比值。
    """
    if not fused:
        return {}
    top = fused[0][1]
    if top <= 0:
        return {}
    return {item: round(score / top, 4) for item, score in fused}
