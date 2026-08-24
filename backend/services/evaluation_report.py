"""评测报告生成 - 把一次运行沉淀为可分享 / 可归档的 Markdown 报告。

报告包含五部分：运行环境快照（可复现）、聚合指标、RAGAs 指标、
逐条明细、失败分析。产出可直接放进技术文档或周报，便于向团队
解释「这次检索配置改动的收益是什么」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _reference_mode_label(mode: str) -> str:
    return {
        "chunk_ids": "精确 chunk 标注",
        "chunk": "单 chunk",
        "file": "整文件（兜底）",
        "negative": "负样本",
    }.get(mode, mode or "-")


def _metrics_table(metrics: Dict[str, Any]) -> str:
    rows = [
        ("HitRate@K", metrics.get("hit_rate_at_k")),
        ("MRR@K", metrics.get("mrr_at_k")),
        ("Recall@K", metrics.get("recall_at_k")),
        ("Precision@K", metrics.get("precision_at_k")),
        ("nDCG@K", metrics.get("ndcg_at_k")),
        ("平均检索得分", metrics.get("avg_score")),
        ("文件级 HitRate@K", metrics.get("file_hit_rate_at_k")),
        ("文件级 MRR@K", metrics.get("file_mrr_at_k")),
    ]
    lines = ["| 指标 | 数值 |", "|---|---|"]
    for name, value in rows:
        lines.append(f"| {name} | {_fmt(value)} |")
    return "\n".join(lines)


def _ragas_table(ragas: Optional[Dict[str, Any]]) -> str:
    if not ragas:
        return "未启用 RAGAs 指标。"
    metrics = ragas.get("metrics") or {}
    if ragas.get("status") in ("completed", "partial") and metrics:
        lines = ["| 指标 | 数值 |", "|---|---|"]
        for key, value in metrics.items():
            lines.append(f"| {key} | {_fmt(value)} |")
        return "\n".join(lines)
    return f"RAGAs 状态：{ragas.get('status', 'unknown')}（{ragas.get('error', '')}）"


def _details_table(details: List[Dict[str, Any]]) -> str:
    if not details:
        return "无逐条数据。"
    lines = [
        "| # | 问题 | 参考类型 | 相关数 | 命中排名 | Top 得分 | 返回数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, d in enumerate(details, start=1):
        rank = d.get("chunk_hit_rank") or d.get("file_hit_rank") or "-"
        lines.append(
            "| {i} | {q} | {mode} | {n} | {rank} | {score} | {ret} |".format(
                i=index,
                q=(d.get("question") or "")[:60],
                mode=_reference_mode_label(d.get("reference_mode") or ""),
                n=d.get("expected_chunk_count", 0),
                rank=rank,
                score=_fmt(d.get("top_score")),
                ret=d.get("returned", 0),
            )
        )
    return "\n".join(lines)


def _analysis_section(analysis: Dict[str, Any]) -> str:
    if not analysis:
        return "无失败分析数据。"
    missed = analysis.get("missed") or []
    low_recall = analysis.get("low_recall") or []
    false_positives = analysis.get("false_positives") or []
    parts = [
        f"- 未命中（missed）：{analysis.get('missed_count', 0)} 条"
    ]
    for item in missed:
        parts.append(f"  - {item.get('question', '')[:60]}（top_score={_fmt(item.get('top_score'))}）")
    parts.append(f"- 低召回（low_recall < 50%）：{analysis.get('low_recall_count', 0)} 条")
    for item in low_recall:
        parts.append(f"  - {item.get('question', '')[:60]}（recall@K={_fmt(item.get('recall_at_k'))}）")
    parts.append(f"- 负样本误报（false positive）：{analysis.get('false_positive_count', 0)} 条")
    for item in false_positives:
        parts.append(f"  - {item.get('question', '')[:60]}（top_score={_fmt(item.get('top_score'))}）")
    return "\n".join(parts)


def build_markdown_report(
    *,
    run_name: str,
    created_at: str,
    knowledge_base_name: str,
    metrics: Dict[str, Any],
    ragas: Optional[Dict[str, Any]] = None,
) -> str:
    """根据一次评估运行的 metrics 载荷生成 Markdown 报告。"""
    run_metadata = metrics.get("run_metadata") or {}
    k = metrics.get("k", "-")
    details = metrics.get("details") or []
    analysis = metrics.get("analysis") or {}

    meta_lines = [
        "| 配置项 | 值 |",
        "|---|---|",
        f"| 运行名称 | {run_name} |",
        f"| 生成时间 | {created_at or '-'} |",
        f"| 知识库 | {knowledge_base_name or '-'} |",
        f"| Top-K | {k} |",
        f"| 用例数 | {len(details)} |",
        f"| 指标版本 | {metrics.get('metrics_version', '-')} |",
    ]
    for key, value in run_metadata.items():
        meta_lines.append(f"| {key} | {value} |")

    return "\n".join([
        f"# RAG 检索评估报告 - {run_name}",
        "",
        "## 1. 运行环境（可复现性快照）",
        "",
        "\n".join(meta_lines),
        "",
        "## 2. 确定性检索指标（无 LLM）",
        "",
        _metrics_table(metrics),
        "",
        "## 3. RAGAs 指标",
        "",
        _ragas_table(ragas),
        "",
        "## 4. 逐条明细",
        "",
        _details_table(details),
        "",
        "## 5. 失败分析",
        "",
        _analysis_section(analysis),
        "",
        "---",
        "",
        "> 报告由 EasyRAG 评测体系自动生成。指标口径：确定性指标基于",
        "> 二值相关性（相关 chunk 命中即计 1）；RAGAs ID 指标做集合匹配；",
        "> LLM 版指标（ContextPrecision/ContextRecall）提供排序与语义维度。",
        "",
    ])