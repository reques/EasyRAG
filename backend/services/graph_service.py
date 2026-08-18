"""知识图谱服务（阶段 2C）— 实体/关系抽取 + 图谱查询。

设计取舍：
- 存储用 PostgreSQL 两张表（entities/relations），不引入 Neo4j — 阶段 2 先验证价值，
  子图查询用 SQL 递归即可覆盖 1-2 跳邻居场景；
- 抽取走 LLM JSON 模式，逐 chunk 调用，失败 chunk 跳过不阻塞上传主链路；
- GRAPH_ENABLED=False 时一切旁路，上传与检索行为与阶段 1 完全一致。
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.storage.postgres.models_knowledge import KnowledgeEntity, KnowledgeRelation
from app.rag.graph_cache import graph_cache

logger = get_logger(__name__)
cfg = get_settings()

_EXTRACT_PROMPT = """从下面的文本中抽取知识图谱元素（实体与关系）。

要求：
1. 实体：名词性概念（技术、产品、人物、组织、方法等），给出 name、type（如 technology/product/person/concept）、一句话 description。
2. 关系：实体之间的有向关系，给出 source、target、relation（如 "属于"/"使用"/"对比"/"依赖"）、一句话 description。
3. 只抽取文本中明确表达的信息，不要臆造。实体名用原文表述。
4. 如果没有可抽取的内容，返回空数组。

严格输出 JSON（不要输出其他内容）：
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]}}

文本：
{chunk}"""


async def extract_graph_from_chunks(
    session: AsyncSession,
    kb_id: uuid.UUID,
    chunks: List[tuple],
    source_name: str,
    progress_callback: Optional[
        Callable[[int, int, str], Awaitable[None]]
    ] = None,
) -> Dict[str, int]:
    """对一组 chunk 抽取实体/关系并入库。返回 {"entities": n, "relations": m}。

    chunks: [(text, metadata), ...]。逐 chunk 调 LLM，单 chunk 失败只记日志。
    """
    from app.llm.client import get_llm_client

    llm = get_llm_client()
    total_entities = 0
    total_relations = 0
    sampled = chunks[: cfg.GRAPH_MAX_CHUNKS_PER_FILE]
    total_sampled = len(sampled)

    for i, (text, meta) in enumerate(sampled):
        if progress_callback:
            await progress_callback(
                i,
                total_sampled,
                f"正在抽取知识图谱 {i}/{total_sampled}",
            )
        try:
            if len(text.strip()) < 50:  # 太短的 chunk 没有抽取价值
                continue
            raw = await llm.chat_json([{"role": "user", "content": _EXTRACT_PROMPT.format(chunk=text[:2000])}])
        except Exception as exc:
            logger.warning("[graph] extract failed for chunk %d of '%s': %s", i, source_name, exc)
            continue
        finally:
            if progress_callback:
                await progress_callback(
                    i + 1,
                    total_sampled,
                    f"正在抽取知识图谱 {i + 1}/{total_sampled}",
                )

        entities = raw.get("entities") or []
        relations = raw.get("relations") or []
        chunk_ref = f"{source_name}#{meta.get('chunk_index', i)}"

        for e in entities[:20]:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            session.add(KnowledgeEntity(
                knowledge_base_id=kb_id,
                name=name,
                entity_type=(e.get("type") or "concept")[:64],
                description=(e.get("description") or "")[:1024],
                source_chunks=chunk_ref,
            ))
            graph_cache.upsert_entity(
                name=name,
                entity_type=(e.get("type") or "concept"),
                description=(e.get("description") or ""),
                kb_id=str(kb_id),
            )
            total_entities += 1

        for r in relations[:20]:
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            rel = (r.get("relation") or "").strip()
            if not (src and tgt and rel):
                continue
            session.add(KnowledgeRelation(
                knowledge_base_id=kb_id,
                source_entity=src,
                target_entity=tgt,
                relation_type=rel[:128],
                description=(r.get("description") or "")[:1024],
            ))
            graph_cache.add_relation(
                source=src,
                target=tgt,
                relation_type=rel,
                description=(r.get("description") or ""),
                kb_id=str(kb_id),
            )
            total_relations += 1

    logger.info(
        "[graph] '%s' -> %d entities, %d relations (from %d chunks)",
        source_name, total_entities, total_relations, len(sampled),
    )
    return {"entities": total_entities, "relations": total_relations}


async def query_related(
    session: AsyncSession,
    kb_id: uuid.UUID,
    query: str,
    top_entities: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """图谱检索：query 关键词匹配实体 → 拉 1 跳邻居关系，返回结构化子图。

    返回 [{"entity": str, "relations": [{"source","relation","target","description"}]}]
    供 prompt 注入使用。匹配策略：实体名包含 query 任一连贯子串（长度≥2）。
    """
    top = top_entities or cfg.GRAPH_QUERY_TOP_ENTITIES

    # 关键词粗匹配：query 中的连续中文/英文片段（≥2 字）逐个 LIKE
    import re
    keywords = [w for w in re.findall(r"[一-鿿]{2,}|[A-Za-z0-9_]{2,}", query)][:5]
    if not keywords:
        return []

    stmt = select(KnowledgeEntity).where(KnowledgeEntity.knowledge_base_id == kb_id)
    all_entities = (await session.execute(stmt)).scalars().all()
    matched = [
        e for e in all_entities
        if any(k in e.name or e.name in k for k in keywords)
    ][:top]
    if not matched:
        return []

    names = {e.name for e in matched}
    rel_stmt = select(KnowledgeRelation).where(
        KnowledgeRelation.knowledge_base_id == kb_id,
    )
    all_rels = (await session.execute(rel_stmt)).scalars().all()

    sub_graphs: List[Dict[str, Any]] = []
    for e in matched:
        rels = [
            {
                "source": r.source_entity,
                "relation": r.relation_type,
                "target": r.target_entity,
                "description": r.description or "",
            }
            for r in all_rels
            if r.source_entity == e.name or r.target_entity == e.name
        ][:6]
        sub_graphs.append({
            "entity": e.name,
            "type": e.entity_type,
            "description": e.description or "",
            "relations": rels,
        })
    return sub_graphs


def format_subgraph_for_prompt(sub_graphs: List[Dict[str, Any]]) -> str:
    """把子图格式化为可注入 prompt 的文本。"""
    lines = ["【知识图谱相关信息】"]
    for g in sub_graphs:
        lines.append(f"实体「{g['entity']}」({g['type']}): {g['description']}")
        for r in g["relations"]:
            lines.append(f"  - {r['source']} --[{r['relation']}]--> {r['target']}: {r['description']}")
    return "\n".join(lines)
