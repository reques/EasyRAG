"""图谱构建服务（GraphRAG 阶段 5）— 从已入库 chunks 构建 Neo4j 图谱。

流程（由知识库详情页「知识图谱」Tab 触发，后台任务执行）：

1. 读取该知识库已入库的 chunks（Milvus chunk collection，按 kb_id 过滤）；
2. 逐 chunk 用可配置抽取器（默认 LLM JSON 抽取）抽取实体/关系；
3. 写入 Neo4j：实体节点 MERGE + RELATES 关系（携带 chunk 引用）；
4. 写入 PostgreSQL：实体/关系本体 + chunk 引用（与上传链路共用表，构建时去重）；
5. 写入 Milvus 图谱语义索引：唯一实体/唯一三元组（BGE-M3 向量）；
6. 全程维护 GraphBuildRun 状态与统计，供前端轮询展示。

任何一步失败都记录到 run.error_message 并置 status=failed，
已写入的部分保留（可重新构建，写入均为幂等）。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_knowledge import (
    GraphBuildRun,
    KnowledgeEntity,
    KnowledgeRelation,
)

logger = get_logger(__name__)
cfg = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# 触发入口
# ═══════════════════════════════════════════════════════════════════════════


async def create_build_run(
    kb_id: uuid.UUID,
    extractor: str = "llm",
) -> uuid.UUID:
    """创建构建运行记录（status=pending），返回 run id。

    同一知识库已有 pending/running 记录时拒绝创建（防重复触发/僵尸状态）。
    """
    async with get_session() as session:
        active = (await session.execute(
            select(GraphBuildRun).where(
                GraphBuildRun.knowledge_base_id == kb_id,
                GraphBuildRun.status.in_(["pending", "running"]),
            ).limit(1)
        )).scalars().all()
        if active:
            raise ValueError("该知识库已有进行中的图谱构建，请等待其完成后再试")
        run = GraphBuildRun(
            knowledge_base_id=kb_id,
            status="pending",
            extractor=extractor,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run.id


async def mark_interrupted_runs() -> int:
    """把遗留的 pending/running 构建记录标记为 failed（服务重启/强杀导致）。

    服务启动时调用：后台任务随进程死亡，状态不可能再被更新，
    不清理的话前端会永远显示"正在构建"。
    """
    async with get_session() as session:
        result = await session.execute(
            GraphBuildRun.__table__.update()
            .where(GraphBuildRun.status.in_(["pending", "running"]))
            .values(
                status="failed",
                error_message="服务重启，构建中断",
                finished_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        return result.rowcount or 0


async def run_build(run_id: uuid.UUID) -> None:
    """执行一次图谱构建（后台任务入口；异常捕获并落库）。"""
    async with get_session() as session:
        run = await session.get(GraphBuildRun, run_id)
        if run is None:
            logger.error("[graph_build] run %s not found", run_id)
            return
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await session.commit()

    kb_id = None
    try:
        async with get_session() as session:
            run = await session.get(GraphBuildRun, run_id)
            kb_id = run.knowledge_base_id
            extractor_name = run.extractor

        # 1) 读取已入库 chunks
        chunks = await _load_kb_chunks(kb_id)
        async with get_session() as session:
            run = await session.get(GraphBuildRun, run_id)
            run.total_chunks = len(chunks)
            await session.commit()
        if not chunks:
            raise ValueError("知识库暂无已入库的文档块（chunks），请先上传文档")

        # 2) 抽取
        from app.rag.extractors import get_extractor

        extractor = get_extractor(extractor_name)

        async def report(i: int, total: int, message: str) -> None:
            if i % 5 == 0 or i == total:
                async with get_session() as s:
                    r = await s.get(GraphBuildRun, run_id)
                    if r is not None:
                        r.processed_chunks = i
                        await s.commit()

        results = await extractor.extract_batch(chunks, progress_callback=report)

        # 3) Neo4j 写入（同步 driver → 线程池）
        await asyncio.to_thread(_write_neo4j, str(kb_id), chunks, results)

        # 4) PostgreSQL 本体写入（去重）
        entities_found, relations_found = await _persist_pg(kb_id, chunks, results)

        # 5) Milvus 唯一实体/三元组语义索引
        entities_indexed, relations_indexed = await asyncio.to_thread(
            _index_unique_graph_items, str(kb_id), results
        )

        async with get_session() as session:
            run = await session.get(GraphBuildRun, run_id)
            run.status = "completed"
            run.processed_chunks = len(chunks)
            run.entities_found = entities_found
            run.relations_found = relations_found
            run.entities_indexed = entities_indexed
            run.relations_indexed = relations_indexed
            run.finished_at = datetime.now(timezone.utc)
            await session.commit()
        logger.info(
            "[graph_build] run %s completed: %d chunks -> %d entities, %d relations "
            "(indexed: %d/%d)",
            run_id, len(chunks), entities_found, relations_found,
            entities_indexed, relations_indexed,
        )
    except Exception as exc:
        logger.exception("[graph_build] run %s failed: %s", run_id, exc)
        async with get_session() as session:
            run = await session.get(GraphBuildRun, run_id)
            if run is not None:
                run.status = "failed"
                run.error_message = str(exc)[:2000]
                run.finished_at = datetime.now(timezone.utc)
                await session.commit()


# ═══════════════════════════════════════════════════════════════════════════
# 内部步骤
# ═══════════════════════════════════════════════════════════════════════════


async def _load_kb_chunks(kb_id: uuid.UUID) -> List[Tuple[str, Dict[str, str]]]:
    """从 Milvus chunk collection 读取某知识库的全部已入库 chunks。

    返回 [(content, {chunk_id, source, knowledge_base_id}), ...]。
    """
    from pymilvus import Collection, connections, utility

    connections.connect(host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
    if not utility.has_collection(cfg.MILVUS_COLLECTION):
        return []
    col = Collection(cfg.MILVUS_COLLECTION)
    try:
        col.load()
    except Exception:
        pass
    expr = f'knowledge_base_id == "{kb_id}"'
    chunks: List[Tuple[str, Dict[str, str]]] = []
    offset = 0
    batch = 1000
    while True:
        res = col.query(
            expr=expr,
            output_fields=["id", "content", "source"],
            limit=batch,
            offset=offset,
        )
        if not res:
            break
        for r in res:
            chunks.append((
                r.get("content", ""),
                {
                    "chunk_id": r.get("id", ""),
                    "source": r.get("source", ""),
                    "knowledge_base_id": str(kb_id),
                },
            ))
        offset += len(res)
        if len(res) < batch:
            break
    logger.info("[graph_build] kb %s: %d chunks loaded from Milvus", kb_id, len(chunks))
    return chunks


def _write_neo4j(
    kb_id: str,
    chunks: Sequence[Tuple[str, Dict[str, str]]],
    results: Sequence[Any],
) -> None:
    """实体/关系幂等写入 Neo4j（带 chunk 引用）。"""
    from backend.storage.neo4j.client import get_neo4j_client

    client = get_neo4j_client()
    if not client.available:
        raise RuntimeError("Neo4j 不可达，图谱构建中止（请先 docker compose up -d neo4j）")
    client.init_schema()
    for (text, meta), result in zip(chunks, results):
        cid = meta.get("chunk_id", "")
        for e in result.entities:
            client.upsert_entity(kb_id, e.name, e.entity_type, e.description)
        for r in result.relations:
            client.upsert_relation(
                kb_id, r.source, r.target, r.relation, r.description, cid
            )


async def _persist_pg(
    kb_id: uuid.UUID,
    chunks: Sequence[Tuple[str, Dict[str, str]]],
    results: Sequence[Any],
) -> Tuple[int, int]:
    """实体/关系本体 + chunk 引用写入 PostgreSQL（构建去重，不重复插行）。"""
    async with get_session() as session:
        existing_entities = set((await session.execute(
            select(KnowledgeEntity.name).where(
                KnowledgeEntity.knowledge_base_id == kb_id
            )
        )).scalars().all())
        existing_rels = set((await session.execute(
            select(
                KnowledgeRelation.source_entity,
                KnowledgeRelation.target_entity,
                KnowledgeRelation.relation_type,
            ).where(KnowledgeRelation.knowledge_base_id == kb_id)
        )).all())

        entities_found = 0
        relations_found = 0
        for (text, meta), result in zip(chunks, results):
            chunk_ref = meta.get("chunk_id", "")
            for e in result.entities:
                entities_found += 1
                if e.name in existing_entities:
                    continue
                session.add(KnowledgeEntity(
                    knowledge_base_id=kb_id,
                    name=e.name,
                    entity_type=e.entity_type,
                    description=e.description,
                    source_chunks=chunk_ref,
                ))
                existing_entities.add(e.name)
            for r in result.relations:
                relations_found += 1
                key = (r.source, r.target, r.relation)
                if key in existing_rels:
                    continue
                session.add(KnowledgeRelation(
                    knowledge_base_id=kb_id,
                    source_entity=r.source,
                    target_entity=r.target,
                    relation_type=r.relation,
                    description=r.description,
                ))
                existing_rels.add(key)
        await session.commit()
        return entities_found, relations_found


def _index_unique_graph_items(
    kb_id: str,
    results: Sequence[Any],
) -> Tuple[int, int]:
    """把唯一实体/唯一三元组写入 Milvus 图谱语义索引。"""
    from app.rag.embeddings import get_embedder
    from app.rag.graph_vector_index import (
        entity_key,
        get_graph_vector_index,
        triple_key,
    )

    # 去重收集
    entities: Dict[str, Any] = {}
    triples: Dict[Tuple[str, str, str], Any] = {}
    for result in results:
        for e in result.entities:
            entities.setdefault(e.name, e)
        for r in result.relations:
            triples.setdefault((r.source, r.relation, r.target), r)

    items: List[Dict[str, Any]] = []
    texts: List[str] = []
    for name, e in entities.items():
        items.append({
            "id": entity_key(kb_id, name),
            "kind": "entity",
            "kb_id": kb_id,
            "text": name,
            "source": "",
            "target": "",
            "relation": "",
        })
        texts.append(f"{name}：{e.entity_type}。{e.description}"[:500])
    for (src, rel, tgt), r in triples.items():
        items.append({
            "id": triple_key(kb_id, src, rel, tgt),
            "kind": "triple",
            "kb_id": kb_id,
            "text": f"{src} -{rel}-> {tgt}",
            "source": src,
            "target": tgt,
            "relation": rel,
        })
        texts.append(f"{src} {rel} {tgt}：{r.description}"[:500])

    if not items:
        return 0, 0

    # 批量向量化（按配置批大小）
    embedder = get_embedder()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), cfg.GRAPH_BUILD_BATCH_SIZE):
        vectors.extend(embedder.embed_texts(
            texts[start:start + cfg.GRAPH_BUILD_BATCH_SIZE]
        ))

    index = get_graph_vector_index()
    index.upsert(items, vectors)
    logger.info(
        "[graph_build] kb %s: indexed %d entities + %d triples into Milvus",
        kb_id, len(entities), len(triples),
    )
    return len(entities), len(triples)


# ═══════════════════════════════════════════════════════════════════════════
# 查询 / 重置
# ═══════════════════════════════════════════════════════════════════════════


async def latest_build_run(kb_id: uuid.UUID) -> Optional[GraphBuildRun]:
    async with get_session() as session:
        runs = (await session.execute(
            select(GraphBuildRun)
            .where(GraphBuildRun.knowledge_base_id == kb_id)
            .order_by(GraphBuildRun.created_at.desc())
            .limit(1)
        )).scalars().all()
        return runs[0] if runs else None


async def reset_kb_graph(kb_id: uuid.UUID) -> None:
    """重置知识库图谱：清 Neo4j 子图 + Milvus 语义索引 + PG 本体 + 内存缓存。

    上传链路抽取的 PG 图谱数据（阶段 2C）一并清理，保持两套图谱一致。
    """
    kb_str = str(kb_id)
    # Neo4j
    try:
        from backend.storage.neo4j.client import get_neo4j_client

        client = get_neo4j_client()
        if client.available:
            await asyncio.to_thread(client.clear_kb, kb_str)
    except Exception as exc:
        logger.warning("[graph_build] neo4j clear failed: %s", exc)
    # Milvus 语义索引
    try:
        from app.rag.graph_vector_index import get_graph_vector_index

        await asyncio.to_thread(get_graph_vector_index().delete_by_kb, kb_str)
    except Exception as exc:
        logger.warning("[graph_build] milvus graph index clear failed: %s", exc)
    # PostgreSQL
    async with get_session() as session:
        await session.execute(
            KnowledgeEntity.__table__.delete().where(
                KnowledgeEntity.knowledge_base_id == kb_id
            )
        )
        await session.execute(
            KnowledgeRelation.__table__.delete().where(
                KnowledgeRelation.knowledge_base_id == kb_id
            )
        )
        await session.commit()
    # 内存缓存
    try:
        from app.rag.graph_cache import graph_cache

        graph_cache.clear_kb(kb_str)
    except Exception:
        pass
