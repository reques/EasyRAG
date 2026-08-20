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
import hashlib
import json
import math
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.extractors.base import ExtractionResult, GraphExtractor
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_knowledge import (
    GraphBuildRun,
    GraphExtractionCache,
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
            async with get_session() as s:
                r = await s.get(GraphBuildRun, run_id)
                if r is not None:
                    r.processed_chunks = i
                    await s.commit()

        results = await _extract_chunks_with_cache(
            kb_id,
            extractor,
            chunks,
            progress_callback=report,
            concurrency=cfg.GRAPH_EXTRACT_CONCURRENCY,
            cache_enabled=cfg.GRAPH_EXTRACT_CACHE_ENABLED,
        )

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


def _graph_cache_key(
    kb_id: uuid.UUID,
    extractor: GraphExtractor,
    text: str,
) -> str:
    """生成知识库隔离、模型/prompt 感知的稳定内容缓存键。"""
    payload = "\0".join((
        str(kb_id),
        extractor.cache_fingerprint(),
        extractor.cache_input(text),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _load_cached_extractions(
    kb_id: uuid.UUID,
    cache_keys: Sequence[str],
) -> Dict[str, ExtractionResult]:
    if not cache_keys:
        return {}
    async with get_session() as session:
        rows = (await session.execute(
            select(GraphExtractionCache).where(
                GraphExtractionCache.knowledge_base_id == kb_id,
                GraphExtractionCache.cache_key.in_(list(set(cache_keys))),
            )
        )).scalars().all()
    cached: Dict[str, ExtractionResult] = {}
    for row in rows:
        try:
            cached[row.cache_key] = ExtractionResult.from_dict(
                json.loads(row.result_json)
            )
        except Exception as exc:
            logger.warning(
                "[graph_build] ignoring corrupt extraction cache %s: %s",
                row.cache_key,
                exc,
            )
    return cached


async def _store_cached_extractions(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    # 同一文档可能包含完全相同的 chunk；单条 upsert 语句不能重复更新同一键。
    unique_records = {record["cache_key"]: record for record in records}
    values = list(unique_records.values())
    statement = pg_insert(GraphExtractionCache).values(values)
    statement = statement.on_conflict_do_update(
        index_elements=[GraphExtractionCache.cache_key],
        set_={
            "chunk_id": statement.excluded.chunk_id,
            "content_hash": statement.excluded.content_hash,
            "extractor": statement.excluded.extractor,
            "model_name": statement.excluded.model_name,
            "prompt_version": statement.excluded.prompt_version,
            "result_json": statement.excluded.result_json,
            "updated_at": func.now(),
        },
    )
    async with get_session() as session:
        await session.execute(statement)
        await session.commit()


async def _extract_chunks_with_cache(
    kb_id: uuid.UUID,
    extractor: GraphExtractor,
    chunks: List[tuple],
    *,
    progress_callback=None,
    concurrency: int = 4,
    cache_enabled: bool = True,
) -> List[ExtractionResult]:
    """复用命中结果，只把未命中的原始 chunk 交给打包抽取器。"""
    if not cache_enabled:
        return await extractor.extract_batch(
            chunks,
            progress_callback=progress_callback,
            concurrency=concurrency,
        )

    cache_keys = [
        _graph_cache_key(kb_id, extractor, text)
        for text, _meta in chunks
    ]
    try:
        cached = await _load_cached_extractions(kb_id, cache_keys)
    except Exception as exc:
        logger.warning("[graph_build] extraction cache lookup skipped: %s", exc)
        cached = {}

    results: List[Optional[ExtractionResult]] = [None] * len(chunks)
    missing_chunks: List[tuple] = []
    missing_indices: List[int] = []
    for index, (chunk, cache_key) in enumerate(zip(chunks, cache_keys)):
        if cache_key in cached:
            results[index] = cached[cache_key]
        else:
            missing_indices.append(index)
            missing_chunks.append(chunk)

    cache_hits = len(chunks) - len(missing_chunks)
    if progress_callback and cache_hits:
        callback_result = progress_callback(
            cache_hits,
            len(chunks),
            f"复用图谱抽取缓存 {cache_hits}/{len(chunks)}",
        )
        if asyncio.iscoroutine(callback_result):
            await callback_result

    if missing_chunks:
        async def report_misses(done: int, _total: int, _message: str) -> None:
            if not progress_callback:
                return
            callback_result = progress_callback(
                cache_hits + done,
                len(chunks),
                f"正在抽取知识图谱 {cache_hits + done}/{len(chunks)}",
            )
            if asyncio.iscoroutine(callback_result):
                await callback_result

        missing_results = await extractor.extract_batch(
            missing_chunks,
            progress_callback=report_misses,
            concurrency=concurrency,
        )
        records: List[Dict[str, Any]] = []
        for original_index, result in zip(missing_indices, missing_results):
            results[original_index] = result
            if not result.cacheable:
                continue
            text, meta = chunks[original_index]
            cache_input = extractor.cache_input(text)
            records.append({
                "cache_key": cache_keys[original_index],
                "knowledge_base_id": kb_id,
                "chunk_id": str(meta.get("chunk_id", ""))[:512],
                "content_hash": hashlib.sha256(
                    cache_input.encode("utf-8")
                ).hexdigest(),
                "extractor": extractor.name[:64],
                "model_name": str(extractor.model_name)[:256],
                "prompt_version": extractor.prompt_version[:64],
                "result_json": json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            })
        try:
            await _store_cached_extractions(records)
        except Exception as exc:
            logger.warning("[graph_build] extraction cache write skipped: %s", exc)

    logger.info(
        "[graph_build] extraction cache: %d hit(s), %d miss(es)",
        cache_hits,
        len(missing_chunks),
    )
    return [
        result if result is not None else ExtractionResult(cacheable=False)
        for result in results
    ]


def _write_neo4j(
    kb_id: str,
    chunks: Sequence[Tuple[str, Dict[str, str]]],
    results: Sequence[Any],
) -> None:
    """实体/关系幂等写入 Neo4j（批量 UNWIND，带 chunk 引用）。"""
    from backend.storage.neo4j.client import get_neo4j_client

    client = get_neo4j_client()
    if not client.available:
        raise RuntimeError("Neo4j 不可达，图谱构建中止（请先 docker compose up -d neo4j）")
    client.init_schema()

    # 去重聚合：实体按 name，关系按 (source, relation, target)，chunk 引用累积
    entities: Dict[str, Tuple[str, str]] = {}
    relations: Dict[Tuple[str, str, str], Tuple[str, set]] = {}
    for (text, meta), result in zip(chunks, results):
        cid = meta.get("chunk_id", "")
        for e in result.entities:
            entities.setdefault(e.name, (e.entity_type, e.description))
        for r in result.relations:
            key = (r.source, r.relation, r.target)
            entry = relations.setdefault(key, (r.description, set()))
            if cid:
                entry[1].add(cid)
            # 关系端点也补进实体（防止端点在抽取结果中未作为实体输出而丢关系）
            entities.setdefault(r.source, ("concept", ""))
            entities.setdefault(r.target, ("concept", ""))

    client.upsert_entities_batch(kb_id, entities)
    client.upsert_relations_batch(
        kb_id,
        {k: (desc, sorted(ids)) for k, (desc, ids) in relations.items()},
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


def _sanitize_graph_embedding_text(text: str) -> str:
    """去除控制字符并统一 Unicode，降低本地 embedding 产生 NaN 的概率。"""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    printable = "".join(
        character if character.isprintable() else " "
        for character in normalized
    )
    return " ".join(printable.split())[:500]


def _validate_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    expected: int,
) -> List[List[float]]:
    if len(vectors) != expected:
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {expected} texts"
        )
    validated: List[List[float]] = []
    for vector in vectors:
        values = [float(value) for value in vector]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("embedder returned an empty or non-finite vector")
        validated.append(values)
    return validated


def _embed_graph_items_with_fallback(
    items: Sequence[Dict[str, Any]],
    texts: Sequence[str],
    embedder: Any,
    *,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """批量向量化；失败时二分定位，只跳过无法恢复的单条脏数据。"""
    kept_items: List[Dict[str, Any]] = []
    kept_vectors: List[List[float]] = []

    def embed_batch(
        batch_items: Sequence[Dict[str, Any]],
        batch_texts: Sequence[str],
    ) -> None:
        try:
            vectors = _validate_embedding_vectors(
                embedder.embed_texts(list(batch_texts)),
                len(batch_texts),
            )
        except Exception as exc:
            if len(batch_items) > 1:
                midpoint = len(batch_items) // 2
                embed_batch(batch_items[:midpoint], batch_texts[:midpoint])
                embed_batch(batch_items[midpoint:], batch_texts[midpoint:])
                return

            item = batch_items[0]
            original_text = batch_texts[0]
            sanitized_text = _sanitize_graph_embedding_text(original_text)
            if sanitized_text and sanitized_text != original_text:
                try:
                    vectors = _validate_embedding_vectors(
                        embedder.embed_texts([sanitized_text]),
                        1,
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "[graph_build] skip non-embeddable graph item %s (%s): %s",
                        item.get("id", ""),
                        item.get("kind", ""),
                        retry_exc,
                    )
                    return
                kept_items.append(dict(item))
                kept_vectors.extend(vectors)
                return

            logger.warning(
                "[graph_build] skip non-embeddable graph item %s (%s): %s",
                item.get("id", ""),
                item.get("kind", ""),
                exc,
            )
            return

        kept_items.extend(dict(item) for item in batch_items)
        kept_vectors.extend(vectors)

    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), safe_batch_size):
        embed_batch(
            items[start:start + safe_batch_size],
            texts[start:start + safe_batch_size],
        )
    return kept_items, kept_vectors


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

    # 批量向量化；单条异常不再导致整次构建在最后一步失败。
    embedder = get_embedder()
    indexed_items, vectors = _embed_graph_items_with_fallback(
        items,
        texts,
        embedder,
        batch_size=cfg.GRAPH_BUILD_BATCH_SIZE,
    )
    if not indexed_items:
        logger.warning("[graph_build] no graph items could be embedded for kb %s", kb_id)
        return 0, 0

    index = get_graph_vector_index()
    index.upsert(indexed_items, vectors)
    entities_indexed = sum(
        1 for item in indexed_items if item.get("kind") == "entity"
    )
    relations_indexed = sum(
        1 for item in indexed_items if item.get("kind") == "triple"
    )
    logger.info(
        "[graph_build] kb %s: indexed %d entities + %d triples into Milvus",
        kb_id, entities_indexed, relations_indexed,
    )
    return entities_indexed, relations_indexed


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
