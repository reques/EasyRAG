"""知识图谱服务（阶段 2C）— 实体/关系抽取 + 图谱查询。

设计取舍：
- 存储用 PostgreSQL 两张表（entities/relations），不引入 Neo4j — 阶段 2 先验证价值，
  子图查询用 SQL 递归即可覆盖 1-2 跳邻居场景；
- 抽取走 LLM JSON 模式，逐 chunk 调用，失败 chunk 跳过不阻塞上传主链路；
- GRAPH_ENABLED=False 时一切旁路，上传与检索行为与阶段 1 完全一致。
"""
from __future__ import annotations

import asyncio
import json
import re
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

_EXTRACT_PROMPT = """从下面的文本中抽取知识图谱元素（实体与关系），支持中英文。

实体类型（选择最贴切的一类，都不贴切用 Other）：
- person（人物）/ organization（机构）/ location（地点）/ event（事件）
- concept（抽象概念）/ method（方法·流程）/ artifact（产品·工具）/ data（数据·指标）

要求：
1. 实体：只抽取明确、有意义的实体。对每个实体给出：
   - name：实体名，用原文表述，全文保持命名一致（同一实体不要用不同名字）
   - type：上述类型之一
   - description：一句话描述实体的属性/作用（仅基于文本，不臆造）
2. 关系：只抽取实体之间明确、直接的关系。对每个关系给出：
   - source：源实体名（须与实体 name 一致）
   - target：目标实体名（须与实体 name 一致）
   - relation：关系类型（如"属于"/"使用"/"导致"/"依赖"/"对比"等）
   - description：一句话说明关系本质（可含检索关键词）
3. 只抽取文本中明确表达的信息，不要臆造；没有可抽取内容时返回空数组。
4. 输出语言与输入文本一致（中文文本输出中文，英文文本输出英文；专有名词保留原文）。

严格输出 JSON（不要输出其他内容）：
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]}}

文本：
{chunk}"""


# ── 法律条文规则抽取 ─────────────────────────────────────────────────────────

_ARTICLE_RE = re.compile(r"第([零一二三四五六七八九十百千〇0-9]+)条")
_SECTION_PREFIX_RE = re.compile(r"\[([^\]]+)\]")
_TITLE_RE = re.compile(r"【([^】]+)】")
_REFERENCE_RE = re.compile(r"(?:依据|依照|按照|根据)本法第([零一二三四五六七八九十百千〇0-9]+)条")


def _looks_like_legal_chunks(chunks: List[tuple], threshold: int = 3) -> bool:
    """检测 chunks 是否为法律条文（前 10 个 chunk 含多个「第X条」）。"""
    hits = 0
    for text, _ in chunks[:10]:
        hits += len(_ARTICLE_RE.findall(text))
        if hits >= threshold:
            return True
    return False


async def _extract_graph_rule_based(
    session: AsyncSession,
    kb_id: uuid.UUID,
    chunks: List[tuple],
    source_name: str,
    progress_callback: Optional[
        Callable[[int, int, str], Awaitable[None]]
    ] = None,
    existing_entities: Optional[set] = None,
    existing_relations: Optional[set] = None,
) -> Dict[str, int]:
    """规则抽取法律条文图谱：条文/章节实体 + 归属/引用关系（毫秒级、免费）。

    法律条文结构固定（第X条【标题】/ 第X章 标题 / 依据本法第X条），正则即可
    精确抽取，无需 LLM —— LLM 逐条抽又慢又贵，对法律文本反而不如规则准。
    """
    total_entities = 0
    total_relations = 0
    # 库级 + 批内去重（实体按 (name, source_file)，关系按 (source, type, target, source_file)）
    seen_entities: set = set(existing_entities or ())
    seen_relations: set = set(existing_relations or ())

    total = len(chunks)
    if progress_callback:
        await progress_callback(0, total, f"正在抽取知识图谱 0/{total}")

    for i, (text, meta) in enumerate(chunks):
        chunk_ref = f"{source_name}#{meta.get('chunk_index', i)}"

        # 章节（split_legal 的 [章节] 前缀）
        section = ""
        m_section = _SECTION_PREFIX_RE.search(text)
        if m_section:
            section = m_section.group(1).strip()

        # 条文编号 + 标题
        m_article = _ARTICLE_RE.search(text)
        if not m_article:
            continue
        article_no = "第" + m_article.group(1) + "条"
        title = ""
        m_title = _TITLE_RE.search(text)
        if m_title:
            title = m_title.group(1).strip()
        article_name = f"{article_no}【{title}】" if title else article_no

        # 条文实体（description 含标题 + 正文，通用：让描述匹配与向量重排能命中
        # 正文而非仅标题，不针对任何特定领域）
        body = _SECTION_PREFIX_RE.sub(
            "", _TITLE_RE.sub("", _ARTICLE_RE.sub("", text, count=1), count=1)
        ).strip()
        article_desc = (f"{title}；{body[:150]}" if title else body[:150]).strip()[:1024]

        ent_key = (article_name, source_name)
        if ent_key not in seen_entities:
            seen_entities.add(ent_key)
            session.add(KnowledgeEntity(
                knowledge_base_id=kb_id,
                name=article_name,
                entity_type="article",
                description=article_desc,
                source_chunks=chunk_ref,
                source_file=source_name,
            ))
            graph_cache.upsert_entity(
                name=article_name,
                entity_type="article",
                description=article_desc,
                kb_id=str(kb_id),
                source_file=source_name,
            )
            total_entities += 1

        # 章节实体 + 归属关系
        if section:
            sec_key = (section, source_name)
            if sec_key not in seen_entities:
                seen_entities.add(sec_key)
                session.add(KnowledgeEntity(
                    knowledge_base_id=kb_id,
                    name=section,
                    entity_type="chapter",
                    description="",
                    source_chunks=chunk_ref,
                    source_file=source_name,
                ))
                graph_cache.upsert_entity(
                    name=section,
                    entity_type="chapter",
                    description="",
                    kb_id=str(kb_id),
                    source_file=source_name,
                )
                total_entities += 1

            rel_key = (article_name, "属于", section, source_name)
            if rel_key not in seen_relations:
                seen_relations.add(rel_key)
                session.add(KnowledgeRelation(
                    knowledge_base_id=kb_id,
                    source_entity=article_name,
                    target_entity=section,
                    relation_type="属于",
                    description="",
                    source_file=source_name,
                ))
                graph_cache.add_relation(
                    source=article_name,
                    target=section,
                    relation_type="属于",
                    description="",
                    kb_id=str(kb_id),
                    source_file=source_name,
                )
                total_relations += 1

        # 引用关系（依据本法第X条）
        for ref_no in _REFERENCE_RE.findall(text):
            ref = "第" + ref_no + "条"
            if ref == article_no:
                continue
            rel_key = (article_name, "引用", ref, source_name)
            if rel_key not in seen_relations:
                seen_relations.add(rel_key)
                session.add(KnowledgeRelation(
                    knowledge_base_id=kb_id,
                    source_entity=article_name,
                    target_entity=ref,
                    relation_type="引用",
                    description="",
                    source_file=source_name,
                ))
                graph_cache.add_relation(
                    source=article_name,
                    target=ref,
                    relation_type="引用",
                    description="",
                    kb_id=str(kb_id),
                    source_file=source_name,
                )
                total_relations += 1

    if progress_callback:
        await progress_callback(total, total, f"正在抽取知识图谱 {total}/{total}")

    logger.info(
        "[graph] rule-based '%s' -> %d entities, %d relations (from %d chunks)",
        source_name, total_entities, total_relations, total,
    )
    return {"entities": total_entities, "relations": total_relations}


# ── 中英文通用实体抽取（NER 层）────────────────────────────────────────────

# 中文 jieba 词性 → LightRAG 风格通用实体类型
# 注意：不含 "eng"（英文单词）——英文实体由 title case / 缩写正则抽取，
# 否则 jieba 会把 uses/and/for 等英文停用词也标成名词，淹没真正的英文短语。
_CN_POS_TYPE = {
    "nr": "person",        # 人名
    "nrt": "person",       # 音译人名
    "ns": "location",      # 地名
    "nt": "organization",  # 机构
    "nz": "proper_noun",   # 其他专名
    "n": "concept",        # 普通名词
    "vn": "concept",       # 动名词
    "an": "concept",       # 名形词
    "ng": "concept",       # 名语素
}
_CN_NOUN_FLAGS = set(_CN_POS_TYPE.keys())

# 英文：title case 多词短语（专有名词，如 "Machine Learning"、"New York"）
_EN_TITLE_CASE_RE = re.compile(
    r"(?<![\w-])(?:[A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)(?![\w-])"
)
# 英文：全大写缩写（如 "API"、"GPU"、"CNN"）
_EN_ACRONYM_RE = re.compile(r"(?<![\w-])(?:[A-Z]{2,})(?![\w-])")

_STOPWORDS = {
    "我们", "你们", "他们", "这个", "那个", "这些", "那些", "什么", "怎么", "为什么",
    "可以", "应该", "需要", "进行", "通过", "以及", "或者", "但是", "因为", "所以",
    "如果", "然后", "就是", "一个", "一种", "没有", "不是", "自己", "对于", "关于",
    "本文", "本章", "如下", "上述", "根据", "规定", "内容", "情况", "问题", "方面",
    "时候", "东西", "部分", "主要", "其中", "其他", "有关", "相关", "本法", "本条",
}

_jieba_pseg = None


def _get_pseg():
    """lazy load jieba.posseg（首次调用约 1-2 秒初始化词典）。"""
    global _jieba_pseg
    if _jieba_pseg is None:
        try:
            import jieba.posseg as pseg
            _jieba_pseg = pseg
        except ImportError:
            _jieba_pseg = False
    return _jieba_pseg


def _extract_entities_generic(text: str, max_entities: int = 10):
    """中英文通用实体抽取：返回 [(name, type, description)]。

    中文走 jieba 词性标注（合并相邻名词为短语），英文走正则（title case
    短语 + 大写缩写）。实体类型复用 LightRAG 的通用分类，不依赖任何特定
    文本结构（如法律条文的「第X条」），适用于整篇自由文本。
    """
    entities = []

    # 中文：jieba 词性标注
    pseg = _get_pseg()
    if pseg:
        buf = []
        buf_type = "concept"
        for word, flag in pseg.cut(text[:2000]):
            if flag in _CN_NOUN_FLAGS:
                if not buf:
                    buf_type = _CN_POS_TYPE.get(flag, "concept")
                buf.append(word)
            else:
                if buf:
                    name = "".join(buf).strip()
                    if len(name) >= 2 and name not in _STOPWORDS:
                        entities.append((name, buf_type, ""))
                    buf = []
                    buf_type = "concept"
        if buf:
            name = "".join(buf).strip()
            if len(name) >= 2 and name not in _STOPWORDS:
                entities.append((name, buf_type, ""))

    # 英文：title case 多词短语 + 全大写缩写
    for m in _EN_TITLE_CASE_RE.finditer(text):
        entities.append((m.group(0), "proper_noun", ""))
    for m in _EN_ACRONYM_RE.finditer(text):
        entities.append((m.group(0), "organization", ""))

    # 去重（大小写不敏感、保序）+ 过滤纯数字/标点 + 限数
    result = []
    seen = set()
    for name, etype, desc in entities:
        key = name.lower()
        if key in seen:
            continue
        if name.isdigit() or all(ch in "0123456789.%×+-/＿_" for ch in name):
            continue
        seen.add(key)
        result.append((name, etype, desc))
        if len(result) >= max_entities:
            break
    return result


async def _extract_graph_generic(
    session: AsyncSession,
    kb_id: uuid.UUID,
    chunks: List[tuple],
    source_name: str,
    progress_callback: Optional[
        Callable[[int, int, str], Awaitable[None]]
    ] = None,
    existing_entities: Optional[set] = None,
    existing_relations: Optional[set] = None,
) -> Dict[str, int]:
    """通用 NER 抽取（中英文）：jieba/正则抽实体 + 同 chunk 共现关系。

    不依赖特定文本结构，适用于整篇自由文本；实体带类型（person/organization/
    concept/...）。jieba 不可用且抽不到英文实体时返回空，由调用方降级到 LLM。
    """
    total_entities = 0
    total_relations = 0
    # 库级 + 批内去重（实体按 (name, source_file)，关系按 (source, type, target, source_file)）
    seen_entities = set(existing_entities or ())
    seen_relations = set(existing_relations or ())

    total = len(chunks)
    for i, (text, meta) in enumerate(chunks):
        # before chunk：报告当前进度
        if progress_callback:
            await progress_callback(i, total, f"正在抽取知识图谱 {i}/{total}")

        chunk_ref = f"{source_name}#{meta.get('chunk_index', i)}"
        entities = _extract_entities_generic(text, max_entities=10)
        if not entities:
            # after chunk（无实体可抽）
            if progress_callback:
                await progress_callback(i + 1, total, f"正在抽取知识图谱 {i + 1}/{total}")
            continue

        # 实体入库
        for name, etype, desc in entities[:6]:
            ent_key = (name, source_name)
            if ent_key in seen_entities:
                continue
            seen_entities.add(ent_key)
            session.add(KnowledgeEntity(
                knowledge_base_id=kb_id,
                name=name,
                entity_type=etype,
                description=desc,
                source_chunks=chunk_ref,
                source_file=source_name,
            ))
            graph_cache.upsert_entity(
                name=name,
                entity_type=etype,
                description=desc,
                kb_id=str(kb_id),
                source_file=source_name,
            )
            total_entities += 1

        # 共现关系：同 chunk 内 top 实体两两「相关」
        top = [e[0] for e in entities[:4]]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                src, tgt = top[a], top[b]
                rel_key = (src, "相关", tgt, source_name)
                if rel_key in seen_relations:
                    continue
                seen_relations.add(rel_key)
                session.add(KnowledgeRelation(
                    knowledge_base_id=kb_id,
                    source_entity=src,
                    target_entity=tgt,
                    relation_type="相关",
                    description="",
                    source_file=source_name,
                ))
                graph_cache.add_relation(
                    source=src,
                    target=tgt,
                    relation_type="相关",
                    description="",
                    kb_id=str(kb_id),
                    source_file=source_name,
                )
                total_relations += 1

        # after chunk
        if progress_callback:
            await progress_callback(i + 1, total, f"正在抽取知识图谱 {i + 1}/{total}")

    logger.info(
        "[graph] generic '%s' -> %d entities, %d relations (from %d chunks)",
        source_name, total_entities, total_relations, total,
    )
    return {"entities": total_entities, "relations": total_relations}


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

    双层抽取（通用，不区分文档类型）：
    1. LLM 语义抽取 —— 概念实体（person/organization/concept/...）+ 语义关系
       （属于/使用/导致/依赖/...），这是「实体=概念」的主体，用于图谱可视化；
    2. 结构抽取 —— 与文档结构对应的实体（如章节/条目编号），其 description 含
       正文，用于检索命中正文（不针对特定领域，是通用结构归属）。
    两者结果合并入库，互不覆盖。
    """
    # 库级去重（2026-08-27）：查该知识库已存在的实体/关系 key，防止同文件
    # 重复处理/多次抽取时插入重复行（与图谱查询接口的去重口径一致：
    # 实体=(name, source_file)，关系=(source, relation_type, target, source_file)）。
    from sqlalchemy import select as _select

    existing_entities = set()
    existing_relations = set()
    rows = await session.execute(
        _select(KnowledgeEntity.name, KnowledgeEntity.source_file).where(
            KnowledgeEntity.knowledge_base_id == kb_id
        )
    )
    for name, sf in rows:
        existing_entities.add((name, sf or ""))
    rows = await session.execute(
        _select(
            KnowledgeRelation.source_entity,
            KnowledgeRelation.relation_type,
            KnowledgeRelation.target_entity,
            KnowledgeRelation.source_file,
        ).where(KnowledgeRelation.knowledge_base_id == kb_id)
    )
    for s, rel, t, sf in rows:
        existing_relations.add((s, rel, t, sf or ""))

    llm_result = await _extract_graph_llm(
        session, kb_id, chunks, source_name, progress_callback,
        existing_entities, existing_relations,
    )
    if _looks_like_legal_chunks(chunks):
        structural = await _extract_graph_rule_based(
            session, kb_id, chunks, source_name, progress_callback,
            existing_entities, existing_relations,
        )
    else:
        structural = await _extract_graph_generic(
            session, kb_id, chunks, source_name, progress_callback,
            existing_entities, existing_relations,
        )
    return {
        "entities": llm_result["entities"] + structural["entities"],
        "relations": llm_result["relations"] + structural["relations"],
    }


async def _extract_graph_llm(
    session: AsyncSession,
    kb_id: uuid.UUID,
    chunks: List[tuple],
    source_name: str,
    progress_callback: Optional[
        Callable[[int, int, str], Awaitable[None]]
    ] = None,
    existing_entities: Optional[set] = None,
    existing_relations: Optional[set] = None,
) -> Dict[str, int]:
    """LLM 并发抽取：入库串行（AsyncSession 非并发安全），单 chunk 失败只记日志。"""
    from app.llm.client import get_llm_client

    llm = get_llm_client()
    sampled = chunks[: cfg.GRAPH_MAX_CHUNKS_PER_FILE]
    # 过滤太短的 chunk（没有抽取价值）
    valid = [
        (i, text, meta)
        for i, (text, meta) in enumerate(sampled)
        if len(text.strip()) >= 50
    ]
    total = len(valid)
    if not valid:
        logger.info("[graph] '%s' -> 0 chunks worth extracting", source_name)
        return {"entities": 0, "relations": 0}

    concurrency = max(1, cfg.GRAPH_LLM_CONCURRENCY)
    sem = asyncio.Semaphore(concurrency)

    async def extract_one(idx: int, text: str):
        async with sem:
            try:
                return idx, await llm.chat_json(
                    [{"role": "user", "content": _EXTRACT_PROMPT.format(chunk=text[:2000])}]
                )
            except Exception as exc:
                logger.warning(
                    "[graph] extract failed for chunk %d of '%s': %s", idx, source_name, exc
                )
                return idx, {}

    # 并发调 LLM，按完成顺序推进进度
    done = 0
    extracted: Dict[int, dict] = {}
    if progress_callback:
        await progress_callback(0, total, f"正在抽取知识图谱 0/{total}")
    coros = [extract_one(i, text) for i, text, _ in valid]
    for coro in asyncio.as_completed(coros):
        idx, raw = await coro
        extracted[idx] = raw
        done += 1
        if progress_callback:
            await progress_callback(done, total, f"正在抽取知识图谱 {done}/{total}")

    # 串行入库（AsyncSession 非并发安全）
    total_entities = 0
    total_relations = 0
    # 库级 + 批内去重（实体 (name, source_file)；关系 (source, type, target, source_file)）
    seen_entities: set = set(existing_entities or ())
    seen_relations: set = set(existing_relations or ())
    for i, text, meta in valid:
        raw = extracted.get(i) or {}
        entities = raw.get("entities") or []
        relations = raw.get("relations") or []
        chunk_ref = f"{source_name}#{meta.get('chunk_index', i)}"

        for e in entities[:20]:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            ent_type = (e.get("type") or "concept")[:64]
            if (name, source_name) in seen_entities:
                continue
            seen_entities.add((name, source_name))
            session.add(KnowledgeEntity(
                knowledge_base_id=kb_id,
                name=name,
                entity_type=ent_type,
                description=(e.get("description") or "")[:1024],
                source_chunks=chunk_ref,
                source_file=source_name,
            ))
            graph_cache.upsert_entity(
                name=name,
                entity_type=(e.get("type") or "concept"),
                description=(e.get("description") or ""),
                kb_id=str(kb_id),
                source_file=source_name,
            )
            total_entities += 1

        for r in relations[:20]:
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            rel = (r.get("relation") or "").strip()
            if not (src and tgt and rel):
                continue
            rel_key = (src, rel, tgt, source_name)
            if rel_key in seen_relations:
                continue
            seen_relations.add(rel_key)
            session.add(KnowledgeRelation(
                knowledge_base_id=kb_id,
                source_entity=src,
                target_entity=tgt,
                relation_type=rel[:128],
                description=(r.get("description") or "")[:1024],
                source_file=source_name,
            ))
            graph_cache.add_relation(
                source=src,
                target=tgt,
                relation_type=rel,
                description=(r.get("description") or ""),
                kb_id=str(kb_id),
                source_file=source_name,
            )
            total_relations += 1

    logger.info(
        "[graph] '%s' -> %d entities, %d relations (from %d chunks)",
        source_name, total_entities, total_relations, total,
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
