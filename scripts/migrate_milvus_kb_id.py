"""重建 Milvus 索引并回填 knowledge_base_id（方案 B 一次性迁移）。

背景:
    旧版 rag_docs collection 只有 id/content/source/vector 四字段,没有
    knowledge_base_id,导致检索结果无法关联到 knowledge_files 记录、前端
    引用无法跳转到文档详情。

做法:
    1. 触发 MilvusRetriever 初始化 — 检测到旧 schema 会自动 drop 并重建
       带 knowledge_base_id 的新 collection(见 retriever.py __init__)。
    2. 遍历 PostgreSQL knowledge_files 中 status=completed 且有 text_content
       的文件,用当前生效的 CHUNK_STRATEGY 对 text_content 重新分块。
    3. 逐文件调 add_documents,metadata 显式写入 knowledge_base_id / source。

幂等性:
    非幂等 — 重复运行会重复插入同一文件的 chunk。运行前请确认 collection
    已被重建(首次初始化会自动完成),或先手动 drop。

用法:
    D:/Anaconda3/envs/stage1-agent/python.exe scripts/migrate_milvus_kb_id.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 允许以脚本方式直接运行:把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.chunker import (
    split_recursive,
    split_text,
    split_markdown,
    split_parent_child,
)
from app.rag.retriever import get_retriever

logger = get_logger(__name__)
cfg = get_settings()


def _chunk_text(full_text: str, filename: str) -> list[tuple[str, dict]]:
    """按当前生效策略对纯文本分块,复用 chunker 的各策略实现。"""
    strategy = cfg.CHUNK_STRATEGY
    base_meta = {"source": filename, "strategy": strategy}

    if strategy == "fixed":
        raw = split_text(full_text)
        return [(c, {**base_meta, "chunk_index": i}) for i, c in enumerate(raw)]
    if strategy == "recursive":
        raw = split_recursive(full_text)
        return [(c, {**base_meta, "chunk_index": i}) for i, c in enumerate(raw)]
    if strategy == "markdown":
        raw = split_markdown(full_text)
        out = []
        for i, c in enumerate(raw):
            section = ""
            if c.startswith("[") and "]\n" in c:
                section = c[1:c.index("]\n")]
            out.append((c, {**base_meta, "chunk_index": i, "section_path": section}))
        return out
    if strategy == "parent_child":
        pairs = split_parent_child(full_text)
        return [
            (child, {**base_meta, "chunk_index": i, "parent_text": parent})
            for i, (child, parent) in enumerate(pairs)
        ]
    raise ValueError(f"Unknown chunk strategy '{strategy}'")


async def main() -> int:
    from backend.storage.postgres.manager import get_session
    from backend.storage.postgres.models_knowledge import KnowledgeFile
    from sqlalchemy import select

    # 1. 初始化 retriever — 若 schema 过旧会自动 drop + 重建
    retriever = get_retriever()
    logger.info("[migrate] retriever ready, strategy=%s", cfg.CHUNK_STRATEGY)

    # 2. 读取所有已完成且有文本的文件
    async with get_session() as session:
        files = (
            await session.execute(
                select(KnowledgeFile).where(KnowledgeFile.status == "completed")
            )
        ).scalars().all()

    if not files:
        logger.warning("[migrate] no completed files found, nothing to do")
        return 0

    total_chunks = 0
    for f in files:
        text = (f.text_content or "").strip()
        if not text:
            logger.warning("[migrate] skip '%s' (empty text_content)", f.filename)
            continue

        chunks = _chunk_text(text, f.filename)
        kb_id = str(f.knowledge_base_id)
        texts = [c[0] for c in chunks]
        metas = []
        for c in chunks:
            m = dict(c[1])
            m["knowledge_base_id"] = kb_id
            metas.append(m)

        n = retriever.add_documents(texts, metas)
        total_chunks += n
        logger.info(
            "[migrate] '%s' -> %d chunks indexed (kb=%s)", f.filename, n, kb_id[:8]
        )

        # 同步 chunk_count,避免与向量库不一致
        if f.chunk_count != n:
            async with get_session() as session:
                from backend.storage.postgres.models_knowledge import KnowledgeFile as KF
                db_f = await session.get(KF, f.id)
                if db_f:
                    db_f.chunk_count = n
                    await session.commit()

    logger.info("[migrate] DONE — %d files, %d chunks total", len(files), total_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
