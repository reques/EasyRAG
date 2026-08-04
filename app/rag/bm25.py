"""轻量级 BM25 稀疏检索 — 无外部依赖的全文精确匹配。

用于增强检索 Path D：补齐向量检索在数字、代码、专有名词等精确匹配场景的盲区。

实现基于 Okapi BM25 标准公式，支持中文分词（jieba 可选）和英文空格分词。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── 分词 ──────────────────────────────────────────────────────────────────────

# 中文分词器（延迟导入，不强制依赖）
_jieba_loaded = False
_jieba_cut = None


def _ensure_jieba():
    global _jieba_loaded, _jieba_cut
    if _jieba_loaded:
        return _jieba_cut
    try:
        import jieba
        _jieba_cut = jieba.cut
        _jieba_loaded = True
        logger.info("[bm25] jieba loaded for Chinese tokenization")
    except ImportError:
        # fallback: 2-gram 字符级切分
        _jieba_loaded = True
        _jieba_cut = None
        logger.info("[bm25] jieba not available, using char 2-gram fallback")
    return _jieba_cut


def tokenize(text: str) -> List[str]:
    """智能分词：中文用 jieba（不可用时用 2-gram 回退），英文/数字按空格拆分。

    保留 2 字符以上的 token，去除纯标点。
    """
    cut = _ensure_jieba()
    if cut is not None:
        tokens = list(cut(text))
    else:
        # 2-gram fallback for Chinese
        tokens = []
        chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        for segment in chinese:
            for i in range(len(segment) - 1):
                tokens.append(segment[i : i + 2])
        # 英文/数字
        mixed = re.sub(r"[\u4e00-\u9fff]+", " ", text)
        tokens.extend(mixed.lower().split())

    # 过滤：至少 2 字符，非纯标点
    return [t for t in tokens if len(t) >= 2 and not re.fullmatch(r"[\W_]+", t)]


# ── BM25 实现 ─────────────────────────────────────────────────────────────────


class BM25Retriever:
    """Okapi BM25 稀疏检索引擎。

    用法::

        bm25 = BM25Retriever()
        bm25.index(documents)  # documents: [{"id": ..., "content": ..., "metadata": {...}}]
        results = bm25.search("查询", top_k=5)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Tuple[str, str, dict]] = []  # (doc_id, content, metadata)
        self._tokenized: List[List[str]] = []          # doc → tokens
        self._doc_len: List[int] = []                   # doc → token count
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}                # token → IDF
        self._inverted: Dict[str, Dict[int, int]] = defaultdict(dict)  # token → {doc_index: tf}

        # 英文停用词（高频无意义词）
        self._stopwords: set = {
            "the", "is", "at", "of", "on", "and", "a", "an", "to", "in", "for",
            "it", "or", "be", "as", "by", "we", "he", "she", "they", "that",
            "this", "are", "was", "were", "been", "has", "have", "had", "not",
            "but", "from", "with", "about", "which", "can", "will", "would",
            "could", "should", "may", "do", "does", "did", "what", "when",
            "where", "who", "how", "all", "each", "its",
        }

    # ── 索引构建 ───────────────────────────────────────────────────────────

    def index(self, documents: List[dict]):
        """批量索引文档。每个文档为 {"id": str, "content": str, "metadata": dict}。"""
        self._docs.clear()
        self._tokenized.clear()
        self._doc_len.clear()
        self._idf.clear()
        self._inverted.clear()

        total_len = 0
        for doc in documents:
            doc_id = doc.get("id", str(len(self._docs)))
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            tokens = [t for t in tokenize(content) if t.lower() not in self._stopwords]

            self._docs.append((doc_id, content, metadata))
            self._tokenized.append(tokens)
            self._doc_len.append(len(tokens))
            total_len += len(tokens)

            # 构建倒排索引
            for token in set(tokens):
                tf = tokens.count(token)
                self._inverted[token][len(self._docs) - 1] = tf

        n = len(self._docs)
        self._avgdl = total_len / max(n, 1)

        # 计算 IDF
        for token, postings in self._inverted.items():
            df = len(postings)
            self._idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        logger.info(
            "[bm25] indexed %d docs, avg length=%.1f, vocab size=%d",
            n, self._avgdl, len(self._idf),
        )

    def add(self, doc_id: str, content: str, metadata: dict = None):
        """追加单个文档（增量索引）。"""
        tokens = [t for t in tokenize(content) if t.lower() not in self._stopwords]
        idx = len(self._docs)
        self._docs.append((doc_id, content, metadata or {}))
        self._tokenized.append(tokens)
        self._doc_len.append(len(tokens))

        for token in set(tokens):
            tf = tokens.count(token)
            self._inverted[token][idx] = tf

        # 更新 IDF（近似：仅更新出现过的 token）
        n = len(self._docs)
        self._avgdl = (self._avgdl * (n - 1) + len(tokens)) / n
        for token in set(tokens):
            df = len(self._inverted.get(token, {}))
            self._idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def remove(self, doc_id: str):
        """按 doc_id 删除文档（标记删除，下次 index 时清理）。"""
        for i, (did, _, _) in enumerate(self._docs):
            if did == doc_id:
                self._docs[i] = ("__DELETED__", "", {})
                break

    # ── 检索 ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[dict]:
        """BM25 检索，返回 [{"id":..., "content":..., "metadata":..., "score":...}]。"""
        if not self._docs:
            return []

        query_tokens = [t for t in tokenize(query) if t.lower() not in self._stopwords]
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for token in set(query_tokens):
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            for doc_idx, tf in self._inverted.get(token, {}).items():
                if self._docs[doc_idx][0] == "__DELETED__":
                    continue
                dl = max(self._doc_len[doc_idx], 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] += idf * numerator / denominator

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in ranked[:top_k]:
            if score < score_threshold:
                break
            doc_id, content, metadata = self._docs[doc_idx]
            if doc_id == "__DELETED__":
                continue
            results.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "score": round(score, 4),
                "retrieval_path": "bm25",
            })

        logger.info("[bm25] query=%r returned %d docs", query[:60], len(results))
        return results

    @property
    def doc_count(self) -> int:
        return sum(1 for d in self._docs if d[0] != "__DELETED__")


# ── 单例 ──────────────────────────────────────────────────────────────────────

_bm25: Optional[BM25Retriever] = None


def get_bm25() -> BM25Retriever:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Retriever()
    return _bm25
