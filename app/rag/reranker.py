"""Reranker 模块 — 交叉编码器精排，显著提升检索质量。

支持三种后端（通过 RERANKER_TYPE 配置）：
  - local          本地加载 HuggingFace cross-encoder（推荐 bge-reranker-v2-m3）
  - openai_compatible  调用 OpenAI 兼容的 /v1/rerank 端点
  - disabled       禁用（默认，行为不变）

LightRAG 实测：启用 reranker 后混合查询性能提升最显著。
延迟代价：本地模型 ~1-2 秒，API 端点 ~200ms。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()


class BaseReranker:
    """抽象基类。"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """对文档列表按与查询的相关性重排序。

        参数
        ----
        query: 原始查询
        documents: 文档文本列表
        top_k: 返回前 K 个（None=全部）

        返回
        ----
        [(原始索引, 相关性分数), ...] 按分数降序排列
        """
        raise NotImplementedError

    def rerank_dicts(
        self,
        query: str,
        docs: List[dict],
        content_key: str = "content",
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """对字典列表精排：为每个 dict 增加 rerank_score 字段，按分降序排列。"""
        texts = [d.get(content_key, "") for d in docs]
        ranked = self.rerank(query, texts, top_k=top_k)
        result = []
        for idx, score in ranked:
            doc = dict(docs[idx])
            doc["rerank_score"] = round(score, 4)
            result.append(doc)
        # 补上被截断的
        included = {idx for idx, _ in ranked}
        for i, d in enumerate(docs):
            if i not in included:
                doc = dict(d)
                doc["rerank_score"] = 0.0
                result.append(doc)
        return result


class LocalCrossEncoderReranker(BaseReranker):
    """本地 HuggingFace cross-encoder 精排器。

    首次加载模型需下载（~2GB for bge-reranker-v2-m3），之后缓存。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = 512,
    ):
        self._model_path = model_path or cfg.RERANKER_MODEL_PATH
        self._max_length = max_length
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            logger.info("[reranker] loading cross-encoder: %s", self._model_path)
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self._model_path,
                    max_length=self._max_length,
                )
                logger.info("[reranker] model loaded, max_length=%d", self._max_length)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load reranker model '{self._model_path}'. "
                    f"Install: pip install sentence-transformers. "
                    f"Error: {exc}"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.error("[reranker] prediction failed: %s", exc)
            # 回退：原顺序
            return [(i, 1.0 - i * 0.001) for i in range(len(documents))]

        # (index, score, text) 三元组
        ranked = sorted(
            enumerate(zip(scores, documents)),
            key=lambda x: x[1][0],
            reverse=True,
        )

        top = ranked[:top_k] if top_k else ranked
        return [(idx, float(score)) for idx, (score, _) in top]


class OpenAIReranker(BaseReranker):
    """调用 OpenAI 兼容的 /v1/rerank 端点。

    支持: Cohere, Jina AI, Voyage AI, 或其他兼容的 rerank API。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        import requests
        self._requests = requests
        self._base_url = (base_url or cfg.RERANKER_API_BASE).rstrip("/")
        self._api_key = api_key or cfg.RERANKER_API_KEY
        self._model = model or cfg.RERANKER_MODEL_NAME

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_k:
            payload["top_n"] = top_k

        try:
            resp = self._requests.post(
                f"{self._base_url}/v1/rerank",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("[reranker] API call failed: %s", exc)
            return [(i, 1.0 - i * 0.001) for i in range(len(documents))]

        results = data.get("results", [])
        ranked = []
        for r in results:
            idx = r.get("index", 0)
            score = r.get("relevance_score", 0.0)
            ranked.append((idx, float(score)))
        return ranked


# ── 工厂 ──────────────────────────────────────────────────────────────────────

_reranker: Optional[BaseReranker] = None


def get_reranker() -> BaseReranker:
    global _reranker
    if _reranker is not None:
        return _reranker

    rtype = cfg.RERANKER_TYPE
    if rtype == "local":
        _reranker = LocalCrossEncoderReranker()
    elif rtype == "openai_compatible":
        _reranker = OpenAIReranker()
    else:
        # disabled: 返回一个 no-op reranker
        _reranker = _NoOpReranker()

    return _reranker


class _NoOpReranker(BaseReranker):
    """禁用精排时的占位实现，保持原顺序。"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        n = len(documents)
        result = [(i, 1.0 - i * 0.0001) for i in range(n)]
        return result[:top_k] if top_k else result
