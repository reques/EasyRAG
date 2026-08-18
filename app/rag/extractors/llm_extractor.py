"""LLM JSON 模式抽取器（默认实现）。

复用阶段 2C 的上传链路抽取 prompt，抽取出实体与关系后交给
构建服务写入 Neo4j / PostgreSQL / Milvus。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.rag.extractors.base import (
    EntityExtraction,
    ExtractionResult,
    GraphExtractor,
    RelationExtraction,
)

logger = get_logger(__name__)

# 与 graph_service（上传链路）共用的抽取 prompt，保持单一来源
EXTRACT_PROMPT = """从下面的文本中抽取知识图谱元素（实体与关系）。

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


class LLMExtractor(GraphExtractor):
    """通过 LLM JSON 模式逐 chunk 抽取实体/关系。"""

    name = "llm"

    def __init__(self, llm=None, max_chars: int = 2000):
        self._llm = llm
        self._max_chars = max_chars

    def _get_llm(self):
        if self._llm is None:
            from app.llm.client import get_llm_client

            self._llm = get_llm_client()
        return self._llm

    async def extract(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        llm = self._get_llm()
        raw = await llm.chat_json([
            {"role": "user", "content": EXTRACT_PROMPT.format(chunk=text[: self._max_chars])}
        ])
        data = raw if isinstance(raw, dict) else {}
        result = ExtractionResult()
        for e in (data.get("entities") or [])[:20]:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            result.entities.append(EntityExtraction(
                name=name[:256],
                entity_type=(e.get("type") or "concept")[:64],
                description=(e.get("description") or "")[:1024],
            ))
        for r in (data.get("relations") or [])[:20]:
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            rel = (r.get("relation") or "").strip()
            if not (src and tgt and rel):
                continue
            result.relations.append(RelationExtraction(
                source=src[:256],
                target=tgt[:256],
                relation=rel[:128],
                description=(r.get("description") or "")[:1024],
            ))
        return result


def get_extractor(name: str = "llm", **kwargs) -> GraphExtractor:
    """抽取器工厂：按名称实例化，未知名称回退到 LLM 抽取器。"""
    if name == "llm":
        return LLMExtractor(**kwargs)
    logger.warning("[extractor] unknown extractor '%s', fallback to 'llm'", name)
    return LLMExtractor(**kwargs)
