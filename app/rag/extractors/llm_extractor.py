"""LLM JSON 模式抽取器（默认实现）。

复用阶段 2C 的上传链路抽取 prompt，抽取出实体与关系后交给
构建服务写入 Neo4j / PostgreSQL / Milvus。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import get_settings
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

EXTRACT_PROMPT_VERSION = "graph-packed-v1"

PACKED_EXTRACT_PROMPT = """从输入 JSON 的每个文本片段中独立抽取知识图谱元素。

要求：
1. 每个 item 必须原样返回 id；禁止跨 item 建立关系。
2. 实体是文本中明确出现的名词性概念，包含 name、type、简短 description。
3. 关系包含 source、target、relation、简短 description，只保留文本明确表达的信息。
4. 每个 item 最多 20 个实体、20 条关系；没有内容时返回空数组。
5. 严格输出 JSON，不要输出解释或 Markdown：
{{"items":[{{"id":"0","entities":[{{"name":"...","type":"concept","description":"..."}}],"relations":[{{"source":"...","target":"...","relation":"...","description":"..."}}]}}]}}

输入 JSON：
<items_json>{items_json}</items_json>"""


class LLMExtractor(GraphExtractor):
    """通过 LLM JSON 模式逐 chunk 抽取实体/关系。"""

    name = "llm"
    prompt_version = EXTRACT_PROMPT_VERSION

    def __init__(
        self,
        llm=None,
        max_chars: int = 2000,
        pack_max_chars: Optional[int] = None,
        pack_max_chunks: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        cfg = get_settings()
        self._llm = llm
        self._max_chars = max(50, int(max_chars))
        self._pack_max_chars = max(
            50,
            int(pack_max_chars or cfg.GRAPH_EXTRACT_PACK_MAX_CHARS),
        )
        self._pack_max_chunks = max(
            1,
            int(pack_max_chunks or cfg.GRAPH_EXTRACT_PACK_MAX_CHUNKS),
        )
        self._max_tokens = max(
            128,
            int(max_tokens or cfg.GRAPH_EXTRACT_MAX_TOKENS),
        )

    def _get_llm(self):
        if self._llm is None:
            from app.llm.client import get_llm_client

            self._llm = get_llm_client()
        return self._llm

    @property
    def model_name(self) -> str:
        return str(getattr(self._get_llm(), "model", "unknown"))

    def cache_input(self, text: str) -> str:
        return (text or "")[: self._max_chars]

    def cache_fingerprint(self) -> str:
        return ":".join((
            self.name,
            self.model_name,
            self.prompt_version,
            str(self._max_chars),
        ))

    @staticmethod
    def _result_from_payload(data: Any) -> ExtractionResult:
        if not isinstance(data, dict):
            return ExtractionResult(cacheable=False)
        return ExtractionResult.from_dict(data)

    async def extract(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        llm = self._get_llm()
        raw = await llm.chat_json([
            {"role": "user", "content": EXTRACT_PROMPT.format(chunk=text[: self._max_chars])}
        ], max_tokens=self._max_tokens, temperature=0)
        return self._result_from_payload(raw)

    def _make_packs(
        self,
        chunks: Sequence[tuple],
        eligible_indices: Sequence[int],
    ) -> List[List[int]]:
        packs: List[List[int]] = []
        current: List[int] = []
        current_chars = 0
        for index in eligible_indices:
            text = self.cache_input(chunks[index][0])
            would_exceed_chars = bool(
                current and current_chars + len(text) > self._pack_max_chars
            )
            if (
                len(current) >= self._pack_max_chunks
                or would_exceed_chars
            ):
                packs.append(current)
                current = []
                current_chars = 0
            current.append(index)
            current_chars += len(text)
        if current:
            packs.append(current)
        return packs

    async def _extract_pack(
        self,
        chunks: Sequence[tuple],
        indices: Sequence[int],
    ) -> Dict[int, ExtractionResult]:
        payload = {
            "items": [
                {"id": str(index), "text": self.cache_input(chunks[index][0])}
                for index in indices
            ]
        }
        raw = await self._get_llm().chat_json(
            [{
                "role": "user",
                "content": PACKED_EXTRACT_PROMPT.format(
                    items_json=json.dumps(payload, ensure_ascii=False)
                ),
            }],
            max_tokens=self._max_tokens,
            temperature=0,
        )
        raw_items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(raw_items, list):
            # 兼容只含一个片段时偶发返回旧版结构的模型。
            if len(indices) == 1 and isinstance(raw, dict):
                return {indices[0]: self._result_from_payload(raw)}
            return {
                index: ExtractionResult(cacheable=False)
                for index in indices
            }

        by_id = {
            str(item.get("id")): item
            for item in raw_items
            if isinstance(item, dict) and item.get("id") is not None
        }
        results: Dict[int, ExtractionResult] = {}
        for index in indices:
            item = by_id.get(str(index))
            results[index] = (
                self._result_from_payload(item)
                if item is not None
                else ExtractionResult(cacheable=False)
            )
        return results

    async def extract_batch(
        self,
        chunks: List[tuple],
        progress_callback=None,
        concurrency: int = 4,
    ) -> List[ExtractionResult]:
        """把多个原始 chunk 打包进一次请求，并保持逐 chunk 结果顺序。"""
        total = len(chunks)
        results = [ExtractionResult() for _ in chunks]
        eligible = [
            index
            for index, (text, _meta) in enumerate(chunks)
            if len((text or "").strip()) >= 50
        ]
        packs = self._make_packs(chunks, eligible)
        semaphore = asyncio.Semaphore(max(1, concurrency))
        progress_lock = asyncio.Lock()
        done = total - len(eligible)
        last_reported = 0

        async def report_progress(increment: int = 0, *, force: bool = False) -> None:
            nonlocal done, last_reported
            async with progress_lock:
                done += increment
                if not progress_callback:
                    return
                if force or done == total or done - last_reported >= 5:
                    last_reported = done
                    callback_result = progress_callback(
                        done,
                        total,
                        f"正在抽取知识图谱 {done}/{total}",
                    )
                    if asyncio.iscoroutine(callback_result):
                        await callback_result

        if done:
            await report_progress(force=True)

        async def run_pack(indices: List[int]) -> None:
            async with semaphore:
                try:
                    packed_results = await self._extract_pack(chunks, indices)
                except Exception as exc:
                    logger.warning(
                        "[extractor] packed chunks %s failed: %s",
                        indices,
                        exc,
                    )
                    packed_results = {
                        index: ExtractionResult(cacheable=False)
                        for index in indices
                    }
                for index in indices:
                    results[index] = packed_results.get(
                        index,
                        ExtractionResult(cacheable=False),
                    )
                await report_progress(len(indices))

        await asyncio.gather(*(run_pack(pack) for pack in packs))
        if total and last_reported != total:
            await report_progress(force=True)
        return results


def get_extractor(name: str = "llm", **kwargs) -> GraphExtractor:
    """抽取器工厂：按名称实例化，未知名称回退到 LLM 抽取器。"""
    if name == "llm":
        return LLMExtractor(**kwargs)
    logger.warning("[extractor] unknown extractor '%s', fallback to 'llm'", name)
    return LLMExtractor(**kwargs)
