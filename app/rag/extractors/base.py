"""图谱抽取器抽象（GraphRAG 阶段 5）。

抽取器把一段 chunk 文本变成结构化的实体/关系列表。当前内置
``LLMExtractor``（LLM JSON 模式抽取），未来可扩展其他方式
（如 neo4j-graphrag 的 SimpleKGPipeline、基于规则的抽取等），
通过 ``get_extractor(name)`` 工厂按配置切换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EntityExtraction:
    """一个实体。"""

    name: str
    entity_type: str = "concept"
    description: str = ""


@dataclass
class RelationExtraction:
    """一条有向关系（三元组）。"""

    source: str
    target: str
    relation: str
    description: str = ""


@dataclass
class ExtractionResult:
    """一次抽取的输出。"""

    entities: List[EntityExtraction] = field(default_factory=list)
    relations: List[RelationExtraction] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.entities and not self.relations


class GraphExtractor:
    """抽取器基类。子类实现 :meth:`extract`。"""

    name: str = "base"

    async def extract(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """从单段文本抽取实体与关系。"""
        raise NotImplementedError

    async def extract_batch(
        self,
        chunks: List[tuple],
        progress_callback: Optional[Callable[[int, int, str], Awaitable[None]]] = None,
        concurrency: int = 4,
    ) -> List[ExtractionResult]:
        """并发抽取（默认 4 路），单 chunk 失败跳过不中断。

        返回与输入等长的结果列表（``asyncio.gather`` 保持输入顺序）。
        chunks: [(text, meta), ...]
        """
        import asyncio

        total = len(chunks)
        sem = asyncio.Semaphore(max(1, concurrency))
        done = 0
        lock = asyncio.Lock()

        async def _one(i: int, text: str, meta) -> ExtractionResult:
            nonlocal done
            try:
                if len((text or "").strip()) < 50:  # 太短没有抽取价值
                    return ExtractionResult()
                return await self.extract(text, meta)
            except Exception as exc:  # 单 chunk 失败不阻塞构建
                logger.warning("[extractor] chunk %d failed: %s", i, exc)
                return ExtractionResult()
            finally:
                async with lock:
                    done += 1
                    if progress_callback and (done % 5 == 0 or done == total):
                        cb = progress_callback(
                            done, total, f"正在抽取知识图谱 {done}/{total}"
                        )
                        if asyncio.iscoroutine(cb):
                            await cb

        async def _guarded(i: int, text: str, meta) -> ExtractionResult:
            async with sem:
                return await _one(i, text, meta)

        return await asyncio.gather(
            *(_guarded(i, t, m) for i, (t, m) in enumerate(chunks))
        )
