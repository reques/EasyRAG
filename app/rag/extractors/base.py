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
    # 网络失败、JSON 缺项等降级结果不能写入持久缓存，否则会把瞬时故障
    # 固化成永久空结果。正常的“没有实体/关系”仍然可以缓存。
    cacheable: bool = True

    @property
    def empty(self) -> bool:
        return not self.entities and not self.relations

    def to_dict(self) -> Dict[str, Any]:
        """转换成稳定的 JSON 兼容结构，供持久化缓存使用。"""
        return {
            "entities": [
                {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "description": entity.description,
                }
                for entity in self.entities
            ],
            "relations": [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "relation": relation.relation,
                    "description": relation.description,
                }
                for relation in self.relations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """从缓存 JSON 恢复；字段上限与 LLM 抽取器保持一致。"""
        if not isinstance(data, dict):
            raise ValueError("extraction cache payload must be an object")
        result = cls()
        for item in (data.get("entities") or [])[:20]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            result.entities.append(EntityExtraction(
                name=name[:256],
                entity_type=str(item.get("type") or "concept")[:64],
                description=str(item.get("description") or "")[:1024],
            ))
        for item in (data.get("relations") or [])[:20]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            relation = str(item.get("relation") or "").strip()
            if not (source and target and relation):
                continue
            result.relations.append(RelationExtraction(
                source=source[:256],
                target=target[:256],
                relation=relation[:128],
                description=str(item.get("description") or "")[:1024],
            ))
        return result


class GraphExtractor:
    """抽取器基类。子类实现 :meth:`extract`。"""

    name: str = "base"
    prompt_version: str = "base-v1"
    model_name: str = ""

    def cache_input(self, text: str) -> str:
        """返回真正送入抽取器的文本，用于生成内容缓存键。"""
        return text or ""

    def cache_fingerprint(self) -> str:
        """模型或 prompt 改变时必须变化，从而自动失效旧缓存。"""
        return f"{self.name}:{self.model_name}:{self.prompt_version}"

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
                return ExtractionResult(cacheable=False)
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
