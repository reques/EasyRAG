"""图谱抽取器（可配置，默认 LLM JSON 抽取，更多方式拓展中）。"""

from app.rag.extractors.base import (
    EntityExtraction,
    ExtractionResult,
    GraphExtractor,
    RelationExtraction,
)
from app.rag.extractors.llm_extractor import EXTRACT_PROMPT, LLMExtractor, get_extractor

__all__ = [
    "EntityExtraction",
    "RelationExtraction",
    "ExtractionResult",
    "GraphExtractor",
    "LLMExtractor",
    "EXTRACT_PROMPT",
    "get_extractor",
]
