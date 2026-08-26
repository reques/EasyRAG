"""Thread-safe, knowledge-base-scoped in-memory graph cache.

Graph extraction writes to this cache during ingestion. Retrieval is always
scoped to an explicit set of authorised knowledge-base ids; an empty scope
returns no data.

Entity identity = (knowledge_base_id, source_file, name) — 同名实体在不同
文件（source_file）中各自独立成节点，避免跨文档同名污染（如两个文件的
「第六章」不再错误合并）。source_file 为 None 时按 name 匹配（老数据兜底）。
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

# (knowledge_base_id, source_file, entity_name) — source_file 可为 None（老数据）
EntityKey = Tuple[str, Optional[str], str]


def _ek(kb_id: str, source_file: Optional[str], name: str) -> EntityKey:
    return (kb_id, source_file or "", name)


class GraphCache:
    """Process-local graph cache with tenant-safe lookup keys."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entities: Dict[EntityKey, Dict[str, Any]] = {}
        self._relations: List[Dict[str, Any]] = []
        self._entity_relations: Dict[EntityKey, List[int]] = defaultdict(list)

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: str = "",
        kb_id: str = "",
        source_file: Optional[str] = None,
    ) -> None:
        if not kb_id:
            logger.warning("[graph_cache] ignored entity without knowledge_base_id")
            return
        key = _ek(kb_id, source_file, name)
        with self._lock:
            existing = self._entities.get(key)
            if existing:
                if entity_type and entity_type != "concept":
                    existing["type"] = entity_type[:64]
                if description and description not in existing.get("description", ""):
                    existing["description"] = (
                        existing.get("description", "") + "; " + description
                    )[:1024]
                return
            self._entities[key] = {
                "name": name,
                "type": entity_type[:64],
                "description": description[:1024],
                "kb_id": kb_id,
                "source_file": source_file,
            }

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        description: str = "",
        kb_id: str = "",
        source_file: Optional[str] = None,
    ) -> None:
        if not kb_id:
            logger.warning("[graph_cache] ignored relation without knowledge_base_id")
            return
        with self._lock:
            idx = len(self._relations)
            rel = {
                "source": source,
                "target": target,
                "relation": relation_type[:128],
                "description": description[:1024],
                "kb_id": kb_id,
                "source_file": source_file,
            }
            self._relations.append(rel)
            self._entity_relations[_ek(kb_id, source_file, source)].append(idx)
            self._entity_relations[_ek(kb_id, source_file, target)].append(idx)

    def match_entities(
        self,
        keywords: List[str],
        top_n: int = 5,
        knowledge_base_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        allowed_ids = set(knowledge_base_ids or [])
        if not allowed_ids:
            return []
        with self._lock:
            name_hits: List[Dict[str, Any]] = []
            desc_hits: List[Dict[str, Any]] = []
            for (kb_id, _sf, name), info in self._entities.items():
                if kb_id not in allowed_ids:
                    continue
                desc = info.get("description", "") or ""
                if any(kw in name or name in kw for kw in keywords):
                    name_hits.append(dict(info))
                elif any(kw and kw in desc for kw in keywords):
                    desc_hits.append(dict(info))
            # name 精确命中优先，description 命中兜底（通用：关键词可命中实体正文描述）
            return (name_hits + desc_hits)[:top_n]

    def get_neighbor_relations(
        self,
        entity_name: str,
        max_relations: int = 6,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        source_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询实体的邻居关系。

        source_file 给定 → 只查该文件内的同名实体（命名空间隔离）；
        source_file 为 None → 按 name 查所有文件（兼容老数据/聚合场景）。
        """
        allowed_ids = list(knowledge_base_ids or [])
        if not allowed_ids:
            return []
        with self._lock:
            indices: List[int] = []
            for kb_id in allowed_ids:
                if source_file is not None:
                    indices.extend(
                        self._entity_relations.get(_ek(kb_id, source_file, entity_name), [])
                    )
                else:
                    # 老数据兜底：source_file 为 None 时匹配所有该 (kb_id, name) 的键
                    for sf_key in (None, ""):
                        indices.extend(
                            self._entity_relations.get((kb_id, sf_key, entity_name), [])
                        )
            return [dict(self._relations[i]) for i in indices[:max_relations]]

    def get_relations_by_predicate(
        self,
        entity_names: List[str],
        predicates: List[str],
        max_chains: int = 10,
        knowledge_base_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        allowed_ids = list(knowledge_base_ids or [])
        if not allowed_ids:
            return []
        with self._lock:
            chains = []
            for kb_id in allowed_ids:
                for start in entity_names[:5]:
                    for sf_key in (None, ""):
                        for idx in self._entity_relations.get((kb_id, sf_key, start), []):
                            rel = self._relations[idx]
                            if rel["source"] != start:
                                continue
                            pred_match = any(
                                p in rel["relation"] or rel["relation"] in p
                                for p in predicates
                            )
                            step1 = dict(rel)
                            if pred_match or len(chains) < 5:
                                chains.append({"steps": [step1], "kb_id": kb_id})
                            if len(chains) < max_chains:
                                for idx2 in self._entity_relations.get(
                                    (kb_id, sf_key, rel["target"]), []
                                )[:3]:
                                    rel2 = self._relations[idx2]
                                    if rel2["source"] == rel["target"]:
                                        chains.append({
                                            "steps": [step1, dict(rel2)],
                                            "kb_id": kb_id,
                                        })
                            if len(chains) >= max_chains:
                                break
                    if len(chains) >= max_chains:
                        break
                if len(chains) >= max_chains:
                    break
            return chains[:max_chains]

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()
            self._relations.clear()
            self._entity_relations.clear()

    def clear_kb(self, kb_id: str) -> None:
        """清空某知识库的全部缓存条目（图谱重置用）。"""
        with self._lock:
            self._entities = {
                k: v for k, v in self._entities.items() if k[0] != kb_id
            }
            keep = [
                (i, r) for i, r in enumerate(self._relations)
                if r.get("kb_id") != kb_id
            ]
            self._relations = [r for _, r in keep]
            rebuilt = defaultdict(list)
            for new_idx, (_, rel) in enumerate(keep):
                rebuilt[_ek(rel["kb_id"], rel.get("source_file"), rel["source"])].append(new_idx)
                rebuilt[_ek(rel["kb_id"], rel.get("source_file"), rel["target"])].append(new_idx)
            self._entity_relations = rebuilt

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entities": len(self._entities),
                "relations": len(self._relations),
            }


graph_cache = GraphCache()
