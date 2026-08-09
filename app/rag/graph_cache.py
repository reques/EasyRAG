"""Thread-safe, knowledge-base-scoped in-memory graph cache.

Graph extraction writes to this cache during ingestion. Retrieval is always
scoped to an explicit set of authorised knowledge-base ids; an empty scope
returns no data.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

EntityKey = Tuple[str, str]  # (knowledge_base_id, entity_name)


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
    ) -> None:
        if not kb_id:
            logger.warning("[graph_cache] ignored entity without knowledge_base_id")
            return
        key = (kb_id, name)
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
            }

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        description: str = "",
        kb_id: str = "",
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
            }
            self._relations.append(rel)
            self._entity_relations[(kb_id, source)].append(idx)
            self._entity_relations[(kb_id, target)].append(idx)

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
            matched = []
            for (kb_id, name), info in self._entities.items():
                if kb_id not in allowed_ids:
                    continue
                if any(kw in name or name in kw for kw in keywords):
                    matched.append(dict(info))
                if len(matched) >= top_n:
                    break
            return matched

    def get_neighbor_relations(
        self,
        entity_name: str,
        max_relations: int = 6,
        knowledge_base_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        allowed_ids = list(knowledge_base_ids or [])
        if not allowed_ids:
            return []
        with self._lock:
            indices: List[int] = []
            for kb_id in allowed_ids:
                indices.extend(self._entity_relations.get((kb_id, entity_name), []))
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
                    for idx in self._entity_relations.get((kb_id, start), []):
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
                            target_key = (kb_id, rel["target"])
                            for idx2 in self._entity_relations.get(target_key, [])[:3]:
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
            return chains[:max_chains]

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()
            self._relations.clear()
            self._entity_relations.clear()

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entities": len(self._entities),
                "relations": len(self._relations),
            }


graph_cache = GraphCache()
