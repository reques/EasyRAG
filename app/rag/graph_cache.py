"""图谱内存缓存 — 索引时写入，检索时读取，避免 PG 连接线程冲突。

在 uvicorn 中，LangGraph 节点是同步函数，通过 _run_async_in_thread 访问 asyncpg
会导致连接池冲突。此模块将图谱数据缓存在进程内存中，检索时零 IO、无连接问题。

用法:
    from app.rag.graph_cache import graph_cache
    graph_cache.upsert_entity(kb_id, name, type, desc)  # 索引时
    entities = graph_cache.match_entities(keywords)       # 检索时
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class GraphCache:
    """进程级图谱缓存（线程安全）。"""

    def __init__(self):
        self._lock = threading.RLock()
        # entity_name → dict
        self._entities: Dict[str, Dict[str, Any]] = {}
        # (source_name, relation_type, target_name) → dict
        self._relations: List[Dict[str, Any]] = []
        # entity_name → 关联的关系索引
        self._entity_relations: Dict[str, List[int]] = defaultdict(list)

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: str = "",
        kb_id: str = "",
    ):
        with self._lock:
            existing = self._entities.get(name)
            if existing:
                if entity_type and entity_type != "concept":
                    existing["type"] = entity_type[:64]
                if description and description not in existing.get("description", ""):
                    existing["description"] = (
                        (existing.get("description", "") + "; " + description)[:1024]
                    )
            else:
                self._entities[name] = {
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
    ):
        with self._lock:
            idx = len(self._relations)
            rel = {
                "source": source,
                "target": target,
                "relation": relation_type[:128],
                "description": description[:1024],
            }
            self._relations.append(rel)
            self._entity_relations[source].append(idx)
            self._entity_relations[target].append(idx)

    def match_entities(self, keywords: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """关键词匹配实体（精确 + 包含）。"""
        with self._lock:
            matched = []
            for name, info in self._entities.items():
                for kw in keywords:
                    if kw in name or name in kw:
                        matched.append(dict(info))
                        break
            return matched[:top_n]

    def get_neighbor_relations(
        self,
        entity_name: str,
        max_relations: int = 6,
    ) -> List[Dict[str, Any]]:
        """获取实体的邻居关系。"""
        with self._lock:
            indices = self._entity_relations.get(entity_name, [])
            return [dict(self._relations[i]) for i in indices[:max_relations]]

    def get_relations_by_predicate(
        self,
        entity_names: List[str],
        predicates: List[str],
        max_chains: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索匹配指定实体和谓词的关系链（最多2跳）。"""
        with self._lock:
            chains = []
            for start in entity_names[:5]:
                # 一跳
                for idx in self._entity_relations.get(start, []):
                    rel = self._relations[idx]
                    if rel["source"] == start:  # 以 start 为 source
                        pred_match = any(p in rel["relation"] or rel["relation"] in p for p in predicates)
                        step1 = dict(rel)
                        if pred_match or len(chains) < 5:
                            chains.append({"steps": [step1]})
                        # 二跳
                        if len(chains) < max_chains:
                            for idx2 in self._entity_relations.get(rel["target"], [])[:3]:
                                rel2 = self._relations[idx2]
                                if rel2["source"] == rel["target"]:
                                    chains.append({"steps": [step1, dict(rel2)]})
                if len(chains) >= max_chains:
                    break
            return chains[:max_chains]

    def clear(self):
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


# 单例
graph_cache = GraphCache()
