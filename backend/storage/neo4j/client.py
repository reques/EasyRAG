"""Neo4j 图谱存储客户端（GraphRAG 阶段 5）。

封装官方 ``neo4j`` driver 的懒加载单例，提供知识库隔离的写入/查询原语：

- 节点统一 label ``Entity``，属性携带 ``kb_id`` —— 多知识库隔离，
  所有查询强制按 kb_id 过滤，绝不跨库读写；
- 唯一约束 ``(kb_id, name)``，写入用 MERGE 幂等；
- 关系 ``RELATES`` 带 ``relation_type``、``chunk_ids``（溯源引用）等属性。

本模块只暴露同步 API；异步调用方用 ``asyncio.to_thread`` 包装（本地
Bolt 查询毫秒级，构建/查询低频，可接受）。Neo4j 不可达时所有方法
抛出 ``Neo4jUnavailableError``，由上层决定降级（检索跳过图谱路）。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

ENTITY_LABEL = "Entity"
REL_LABEL = "RELATES"

# 关系属性中保存的 chunk 引用上限（防止单条关系属性无限膨胀）
MAX_CHUNK_REFS_PER_REL = 200


class Neo4jUnavailableError(RuntimeError):
    """Neo4j 不可达或操作失败。"""


class Neo4jClient:
    """懒加载单例 driver + 图谱原语。"""

    def __init__(self) -> None:
        self._driver = None
        self._lock = threading.Lock()
        self._schema_ready = False

    # ── driver 生命周期 ──────────────────────────────────────────────────

    def _get_driver(self):
        if self._driver is None:
            with self._lock:
                if self._driver is None:
                    from neo4j import GraphDatabase

                    try:
                        self._driver = GraphDatabase.driver(
                            cfg.NEO4J_URI,
                            auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
                        )
                        self._driver.verify_connectivity()
                        logger.info(
                            "[neo4j] connected to %s (user=%s)",
                            cfg.NEO4J_URI, cfg.NEO4J_USER,
                        )
                    except Exception as exc:
                        self._driver = None
                        raise Neo4jUnavailableError(
                            f"Neo4j unreachable at {cfg.NEO4J_URI}: {exc}"
                        ) from exc
        return self._driver

    def close(self) -> None:
        with self._lock:
            if self._driver is not None:
                self._driver.close()
                self._driver = None
                self._schema_ready = False

    @property
    def available(self) -> bool:
        try:
            self._get_driver()
            return True
        except Neo4jUnavailableError:
            return False

    # ── schema 初始化 ────────────────────────────────────────────────────

    def init_schema(self) -> None:
        """创建唯一约束/索引（幂等）。仅在连接成功后执行一次。"""
        if self._schema_ready:
            return
        driver = self._get_driver()
        with driver.session() as session:
            # 复合唯一约束：同一知识库内实体名唯一（Neo4j 5.7+）
            session.run(
                "CREATE CONSTRAINT entity_kb_name_unique IF NOT EXISTS "
                f"FOR (e:{ENTITY_LABEL}) REQUIRE (e.kb_id, e.name) IS UNIQUE"
            )
            # 实体名模糊查询索引（子图搜索 CONTAINS 加速）
            session.run(
                "CREATE INDEX entity_name_idx IF NOT EXISTS "
                f"FOR (e:{ENTITY_LABEL}) ON (e.name)"
            )
        self._schema_ready = True

    # ── 写入 ─────────────────────────────────────────────────────────────

    def upsert_entity(
        self,
        kb_id: str,
        name: str,
        entity_type: str = "concept",
        description: str = "",
    ) -> None:
        """幂等写入实体节点；已存在时合并类型/描述。"""
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                f"MERGE (e:{ENTITY_LABEL} {{kb_id: $kb_id, name: $name}}) "
                "SET e.entity_type = CASE WHEN $entity_type <> '' "
                "THEN $entity_type ELSE e.entity_type END, "
                "e.description = CASE WHEN $description <> '' "
                "THEN $description ELSE e.description END",
                kb_id=kb_id, name=name,
                entity_type=(entity_type or "concept")[:64],
                description=(description or "")[:1024],
            )

    def upsert_relation(
        self,
        kb_id: str,
        source: str,
        target: str,
        relation_type: str,
        description: str = "",
        chunk_id: str = "",
    ) -> None:
        """幂等写入关系；chunk_id 累积进 chunk_ids（去重，上限保护）。"""
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                f"MATCH (a:{ENTITY_LABEL} {{kb_id: $kb_id, name: $source}}) "
                f"MATCH (b:{ENTITY_LABEL} {{kb_id: $kb_id, name: $target}}) "
                f"MERGE (a)-[r:{REL_LABEL} {{kb_id: $kb_id, relation_type: $relation_type}}]->(b) "
                "SET r.description = CASE WHEN $description <> '' "
                "THEN $description ELSE r.description END, "
                "r.chunk_ids = apoc.coll.toSet("
                "  CASE WHEN $chunk_id <> '' "
                "  THEN apoc.coll.union(coalesce(r.chunk_ids, []), [$chunk_id]) "
                "  ELSE coalesce(r.chunk_ids, []) END)[0..$max_refs]",
                kb_id=kb_id, source=source, target=target,
                relation_type=relation_type[:128],
                description=(description or "")[:1024],
                chunk_id=chunk_id,
                max_refs=MAX_CHUNK_REFS_PER_REL,
            )

    # ── 批量写入（构建加速：UNWIND 一次会话写入全部）────────────────────

    def upsert_entities_batch(
        self,
        kb_id: str,
        entities: Dict[str, Tuple[str, str]],
    ) -> None:
        """批量 upsert 实体。entities: {name: (entity_type, description)}。"""
        rows = [
            {
                "name": name,
                "entity_type": (entity_type or "concept")[:64],
                "description": (description or "")[:1024],
            }
            for name, (entity_type, description) in entities.items()
        ]
        if not rows:
            return
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                f"UNWIND $rows AS row "
                f"MERGE (e:{ENTITY_LABEL} {{kb_id: $kb_id, name: row.name}}) "
                "SET e.entity_type = CASE WHEN row.entity_type <> '' "
                "THEN row.entity_type ELSE e.entity_type END, "
                "e.description = CASE WHEN row.description <> '' "
                "THEN row.description ELSE e.description END",
                rows=rows, kb_id=kb_id,
            )

    def upsert_relations_batch(
        self,
        kb_id: str,
        relations: Dict[Tuple[str, str, str], Tuple[str, List[str]]],
    ) -> None:
        """批量 upsert 关系（Python 侧已按 (source, relation, target) 聚合）。

        relations: {(source, relation, target): (description, chunk_ids)}。
        """
        rows = [
            {
                "source": src,
                "relation": rel[:128],
                "target": tgt,
                "description": (desc or "")[:1024],
                "chunk_ids": [c for c in chunk_ids if c][:MAX_CHUNK_REFS_PER_REL],
            }
            for (src, rel, tgt), (desc, chunk_ids) in relations.items()
        ]
        if not rows:
            return
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:{ENTITY_LABEL} {{kb_id: $kb_id, name: row.source}}) "
                f"MATCH (b:{ENTITY_LABEL} {{kb_id: $kb_id, name: row.target}}) "
                f"MERGE (a)-[r:{REL_LABEL} {{kb_id: $kb_id, relation_type: row.relation}}]->(b) "
                "SET r.description = CASE WHEN row.description <> '' "
                "THEN row.description ELSE r.description END, "
                "r.chunk_ids = apoc.coll.toSet("
                "  apoc.coll.union(coalesce(r.chunk_ids, []), row.chunk_ids))",
                rows=rows, kb_id=kb_id,
            )

    # ── 查询 ─────────────────────────────────────────────────────────────

    def search_entities(self, kb_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """按名称模糊匹配实体（子图搜索入口）。"""
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(
                f"MATCH (e:{ENTITY_LABEL}) "
                "WHERE e.kb_id = $kb_id AND e.name CONTAINS $query "
                "RETURN e.name AS name, e.entity_type AS entity_type, "
                "e.description AS description "
                "ORDER BY size(e.name) LIMIT $limit",
                kb_id=kb_id, query=query, limit=limit,
            )
            return [dict(rec) for rec in result]

    def get_subgraph(
        self,
        kb_id: str,
        entity_name: str,
        depth: int = 1,
        max_nodes: int = 60,
    ) -> Dict[str, Any]:
        """以实体为中心扩展子图（默认 1 跳），返回 nodes + edges。

        供前端子图展示：nodes=[{id, name, entity_type, description}],
        edges=[{source, target, relation_type, chunk_ids}].
        """
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(
                f"MATCH path = (start:{ENTITY_LABEL} {{kb_id: $kb_id, name: $name}})"
                f"-[:{REL_LABEL}*1..{max(1, int(depth))}]-(:{ENTITY_LABEL}) "
                "WITH nodes(path) AS ns, relationships(path) AS rels "
                "LIMIT $max_nodes "
                "WITH apoc.coll.toSet(apoc.coll.flatten(collect(ns))) AS all_nodes, "
                "     apoc.coll.toSet(apoc.coll.flatten(collect(rels))) AS all_rels "
                "RETURN all_nodes, all_rels",
                kb_id=kb_id, name=entity_name, max_nodes=max_nodes,
            )
            rec = result.single()
            if not rec:
                return {"nodes": [], "edges": []}
            nodes: Dict[str, Dict[str, Any]] = {}
            for node in rec["all_nodes"]:
                nid = f"{node['kb_id']}:{node['name']}"
                if nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "name": node["name"],
                        "entity_type": node.get("entity_type", "concept"),
                        "description": node.get("description", ""),
                    }
            edges: List[Dict[str, Any]] = []
            for rel in rec["all_rels"]:
                src_node = rel.nodes[0]
                tgt_node = rel.nodes[-1]
                edges.append({
                    "source": f"{src_node['kb_id']}:{src_node['name']}",
                    "target": f"{tgt_node['kb_id']}:{tgt_node['name']}",
                    "relation_type": rel["relation_type"],
                    "chunk_ids": list(rel.get("chunk_ids") or []),
                })
            return {"nodes": list(nodes.values()), "edges": edges}

    def get_entity_chunk_refs(self, kb_id: str, entity_name: str, limit: int = 20) -> List[str]:
        """实体直接关联的 chunk 引用（检索召回时映射回 chunk）。"""
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(
                f"MATCH (e:{ENTITY_LABEL} {{kb_id: $kb_id, name: $name}})"
                f"-[r:{REL_LABEL}]-( ) "
                "RETURN apoc.coll.toSet(apoc.coll.flatten(collect(coalesce(r.chunk_ids, [])))) AS ids "
                "LIMIT 1",
                kb_id=kb_id, name=entity_name,
            )
            rec = result.single()
            if not rec:
                return []
            return [str(i) for i in (rec["ids"] or [])][:limit]

    def get_relation_chunk_refs(
        self,
        kb_id: str,
        source: str,
        target: str,
        relation: str,
        limit: int = 20,
    ) -> List[str]:
        """三元组命中的 chunk 引用（检索召回时映射回 chunk）。"""
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(
                f"MATCH (a:{ENTITY_LABEL} {{kb_id: $kb_id, name: $source}})"
                f"-[r:{REL_LABEL} {{kb_id: $kb_id, relation_type: $relation}}]->"
                f"(b:{ENTITY_LABEL} {{kb_id: $kb_id, name: $target}}) "
                "RETURN apoc.coll.toSet(apoc.coll.flatten(collect(coalesce(r.chunk_ids, [])))) AS ids "
                "LIMIT 1",
                kb_id=kb_id, source=source, target=target, relation=relation,
            )
            rec = result.single()
            if not rec:
                return []
            return [str(i) for i in (rec["ids"] or [])][:limit]

    def count_stats(self, kb_id: str) -> Dict[str, int]:
        """知识库图谱统计：实体数 / 关系数。"""
        driver = self._get_driver()
        with driver.session() as session:
            ent = session.run(
                f"MATCH (e:{ENTITY_LABEL} {{kb_id: $kb_id}}) RETURN count(e) AS n",
                kb_id=kb_id,
            ).single()["n"]
            rel = session.run(
                f"MATCH ()-[r:{REL_LABEL} {{kb_id: $kb_id}}]->() RETURN count(r) AS n",
                kb_id=kb_id,
            ).single()["n"]
            return {"entities": int(ent), "relations": int(rel)}

    # ── 清理 ─────────────────────────────────────────────────────────────

    def clear_kb(self, kb_id: str) -> None:
        """删除知识库的全部图谱数据（重置用）。"""
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                f"MATCH (e:{ENTITY_LABEL} {{kb_id: $kb_id}}) DETACH DELETE e",
                kb_id=kb_id,
            )


_neo4j_client: Optional[Neo4jClient] = None
_client_lock = threading.Lock()


def get_neo4j_client() -> Neo4jClient:
    """进程级单例。"""
    global _neo4j_client
    if _neo4j_client is None:
        with _client_lock:
            if _neo4j_client is None:
                _neo4j_client = Neo4jClient()
    return _neo4j_client


neo4j_client = get_neo4j_client()
