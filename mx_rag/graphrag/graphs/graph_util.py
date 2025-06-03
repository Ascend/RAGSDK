# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langchain_opengauss import openGaussAGEGraph, OpenGaussSettings


def cypher_value(v, depth=0, seen=None):
    """
    Convert a Python value to a safe representation for Cypher queries

    Parameters:
        v: The value to convert (str, int, float, bool, None, list, dict)
        depth: The current recursion depth (default is 0)
        seen: A set of object IDs already processed to detect circular references (default is None)

    Returns:
        str: A string representation safe to embed in Cypher queries

    Raises:
        ValueError: If the structure is too deep or contains circular references
    """
    if seen is None:
        seen = set()

    if depth > sys.getrecursionlimit():
        raise ValueError("Structure too deep - possible circular reference")

    if id(v) in seen:
        raise ValueError("Circular reference detected")
    seen.add(id(v))

    try:
        if v is None:
            return 'null'
        elif isinstance(v, bool):
            return str(v).lower()
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, str):
            # Escape single quotes and enclose in single quotes
            escaped = v.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(v, (list, tuple, dict)):
            new_seen = set(seen)
            if isinstance(v, (list, tuple)):
                items = [cypher_value(item, depth + 1, new_seen) for item in v]
                return f'[{", ".join(items)}]'
            else:
                pairs = [f'{key}: {cypher_value(value, depth + 1, new_seen)}' for key, value in v.items()]
                return f'{{{", ".join(pairs)}}}'
        else:
            raise ValueError(f"Unsupported type for Cypher value: {type(v)}")
    finally:
        seen.remove(id(v))


class OpenGaussAGEAdapter(openGaussAGEGraph):
    """
    Adapter class that extends openGaussAGEGraph to expose additional utility methods
    for database operations while maintaining full compatibility with the parent class.
    """

    def __init__(self, graph_name: str, conf: OpenGaussSettings, create: bool = True):
        """
        Initialize the adapter by calling the parent constructor.

        Args:
            graph_name (str): The name of the graph to connect to or create
            conf (OpenGaussSettings): The openGauss connection configuration
            create (bool): If True and graph doesn't exist, attempt to create it
        """
        super().__init__(graph_name, conf, create)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @contextmanager
    def get_cursor(self):
        """
        Expose the _get_cursor method as a public method.

        Returns:
            A database cursor context manager
        """
        with self._get_cursor() as cursor:
            yield cursor

    def execute_raw_query(self, query: str) -> Any:
        """
        Execute a raw SQL query using the exposed cursor.

        Args:
            query (str): The SQL query to execute

        Returns:
            Query results
        """
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def execute_cypher_query(self, cypher_query: str) -> Any:
        """
        Execute a Cypher query through the graph instance.

        Args:
            cypher_query (str): The Cypher query to execute

        Returns:
            Query results
        """
        return self.query(cypher_query)

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()


class CypherQueryBuilder:
    """Helper class for building Cypher queries."""

    @staticmethod
    def merge_node(attributes: Dict[str, Any]) -> str:
        query = f"CREATE (n:Node {cypher_value(attributes)})"
        return query

    @staticmethod
    def match_node(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}}) RETURN n LIMIT 1"

    @staticmethod
    def delete_node(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}}) DETACH DELETE n"

    @staticmethod
    def match_node_properties(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}}) RETURN properties(n) AS props"

    @staticmethod
    def match_node_attribute(label: str, key: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}}) RETURN n.{key} AS value"

    @staticmethod
    def set_node_attribute(label: str, key: str, value, append: bool = False) -> str:
        val = cypher_value(value)
        if append:
            return (
                f"MATCH (n:Node {{id: \"{label}\"}}) "
                f"WITH n, CASE WHEN coalesce(n.{key}, '') = '' THEN {val} else n.{key} + ',' + {val} END AS new_value "
                f"SET n.{key} = new_value"
            )
        return f"MATCH (n:Node {{id: \"{label}\"}}) SET n.{key} = {cypher_value(value)}"

    @staticmethod
    def set_node_attributes(name: str, props) -> str:
        return (
            f"UNWIND {props} AS item "
            f"MATCH (n:Node) WHERE n.id = item.label "
            f"SET n.{name} = item.value"
        )

    @staticmethod
    def match_nodes(with_data: bool = True) -> str:
        if with_data:
            return "MATCH (n:Node) RETURN n.text AS label, properties(n) AS props"
        return "MATCH (n:Node) RETURN n.text AS label"

    @staticmethod
    def match_nodes_by_attribute(key: str, value) -> str:
        return f"MATCH (n:Node) WHERE n.{key} = {cypher_value(value)} RETURN properties(n) AS props"

    @staticmethod
    def match_nodes_containing_attribute(key: str, value: str) -> str:
        return (
            f"MATCH (n:Node) WHERE toString(n.{key}) CONTAINS {cypher_value(value)} "
            "RETURN properties(n) AS props"
        )

    @staticmethod
    def merge_edge(source_label: str, target_label: str, attributes: Dict[str, Any]) -> str:
        props = cypher_value(attributes)
        relation = attributes.get("relation", "related")
        if props:
            query = (
                f"MATCH (a:Node {{id: \"{source_label}\"}}), (b:Node {{id: \"{target_label}\"}}) "
                f"MERGE (a)-[r:`{relation}` {props}]->(b)"
            )
        else:
            query = (
                f"MATCH (a:Node {{id: \"{source_label}\"}}), (b:Node {{id: \"{target_label}\"}}) "
                f"MERGE (a)-[r:`{relation}`]->(b)"
            )
        return query

    @staticmethod
    def delete_edge(source_label: str, target_label: str) -> str:
        return f"MATCH (a:Node {{id: \"{source_label}\"}})-[r]->(b:Node {{id: \"{target_label}\"}}) DELETE r"

    @staticmethod
    def match_edge(source_label: str, target_label: str) -> str:
        return f"MATCH (a:Node {{id: \"{source_label}\"}})-[r]->(b:Node {{id: \"{target_label}\"}}) RETURN r LIMIT 1"

    @staticmethod
    def match_edges(with_data: bool = True) -> str:
        base = (
            "MATCH (a:Node)-[r]->(b:Node) "
            "RETURN a.text AS source, b.text AS target, a.id AS start_id, b.id AS end_id"
        )
        if with_data:
            return f"{base}, properties(r) AS props"
        return base

    @staticmethod
    def match_edge_attribute(source_label: str, target_label: str, key: Optional[str] = None) -> str:
        if key:
            return (
                f"MATCH (:Node {{id: \"{source_label}\"}})-[r]->(:Node {{id: \"{target_label}\"}}) "
                f"RETURN r.{key} AS value"
            )
        return (
            f"MATCH (:Node {{id: \"{source_label}\"}})-[r]->(:Node {{id: \"{target_label}\"}}) "
            f"RETURN properties(r) AS props"
        )

    @staticmethod
    def set_edge_attribute(source_label: str, target_label: str, key: str, value, append: bool = False) -> str:
        if append:
            return (
                f"MATCH (a:Node {{id: \"{source_label}\"}})-[r]->(b:Node {{id: \"{target_label}\"}}) "
                f"SET r.{key} = coalesce(r.{key}, []) + {cypher_value(value)}"
            )
        return (
            f"MATCH (a:Node {{id: \"{source_label}\"}})-[r]->(b:Node {{id: \"{target_label}\"}}) "
            f"SET r.{key} = {cypher_value(value)}"
        )

    @staticmethod
    def match_edges_by_attribute(key: str) -> str:
        return (
            f"MATCH (a:Node)-[r]->(b:Node) WHERE exists(r.{key}) "
            "RETURN a.id as source, b.id as target, properties(r) AS props"
        )

    @staticmethod
    def in_degree(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}})<-[r]-() RETURN count(r) AS deg"

    @staticmethod
    def out_degree(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}})-[r]->() RETURN count(r) AS deg"

    @staticmethod
    def neighbors(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}})--(m) RETURN m.text as label"

    @staticmethod
    def successors(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}})-->(m) RETURN m.text as label"

    @staticmethod
    def predecessors(label: str) -> str:
        return f"MATCH (n:Node {{id: \"{label}\"}})<--(m) RETURN m.text as label"

    @staticmethod
    def count_nodes() -> str:
        return "MATCH (n:Node) RETURN count(n) AS cnt"

    @staticmethod
    def count_edges() -> str:
        return "MATCH ()-[r]->() RETURN count(r) AS cnt"
