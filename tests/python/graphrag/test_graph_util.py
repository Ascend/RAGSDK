#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import unittest
from unittest.mock import patch, MagicMock

from mx_rag.graphrag.graphs.graph_util import OpenGaussAGEAdapter, CypherQueryBuilder


class TestOpenGaussAGEAdapter(unittest.TestCase):
    def setUp(self):
        # Patch only the __init__ of openGaussAGEGraph and OpenGaussSettings
        patcher_graph_init = patch(
            'mx_rag.graphrag.graphs.graph_util.openGaussAGEGraph.__init__', return_value=None)
        patcher_settings = patch(
            'mx_rag.graphrag.graphs.graph_util.OpenGaussSettings', autospec=True)
        self.mock_graph_init = patcher_graph_init.start()
        self.mock_settings = patcher_settings.start()
        self.addCleanup(patcher_graph_init.stop)
        self.addCleanup(patcher_settings.stop)
        self.conf = self.mock_settings()
        self.conf.host = "localhost"
        self.conf.port = 5432
        self.conf.user = "user"
        self.conf.password = "pass"
        self.conf.database = "db"
        self.adapter = OpenGaussAGEAdapter(
            'test_graph', self.conf, create=False)

    def test_context_manager(self):
        # __enter__ returns self, __exit__ calls close
        with patch.object(self.adapter, 'close') as mock_close:
            result = self.adapter.__enter__()
            self.assertIs(result, self.adapter)
            self.adapter.__exit__(None, None, None)
            mock_close.assert_called_once()

    def test_get_cursor_yields_cursor(self):
        # _get_cursor should be called and yield its cursor
        mock_cursor = MagicMock()
        self.adapter._get_cursor = MagicMock()
        self.adapter._get_cursor.return_value.__enter__.return_value = mock_cursor
        with self.adapter.get_cursor() as cursor:
            self.assertIs(cursor, mock_cursor)
        self.adapter._get_cursor.assert_called_once()

    def test_execute_raw_query(self):
        # Should execute the query and fetchall
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [('row1',), ('row2',)]
        self.adapter.get_cursor = MagicMock()
        self.adapter.get_cursor.return_value.__enter__.return_value = mock_cursor
        result = self.adapter.execute_raw_query("SELECT 1")
        mock_cursor.execute.assert_called_once_with("SELECT 1")
        self.assertEqual(result, [('row1',), ('row2',)])

    def test_execute_cypher_query(self):
        # Should call self.query with the cypher query
        self.adapter.query = MagicMock(return_value='cypher_result')
        result = self.adapter.execute_cypher_query("MATCH (n) RETURN n")
        self.adapter.query.assert_called_once_with("MATCH (n) RETURN n")
        self.assertEqual(result, 'cypher_result')

    def test_close_closes_connection(self):
        # Should close connection if present
        mock_conn = MagicMock()
        self.adapter.connection = mock_conn
        self.adapter.close()
        mock_conn.close.assert_called_once()

    def test_close_no_connection(self):
        # Should not fail if no connection
        self.adapter.connection = None
        try:
            self.adapter.close()
        except Exception as e:
            self.fail(f"close() raised Exception unexpectedly: {e}")


class TestCypherQueryBuilder(unittest.TestCase):
    def test_merge_node(self):
        result = CypherQueryBuilder.merge_node({"id": "abc", "foo": 1})
        self.assertIn("CREATE (n:Node", result)
        self.assertIn("id: 'abc'", result)
        self.assertIn('foo: 1', result)

    def test_match_node(self):
        result = CypherQueryBuilder.match_node("label123")
        self.assertEqual(result, 'MATCH (n:Node {id: "label123"}) RETURN n LIMIT 1')

    def test_delete_node(self):
        result = CypherQueryBuilder.delete_node("label456")
        self.assertEqual(result, 'MATCH (n:Node {id: "label456"}) DETACH DELETE n')

    def test_match_node_properties(self):
        result = CypherQueryBuilder.match_node_properties("label789")
        self.assertEqual(result, 'MATCH (n:Node {id: "label789"}) RETURN properties(n) AS props')

    def test_match_node_attribute(self):
        result = CypherQueryBuilder.match_node_attribute("label", "foo")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"}) RETURN n.foo AS value')

    def test_set_node_attribute(self):
        result = CypherQueryBuilder.set_node_attribute("label", "foo", "bar")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"}) SET n.foo = \'bar\'')

    def test_set_node_attribute_append(self):
        result = CypherQueryBuilder.set_node_attribute("label", "foo", "bar", append=True)
        self.assertIn(
            "WITH n, CASE WHEN coalesce(n.foo, '') = '' THEN 'bar' else n.foo + ',' + 'bar' END AS new_value",
            result
        )
        self.assertIn("SET n.foo = new_value", result)

    def test_set_node_attributes(self):
        props = '[{label: "abc", value: 1}, {label: "def", value: 2}]'
        result = CypherQueryBuilder.set_node_attributes("foo", props)
        self.assertIn("UNWIND", result)
        self.assertIn("MATCH (n:Node) WHERE n.id = item.label", result)
        self.assertIn("SET n.foo = item.value", result)

    def test_match_nodes_with_data(self):
        result = CypherQueryBuilder.match_nodes(True)
        self.assertEqual(result, "MATCH (n:Node) RETURN n.text AS label, properties(n) AS props")

    def test_match_nodes_without_data(self):
        result = CypherQueryBuilder.match_nodes(False)
        self.assertEqual(result, "MATCH (n:Node) RETURN n.text AS label")

    def test_match_nodes_by_attribute(self):
        result = CypherQueryBuilder.match_nodes_by_attribute("foo", "bar")
        self.assertEqual(result, 'MATCH (n:Node) WHERE n.foo = \'bar\' RETURN properties(n) AS props')

    def test_match_nodes_containing_attribute(self):
        result = CypherQueryBuilder.match_nodes_containing_attribute("foo", "bar")
        self.assertIn("toString(n.foo) CONTAINS 'bar'", result)
        self.assertIn("RETURN properties(n) AS props", result)

    def test_merge_edge_with_props(self):
        result = CypherQueryBuilder.merge_edge("src", "dst", {"relation": "KNOWS", "weight": 2})
        self.assertIn("MERGE (a)-[r:`KNOWS` {relation: 'KNOWS', weight: 2}]->(b)", result)

    def test_merge_edge_without_props(self):
        result = CypherQueryBuilder.merge_edge("src", "dst", {})
        self.assertIn("MERGE (a)-[r:`related` {}]->(b)", result)

    def test_delete_edge(self):
        result = CypherQueryBuilder.delete_edge("src", "dst")
        self.assertEqual(result, 'MATCH (a:Node {id: "src"})-[r]->(b:Node {id: "dst"}) DELETE r')

    def test_match_edge(self):
        result = CypherQueryBuilder.match_edge("src", "dst")
        self.assertEqual(result, 'MATCH (a:Node {id: "src"})-[r]->(b:Node {id: "dst"}) RETURN r LIMIT 1')

    def test_match_edges_with_data(self):
        result = CypherQueryBuilder.match_edges(True)
        self.assertIn(
            "RETURN a.text AS source, b.text AS target, a.id AS start_id, b.id AS end_id, properties(r) AS props",
            result
        )

    def test_match_edges_without_data(self):
        result = CypherQueryBuilder.match_edges(False)
        self.assertIn("RETURN a.text AS source, b.text AS target, a.id AS start_id, b.id AS end_id", result)
        self.assertNotIn("properties(r) AS props", result)

    def test_match_edge_attribute_with_key(self):
        result = CypherQueryBuilder.match_edge_attribute("src", "dst", "foo")
        self.assertIn("RETURN r.foo AS value", result)

    def test_match_edge_attribute_without_key(self):
        result = CypherQueryBuilder.match_edge_attribute("src", "dst")
        self.assertIn("RETURN properties(r) AS props", result)

    def test_set_edge_attribute(self):
        result = CypherQueryBuilder.set_edge_attribute("src", "dst", "foo", "bar")
        self.assertIn('SET r.foo = \'bar\'', result)

    def test_set_edge_attribute_append(self):
        result = CypherQueryBuilder.set_edge_attribute("src", "dst", "foo", "bar", append=True)
        self.assertIn("SET r.foo = coalesce(r.foo, []) + 'bar'", result)

    def test_match_edges_by_attribute(self):
        result = CypherQueryBuilder.match_edges_by_attribute("foo")
        self.assertIn("WHERE exists(r.foo)", result)
        self.assertIn("RETURN a.id as source, b.id as target, properties(r) AS props", result)

    def test_in_degree(self):
        result = CypherQueryBuilder.in_degree("label")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"})<-[r]-() RETURN count(r) AS deg')

    def test_out_degree(self):
        result = CypherQueryBuilder.out_degree("label")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"})-[r]->() RETURN count(r) AS deg')

    def test_neighbors(self):
        result = CypherQueryBuilder.neighbors("label")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"})--(m) RETURN m.text as label')

    def test_successors(self):
        result = CypherQueryBuilder.successors("label")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"})-->(m) RETURN m.text as label')

    def test_predecessors(self):
        result = CypherQueryBuilder.predecessors("label")
        self.assertEqual(result, 'MATCH (n:Node {id: "label"})<--(m) RETURN m.text as label')

    def test_count_nodes(self):
        result = CypherQueryBuilder.count_nodes()
        self.assertEqual(result, "MATCH (n:Node) RETURN count(n) AS cnt")

    def test_count_edges(self):
        result = CypherQueryBuilder.count_edges()
        self.assertEqual(result, "MATCH ()-[r]->() RETURN count(r) AS cnt")
