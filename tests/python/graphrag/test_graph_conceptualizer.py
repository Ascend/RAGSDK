# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import unittest
from unittest.mock import Mock, patch

from mx_rag.graphrag.graph_conceptualizer import GraphConceptualizer
from mx_rag.utils.common import Lang


class TestGraphConceptualizer(unittest.TestCase):
    def setUp(self):
        """Set up mocks for GraphConceptualizer dependencies."""
        self.mock_llm = Mock()
        self.mock_llm.model_name = "mock_model"
        self.mock_llm.chat.return_value = "mock_response"
        self.mock_graph = Mock()
        self.mock_graph.get_nodes_by_attribute.return_value = ["node1", "node2"]
        self.mock_graph.get_edge_attribute_values.return_value = ["relation1", "relation2"]

    def test_initialization(self):
        """Test initialization of GraphConceptualizer."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            sample_num=None,
            lang=Lang.EN,
        )
        self.assertEqual(conceptualizer.llm, self.mock_llm)
        self.assertEqual(conceptualizer.graph, self.mock_graph)
        self.assertIsNone(conceptualizer.sample_num)
        self.assertEqual(conceptualizer.events, ["node1", "node2"])
        self.assertEqual(conceptualizer.entities, ["node1", "node2"])
        self.assertEqual(conceptualizer.relations, ["relation1", "relation2"])

    def test_conceptualize(self):
        """Test the conceptualize method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            sample_num=1,
            lang=Lang.EN,
        )
        with patch.object(conceptualizer, "_conceptualize_event", return_value={"node_type": "event"}), \
                patch.object(conceptualizer, "_conceptualize_entity", return_value={"node_type": "entity"}), \
                patch.object(conceptualizer, "_conceptualize_relation", return_value={"node_type": "relation"}):
            result = conceptualizer.conceptualize()
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["node_type"], "event")
            self.assertEqual(result[1]["node_type"], "entity")
            self.assertEqual(result[2]["node_type"], "relation")

    def test_conceptualize_event(self):
        """Test the _conceptualize_event method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        event = "event1"
        conceptualizer.prompt_map["event_prompt"] = "Event: [EVENT]"
        result = conceptualizer._conceptualize_event(event)
        self.mock_llm.chat.assert_called_once_with("Event: event1")
        self.assertEqual(result["node"], event)
        self.assertEqual(result["conceptualized_node"], "mock_response")
        self.assertEqual(result["node_type"], "event")

    def test_conceptualize_entity(self):
        """Test the _conceptualize_entity method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        entity = "entity1"
        self.mock_graph.predecessors.return_value = ["pred1"]
        self.mock_graph.successors.return_value = ["succ1"]
        self.mock_graph.get_edge_attributes.side_effect = lambda src, tgt, attr: f"{src} -> {tgt}"
        conceptualizer.prompt_map["entity_prompt"] = "Entity: [ENTITY] Context: [CONTEXT]"
        result = conceptualizer._conceptualize_entity(entity)
        self.mock_llm.chat.assert_called_once()
        self.assertIn("pred1 -> entity1", self.mock_llm.chat.call_args[0][0])
        self.assertIn("entity1 -> succ1", self.mock_llm.chat.call_args[0][0])
        self.assertEqual(result["node"], entity)
        self.assertEqual(result["conceptualized_node"], "mock_response")
        self.assertEqual(result["node_type"], "entity")

    def test_conceptualize_relation(self):
        """Test the _conceptualize_relation method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        relation = "relation1"
        conceptualizer.prompt_map["relation_prompt"] = "Relation: [RELATION]"
        result = conceptualizer._conceptualize_relation(relation)
        self.mock_llm.chat.assert_called_once_with("Relation: relation1")
        self.assertEqual(result["node"], relation)
        self.assertEqual(result["conceptualized_node"], "mock_response")
        self.assertEqual(result["node_type"], "relation")
