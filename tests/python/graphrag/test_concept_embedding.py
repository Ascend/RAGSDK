# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import Mock

from mx_rag.graphrag.concept_embedding import ConceptEmbedding


class TestConceptEmbedding(unittest.TestCase):
    def setUp(self):
        """Set up a mock embedding function for testing."""
        self.mock_embed_func = Mock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        self.concept_embedding = ConceptEmbedding(self.mock_embed_func)

    def test_init_with_valid_callable(self):
        """Test initialization with a valid embedding function."""
        self.assertIsInstance(self.concept_embedding, ConceptEmbedding)

    def test_init_with_invalid_callable(self):
        """Test initialization with an invalid embedding function."""
        with self.assertRaises(ValueError):
            ConceptEmbedding(embed_func="not_callable")

    def test_parse_conceptualized_nodes(self):
        """Test parsing conceptualized nodes."""
        concept_data = [
            {"conceptualized_node": "node1, node2"},
            {"conceptualized_node": "node2, node3"},
            {"conceptualized_node": ""},
        ]
        expected = {"node1", "node2", "node3"}
        result = self.concept_embedding._parse_conceptualized_nodes(concept_data)
        self.assertEqual(result, expected)

    def test_parse_conceptualized_nodes_empty(self):
        """Test parsing with empty concept data."""
        concept_data = []
        result = self.concept_embedding._parse_conceptualized_nodes(concept_data)
        self.assertEqual(result, set())

    def test_extract_concepts(self):
        """Test extracting sorted unique concepts."""
        concept_data = [
            {"conceptualized_node": "node2, node1"},
            {"conceptualized_node": "node3"},
        ]
        expected = ["node1", "node2", "node3"]
        result = self.concept_embedding.extract_concepts(concept_data)
        self.assertEqual(result, expected)

    def test_extract_concepts_empty(self):
        """Test extracting concepts from empty concept data."""
        concept_data = []
        result = self.concept_embedding.extract_concepts(concept_data)
        self.assertEqual(result, [])

    def test_embed(self):
        """Test embedding generation."""
        concept_data = [
            {"conceptualized_node": "node1, node2"},
        ]
        expected = {"node1": [0.1, 0.2], "node2": [0.3, 0.4]}
        result = self.concept_embedding.embed(concept_data)
        self.assertEqual(result, expected)
        self.mock_embed_func.assert_called_once_with(["node1", "node2"], batch_size=1)

    def test_embed_empty_concepts(self):
        """Test embedding with empty concept data."""
        concept_data = []
        result = self.concept_embedding.embed(concept_data)
        self.assertEqual(result, {})
        self.mock_embed_func.assert_not_called()

    def test_embed_mismatched_embeddings(self):
        """Test embedding with mismatched embedding function output."""
        self.mock_embed_func.return_value = [[0.1, 0.2]]  # Mismatched length
        concept_data = [
            {"conceptualized_node": "node1, node2"},
        ]
        with self.assertRaises(ValueError):
            self.concept_embedding.embed(concept_data)
