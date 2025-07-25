# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
from unittest.mock import Mock
import numpy as np
from mx_rag.graphrag.concept_clustering import ConceptCluster


class TestConceptCluster(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_vector_store = Mock()
        self.mock_graph = Mock()
        self.cluster = ConceptCluster(self.mock_vector_store, self.mock_graph)

    def test_initialization(self):
        """Test that ConceptCluster initializes correctly."""
        self.assertEqual(self.cluster.vector_store, self.mock_vector_store)
        self.assertEqual(self.cluster.graph, self.mock_graph)

    def test_build_edges(self):
        """Test _build_edges method."""
        concept_names = ["concept1", "concept2", "concept3"]
        distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.4, 0.5]])
        indices = np.array([[1, 2], [0, 2], [0, 1]])
        threshold = 0.15
        edges = self.cluster._build_edges(concept_names, distances, indices, threshold)
        self.assertEqual(
            edges,
            [
                ("concept1", "concept3"),
                ("concept2", "concept3"),
                ("concept3", "concept1"),
                ("concept3", "concept2"),
            ],
        )

    def test_find_clusters(self):
        """Test find_clusters method."""
        concept_embeddings = {
            "concept1": np.array([1.0, 2.0]),
            "concept2": np.array([3.0, 4.0]),
        }
        self.mock_vector_store.search.return_value = (np.array([[0.1, 0.2]]), np.array([[1, 0]]))
        self.mock_graph.add_edges_from.return_value = None
        self.mock_graph.connected_components.return_value = [["concept1", "concept2"]]

        clusters = self.cluster.find_clusters(concept_embeddings, top_k=2, threshold=0.15)
        self.assertEqual(clusters, [["concept1", "concept2"]])
        self.mock_vector_store.search.assert_called_once()
        self.mock_graph.add_edges_from.assert_called_once()
        self.mock_graph.connected_components.assert_called_once()

    def test_find_clusters_with_empty_embeddings(self):
        """Test find_clusters with empty concept_embeddings."""
        clusters = self.cluster.find_clusters({})
        self.assertEqual(clusters, [])
