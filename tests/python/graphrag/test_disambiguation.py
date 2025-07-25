# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import Mock, patch
from concurrent.futures import Future

import networkx as nx

from mx_rag.utils.common import Lang
from mx_rag.graphrag.disambiguation import EntityDisambiguator, EntityContextManager
from mx_rag.graphrag.graphs.graph_store import GraphStore


class TestEntityDisambiguator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_llm.chat.return_value = "yes"
        self.model_name = "test_model"
        self.mock_graph = Mock(spec=GraphStore)
        self.max_workers = 2

        self.disambiguator = EntityDisambiguator(
            self.mock_llm,
            self.model_name,
            self.mock_graph,
            self.max_workers
        )

    def test_init(self):
        """Test EntityDisambiguator initialization."""
        self.assertEqual(self.disambiguator.llm, self.mock_llm)
        self.assertEqual(self.disambiguator.model_name, self.model_name)
        self.assertEqual(self.disambiguator.graph, self.mock_graph)
        self.assertEqual(self.disambiguator.max_workers, self.max_workers)

    @patch('mx_rag.graphrag.disambiguation.tqdm')
    def test_disambiguate_entities(self, mock_tqdm):
        """Test disambiguate_entities method."""
        mock_tqdm.return_value = ["entity1", "entity2"]
        mock_context_manager = Mock(spec=EntityContextManager)
        named_entity_nodes = ["entity1", "entity2"]
        lang = Lang.EN

        with patch.object(self.disambiguator, '_process_single_entity') as mock_process:
            self.disambiguator.disambiguate_entities(named_entity_nodes, mock_context_manager, lang)

            self.assertEqual(mock_process.call_count, 2)
            mock_process.assert_any_call("entity1", mock_context_manager, lang)
            mock_process.assert_any_call("entity2", mock_context_manager, lang)

    def test_process_single_entity_single_file_id(self):
        """Test _process_single_entity with single file ID (no processing needed)."""
        self.mock_graph.get_node_attributes.return_value = "file1"
        mock_context_manager = Mock(spec=EntityContextManager)

        with patch.object(self.disambiguator, '_build_context_graph') as mock_build:
            self.disambiguator._process_single_entity("entity1", mock_context_manager, Lang.EN)
            mock_build.assert_not_called()

    @patch('mx_rag.graphrag.disambiguation.nx.connected_components')
    @patch('mx_rag.graphrag.disambiguation.logger')
    def test_process_single_entity_multiple_components(self, mock_logger, mock_connected_components):
        """Test _process_single_entity with multiple connected components."""
        self.mock_graph.get_node_attributes.return_value = "file1,file2,file3"
        mock_context_manager = Mock(spec=EntityContextManager)

        # Mock connected components
        mock_context_graph = Mock()
        mock_connected_components.return_value = [{"file1", "file2"}, {"file3"}]

        with patch.object(self.disambiguator, '_build_context_graph', return_value=mock_context_graph):
            with patch.object(self.disambiguator, '_split_node_by_components') as mock_split:
                self.disambiguator._process_single_entity("entity1", mock_context_manager, Lang.EN)

                mock_split.assert_called_once_with("entity1", [{"file1", "file2"}, {"file3"}])
                mock_logger.info.assert_called_once_with("Disambiguating entity1...")

    def test_build_context_graph_no_pairs(self):
        """Test _build_context_graph with no valid context pairs."""
        mock_context_manager = Mock(spec=EntityContextManager)
        mock_context_manager.get_context.return_value = None

        result = self.disambiguator._build_context_graph("entity1", ["file1", "file2"], mock_context_manager, Lang.EN)

        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(list(result.nodes()), ["file1", "file2"])
        self.assertEqual(list(result.edges()), [])

    def test_build_context_graph_with_similar_entities(self):
        """Test _build_context_graph with similar entities."""
        mock_context_manager = Mock(spec=EntityContextManager)
        mock_context_manager.get_context.side_effect = lambda entity, file_id: f"context_{file_id}"

        with patch.object(self.disambiguator, '_check_entity_similarity', return_value=[True]):
            result = self.disambiguator._build_context_graph(
                "entity1", ["file1", "file2"], mock_context_manager, Lang.EN)

            self.assertIsInstance(result, nx.Graph)
            self.assertTrue(result.has_edge("file1", "file2"))

    @patch('mx_rag.graphrag.disambiguation.ENTITY_PROMPT_EN', "test prompt [NODE1] [CONTEXT1] [NODE2] [CONTEXT2]")
    def test_check_entity_similarity(self):
        """Test _check_entity_similarity method."""
        node_pairs = [("entity1", "context1", "entity1", "context2")]

        with patch.object(self.disambiguator, '_process_parallel_similarity_checks', return_value=["yes"]):
            result = self.disambiguator._check_entity_similarity(node_pairs, Lang.EN)

            self.assertEqual(result, [True])

    @patch('mx_rag.graphrag.disambiguation.ThreadPoolExecutor')
    @patch('mx_rag.graphrag.disambiguation.as_completed')
    @patch('mx_rag.graphrag.disambiguation.tqdm')
    def test_process_parallel_similarity_checks(self, mock_tqdm, mock_as_completed, mock_executor):
        """Test _process_parallel_similarity_checks method."""
        # Setup mocks
        mock_future = Mock(spec=Future)
        mock_future.result.return_value = "yes\n"
        mock_as_completed.return_value = [mock_future]
        mock_tqdm.return_value = [mock_future]

        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        prompts = ["test prompt"]
        result = self.disambiguator._process_parallel_similarity_checks(prompts)

        self.assertEqual(result, ["yes"])

    def test_split_node_by_components(self):
        """Test _split_node_by_components method."""
        connected_components = [{"file1", "file2"}, {"file3"}]

        with patch.object(self.disambiguator, '_create_split_node') as mock_create_node:
            with patch.object(self.disambiguator, '_create_edges_for_split_node', 
                            return_value=[("new_node", "target", {})]) as mock_create_edges:
                self.disambiguator._split_node_by_components("entity1", connected_components)

                # Verify split nodes were created
                self.assertEqual(mock_create_node.call_count, 2)
                
                # Check calls by examining the actual arguments, accounting for set ordering
                calls = mock_create_node.call_args_list
                
                # First call should be for component 1
                first_call = calls[0]
                self.assertEqual(first_call[0][0], "entity1")  # original_node
                self.assertEqual(first_call[0][1], "entity1:::1")  # new_node
                self.assertEqual(set(first_call[0][2]), {"file1", "file2"})  # component_list as set

                # Second call should be for component 2
                second_call = calls[1]
                self.assertEqual(second_call[0][0], "entity1")  # original_node
                self.assertEqual(second_call[0][1], "entity1:::2")  # new_node
                self.assertEqual(set(second_call[0][2]), {"file3"})  # component_list as set

                # Verify graph operations
                self.mock_graph.add_edges_from.assert_called_once()
                self.mock_graph.remove_node.assert_called_once_with("entity1")

    def test_create_split_node(self):
        """Test _create_split_node method."""
        self.mock_graph.get_node_attributes.return_value = "file1,file2,file3"
        self.mock_graph.has_node.return_value = False

        self.disambiguator._create_split_node("original_node", "new_node", ["file1", "file2"])

        self.mock_graph.add_node.assert_called_once_with("new_node")
        self.mock_graph.update_node_attribute.assert_any_call("new_node", "type", "named_entity")
        self.mock_graph.update_node_attribute.assert_any_call("new_node", "file_id", "file1,file2")

    def test_create_edges_for_split_node(self):
        """Test _create_edges_for_split_node method."""
        self.mock_graph.successors.return_value = ["tail1", "tail2"]
        self.mock_graph.predecessors.return_value = ["head1"]

        with patch.object(self.disambiguator, '_build_new_edge') as mock_build_edge:
            mock_build_edge.return_value = ("new_node", "target", {"relation": "test"})

            result = self.disambiguator._create_edges_for_split_node("original_node", "new_node", ["file1"])

            # Should call _build_new_edge for successors and predecessors
            self.assertEqual(mock_build_edge.call_count, 3)  # 2 successors + 1 predecessor

    def test_build_new_edge_outgoing(self):
        """Test _build_new_edge for outgoing edge."""
        self.mock_graph.get_edge_attributes.side_effect = (
            lambda src, dst, attr: "file1,file2" if attr == "file_id" else "test_relation"
        )
        self.mock_graph.get_node_attributes.return_value = "file1,file2,file3"

        result = self.disambiguator._build_new_edge("head", "tail", "new_node", ["file1"], outgoing=True)

        expected = ("new_node", "tail", {"relation": "test_relation", "file_id": "file1"})
        self.assertEqual(result, expected)

    def test_build_new_edge_incoming(self):
        """Test _build_new_edge for incoming edge."""
        self.mock_graph.get_edge_attributes.side_effect = (
            lambda src, dst, attr: "file1,file2" if attr == "file_id" else "test_relation"
        )
        self.mock_graph.get_node_attributes.return_value = "file1,file2,file3"

        result = self.disambiguator._build_new_edge("head", "tail", "new_node", ["file1"], outgoing=False)

        expected = ("head", "new_node", {"relation": "test_relation", "file_id": "file1"})
        self.assertEqual(result, expected)

    def test_build_new_edge_no_matching_files(self):
        """Test _build_new_edge with no matching file IDs."""
        self.mock_graph.get_edge_attributes.return_value = "file3,file4"  # No overlap with component

        result = self.disambiguator._build_new_edge("head", "tail", "new_node", ["file1"], outgoing=True)

        self.assertEqual(result, (None, None, None))

    @patch('mx_rag.graphrag.disambiguation.logger')
    def test_build_new_edge_exception_handling(self, mock_logger):
        """Test _build_new_edge exception handling."""
        self.mock_graph.get_edge_attributes.side_effect = Exception("Test exception")

        result = self.disambiguator._build_new_edge("head", "tail", "new_node", ["file1"], outgoing=True)

        self.assertEqual(result, (None, None, None))
        mock_logger.error.assert_called_once()

    @patch('mx_rag.graphrag.disambiguation.logger')
    def test_process_parallel_similarity_checks_with_exception(self, mock_logger):
        """Test _process_parallel_similarity_checks with LLM exception."""
        with patch('mx_rag.graphrag.disambiguation.ThreadPoolExecutor') as mock_executor:
            with patch('mx_rag.graphrag.disambiguation.as_completed') as mock_as_completed:
                with patch('mx_rag.graphrag.disambiguation.tqdm') as mock_tqdm:
                    # Setup mocks
                    mock_future = Mock(spec=Future)
                    mock_future.result.side_effect = Exception("LLM error")
                    mock_as_completed.return_value = [mock_future]
                    mock_tqdm.return_value = [mock_future]

                    mock_executor_instance = Mock()
                    mock_executor_instance.submit.return_value = mock_future
                    mock_executor.return_value.__enter__.return_value = mock_executor_instance

                    result = self.disambiguator._process_parallel_similarity_checks(["test prompt"])

                    self.assertEqual(result, [""])
                    mock_logger.error.assert_called_once()
