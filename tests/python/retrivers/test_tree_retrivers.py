import unittest
from unittest.mock import MagicMock

import numpy as np
from transformers import PreTrainedTokenizerBase

from mx_rag.chain import TreeText2TextChain
from mx_rag.retrievers import TreeRetrieverConfig, TreeRetriever, TreeBuilderConfig, TreeBuilder
from mx_rag.retrievers.tree_retriever import Node


class TestTreeRetriever(unittest.TestCase):
    def setUp(self):
        tree = TestTreeRetriever.mock_tree()
        embed_func = MagicMock(return_value=np.array([1, 2, 3]))
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        tree_retriver_config = TreeRetrieverConfig(tokenizer=tokenizer, embed_func=embed_func)
        self.tree_retriever = TreeRetriever(tree_retriver_config, tree)

    def test_exception_parameters(self):
        with self.assertRaises(TypeError):
            TreeRetrieverConfig()
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        with self.assertRaises(ValueError):
            TreeRetrieverConfig(tokenizer, None)

    def test_create_embedding(self):
        result = self.tree_retriever._create_embedding("test")
        self.assertEqual([1, 2, 3], result)

    def test_retrieve_information_collapse_tree(self):
        selected_nodes, context = self.tree_retriever._retrieve_information_collapse_tree("hello", 2, 5)
        self.assertEqual("test1\n\ntest2\n\n", context)
        self.assertEqual(2, len(selected_nodes))

    def test_retrieve_information(self):
        node_list = [Node("test1", 0, {0}, {"filepath": "xxx"}), Node("test2", 1, {1}, [1, 2, 3])]
        selected_nodes, context = self.tree_retriever._retrieve_information(node_list, "test", 0)
        self.assertEqual("", context)
        self.assertEqual([], selected_nodes)

    @staticmethod
    def mock_tree():
        embed_func = MagicMock(return_value=np.array([1, 2, 3]))
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        tree_build_config = TreeBuilderConfig(tokenizer, summarization_model=MagicMock(spec=TreeText2TextChain))
        tree_builder = TreeBuilder(tree_build_config)
        return tree_builder.build_from_text(embed_func, ["test1", "test2"])
