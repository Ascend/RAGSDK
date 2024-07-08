import unittest
from unittest.mock import MagicMock

import numpy as np
from transformers import PreTrainedTokenizerBase

from mx_rag.chain import TreeText2TextChain
from mx_rag.retrievers import TreeBuilderConfig, TreeBuilder


class TestTreeBuilder(unittest.TestCase):
    def test_exception_parameters(self):
        with self.assertRaises(ValueError):
            TreeBuilderConfig()
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        with self.assertRaises(ValueError):
            TreeBuilderConfig(tokenizer, 0)

    def test_create_node(self):
        embed_func = MagicMock(return_value=np.array([1, 2, 3]))
        index, node = TreeBuilder.create_node(0, "test", embed_func)
        self.assertEqual(0, index)
        self.assertEqual("test", node.text)

    def test_summarize(self):
        tree_builder = MagicMock(spes=TreeBuilder)
        tree_builder.summarize = MagicMock(return_value="test")
        self.assertEqual("test", tree_builder.summarize("test"))

    def test_build_from_text(self, build_from_text=None):
        tree = TestTreeBuilder.mock_tree()
        self.assertEqual(['test1', 'test2'], [node.text for node in tree.all_nodes.values()])

    @staticmethod
    def mock_tree():
        embed_func = MagicMock(return_value=np.array([1, 2, 3]))
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        tree_build_config = TreeBuilderConfig(tokenizer, summarization_model=MagicMock(spec=TreeText2TextChain))
        tree_builder = TreeBuilder(tree_build_config)
        return tree_builder.build_from_text(embed_func, ["test1", "test2"])
