import os
import unittest

from mx_rag.knowledge.handler import load_tree, save_tree
from mx_rag.retrievers.tree_retriever.src.tree_retriever import TreeRetrieverConfig, TreeRetriever

SAVE_PATH = "./tree.json"


class Test(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.tree_file = os.path.join(current_dir, "../../data/tree.json")
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
        self.tree = load_tree(self.tree_file)

    def test_load_tree(self):
        self.assertEqual(3, len(self.tree.all_nodes))
        self.assertEqual({1, 25}, self.tree.root_nodes[0].children)
        config = TreeRetrieverConfig()
        TreeRetriever(config, self.tree)

    def test_save_tree(self):
        save_tree(self.tree, SAVE_PATH)
        self.assertTrue(os.path.exists(SAVE_PATH))

    def tearDown(self):
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
