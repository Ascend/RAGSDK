import os
import unittest
import numpy as np

from pathlib import Path
from mx_rag.knowledge.handler import load_tree, save_tree

SAVE_PATH = "./tree.json"


class TestHandler(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.tree_file = str(Path(os.path.join(current_dir, "../../data/tree.json")).resolve())
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
        self.tree = load_tree(self.tree_file, [current_dir.split("tests")[0]], np.float16)

    def test_load_tree(self):
        self.assertEqual(3, len(self.tree.all_nodes))
        self.assertEqual({1, 25}, self.tree.root_nodes[0].children)

    def test_save_tree(self):
        save_tree(self.tree, SAVE_PATH)
        self.assertTrue(os.path.exists(SAVE_PATH))

    def tearDown(self):
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
