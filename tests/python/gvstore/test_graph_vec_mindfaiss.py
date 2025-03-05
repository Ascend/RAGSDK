# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from paddle.base import libpaddle
import unittest
import numpy as np
from typing import List
from unittest.mock import patch
from networkx import DiGraph
from mx_rag.embedding.local import TextEmbedding
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.gvstore.graph_creator.vdb.vector_db import GraphVecMindfaissDB


class PatchEmbedding():
    def __init__(self):
        pass

    @property
    def __class__(self):
        return TextEmbedding

    def embed_documents(self, texts: List[str], in_batch: int = 1):
        batch = len(texts)
        batch = batch if batch > 0 else in_batch
        return np.random.random(size=(batch, 1024)).tolist()


class PatchNodes():
    def data(self):
        return [[0, {"id": 1, "info": "这是一个节点", "label": "text"}]]


class PatchGraph():
    def __init__(self):
        self.nodes = PatchNodes()

    @property
    def __class__(self):
        return DiGraph


class TestGvStore(unittest.TestCase):

    @patch("mx_rag.storage.vectorstore.faiss_npu.MindFAISS.search")
    @patch("mx_rag.storage.vectorstore.faiss_npu.MindFAISS.add")
    @patch("mx_rag.storage.vectorstore.faiss_npu.MindFAISS.__init__")
    @patch("mx_rag.embedding.local.TextEmbedding")
    def test_create_index(self, patch_emb, mindfaiss_init, mindfaiss_add, mindfaiss_search):
        patch_emb = PatchEmbedding
        emb = patch_emb()
        mindfaiss_init.return_value = None
        mindfaiss_add.return_value = None
        mindfaiss_search.return_value = [0.1, 0.1, 0.1], [[0, 1, 2]]
        mind_faiss = MindFAISS()
        graph_vec_db = GraphVecMindfaissDB(mind_faiss=mind_faiss, embedding_model=emb, db_path="./sql.db")
        graph = PatchGraph()
        graph_vec_db.create_index(graph)
        graph_vec_db.get_data([0])
        graph_vec_db.search_indexes("这是一个节点", k=3)


if __name__ == '__main__':
    unittest.main()
