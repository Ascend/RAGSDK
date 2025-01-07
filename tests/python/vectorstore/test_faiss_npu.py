# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os.path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from mx_rag.storage.vectorstore import MindFAISS, VectorStore
from mx_rag.storage.vectorstore.faiss_npu import MindFAISSError


class TestMindFAISS(unittest.TestCase):
    def test_faiss(self):
        with patch("mx_rag.storage.vectorstore.faiss_npu.ascendfaiss") as ascendfaiss:
            with patch("mx_rag.storage.vectorstore.faiss_npu.faiss") as faiss:
                from mx_rag.storage.vectorstore.vectorstore import SimilarityStrategy

                total = np.random.random((3, 1024))
                query = np.array([total[0]])

                def embed_func(texts):
                    if len(texts) > 1:
                        return total
                    return query

                os.system = MagicMock(return_value=0)
                os.chmod = MagicMock()

                with self.assertRaises(KeyError):
                    index = MindFAISS.create(similarity_strategy=SimilarityStrategy.FLAT_L2, devs=[0],
                                             load_local_index="./faiss.index")
                with self.assertRaises(KeyError):
                    index = MindFAISS.create(x_dim=1024, devs=[0], load_local_index="./faiss.index")
                with self.assertRaises(KeyError):
                    index = MindFAISS.create(x_dim=1024, similarity_strategy=SimilarityStrategy.FLAT_L2,
                                             load_local_index="./faiss.index")
                with self.assertRaises(KeyError):
                    index = MindFAISS.create(x_dim=1024, similarity_strategy=SimilarityStrategy.FLAT_L2, devs=[0])
                with self.assertRaises(MindFAISSError):
                    index = MindFAISS(x_dim=1024, similarity_strategy=SimilarityStrategy.FLAT_L2, devs=0,
                                      load_local_index="./faiss.index")
                with self.assertRaises(MindFAISSError):
                    index = MindFAISS.create(x_dim=1024, similarity_strategy=SimilarityStrategy.FLAT_L2, devs=[0, 1],
                                             load_local_index="./faiss.index")

                index = MindFAISS.create(x_dim=1024, similarity_strategy=SimilarityStrategy.FLAT_L2, devs=[0],
                                         load_local_index="./faiss.index")

                index.search(query, k=1)
                with self.assertRaises(MindFAISSError):
                    vecs = np.random.randn(3, 2, 1024)
                    index.search(vecs)

                index.add(query, [1])
                with self.assertRaises(MindFAISSError):
                    vecs = np.random.randn(3, 2, 1024)
                    index.add(vecs, [0, 1, 2])
                with self.assertRaises(MindFAISSError):
                    vecs = np.random.randn(2, 1024)
                    index.add(vecs, [0, 1, 2])
                with patch.object(VectorStore, 'MAX_VEC_NUM', 1):
                    with self.assertRaises(MindFAISSError):
                        vecs = np.random.randn(3, 1024)
                        index.add(vecs, [0, 1, 2])
                    with self.assertRaises(MindFAISSError):
                        index.delete([1, 2, 3])
                with patch.object(VectorStore, 'MAX_SEARCH_BATCH', 1):
                    with self.assertRaises(MindFAISSError):
                        vecs = np.random.randn(3, 1024)
                        index.search(vecs)
                with self.assertRaises(AttributeError):
                    index.get_all_ids()
                index.delete([1])
                index.save_local()
                index.get_save_file()
