# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os.path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestMindFAISS(unittest.TestCase):
    def test_faiss(self):
        with patch("mx_rag.storage.vectorstore.faiss_npu.ascendfaiss") as ascendfaiss:
            with patch("mx_rag.storage.vectorstore.faiss_npu.faiss") as faiss:
                from mx_rag.storage.vectorstore.faiss_npu import MindFAISS, SimilarityStrategy

                total = np.random.random((3, 1024))
                query = np.array([total[0]])

                def embed_func(texts):
                    if len(texts) > 1:
                        return total
                    return query

                os.system = MagicMock(return_value=0)
                os.chmod = MagicMock()
                index = MindFAISS(1024, similarity_strategy=SimilarityStrategy.FLAT_L2, devs=[0],
                                  load_local_index="./faiss.index")
                index.search(query, k=1)
                index.add(query, [1])
                index.delete([1])
                index.save_local()
