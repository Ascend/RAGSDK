# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os.path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestMindFAISS(unittest.TestCase):
    def test_faiss(self):
        with patch("mx_rag.vectorstore.faiss_npu.ascendfaiss") as ascendfaiss:
            with patch("mx_rag.vectorstore.faiss_npu.faiss") as faiss:
                from mx_rag.vectorstore.faiss_npu import MindFAISS

                total = np.random.random((3, 1024))
                query = np.array([total[0]])

                def embed_func(texts):
                    if len(texts) > 1:
                        return total
                    return query

                os.chmod = MagicMock()
                MindFAISS.set_device = MagicMock()
                MindFAISS.set_device(0)
                MindFAISS.DEVICES = MagicMock()
                index = MindFAISS(1024, "FLAT:L2")
                index.search(query, k=1)
                index.add(query, [1])
                index.delete([1])
                index.save_local("./faiss.index")
                MindFAISS.load_local("./faiss.index")
                MindFAISS.clear_device()
