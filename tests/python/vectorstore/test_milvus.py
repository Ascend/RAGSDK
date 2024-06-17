# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os.path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestMlvusClient(unittest.TestCase):
    def test_faiss(self):
        with patch("pymilvus.MilvusClient") as MilvusClient:
            from mx_rag.vectorstore import MilvusDB
            embeddings = np.random.random((3, 1024))
            query = embeddings[0]
            my_milvus = MagicMock()
            my_milvus.set_collection_name("test_ccc")
            my_milvus.client.has_collection = MagicMock(return_value=False)
            my_milvus.create_collection(768, "FLAT", "IP")
            my_milvus.client.has_collection = MagicMock(return_value=True)
            my_milvus.add(embeddings, [0, 1, 2])
            res = my_milvus.search(query)
            my_milvus.delete([0, 1, 2])
            my_milvus.drop_collection()
