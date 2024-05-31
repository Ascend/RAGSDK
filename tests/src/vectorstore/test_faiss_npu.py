# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import builtins
import os.path
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np

with patch("mx_rag.vectorstore.faiss_npu.ascendfaiss"):
    with patch("mx_rag.vectorstore.faiss_npu.faiss"):
        from mx_rag.vectorstore.faiss_npu import MindFAISS
        from mx_rag.vectorstore.storage import SQLiteDocstore, Document


class TestAscendFAISS(unittest.TestCase):
    def test_faiss(self):
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
        index = MindFAISS(1024, "FLAT:L2", SQLiteDocstore("./sql.db"), embed_func)
        texts = ["1111", "2222", "3333"]

        index.add_texts("test.txt", texts, metadatas=[{"name": "yiyiyi"}, {"name": "ererere"}, {"name": "sansansan"}])
        index.index.search = MagicMock(return_value=([[0.1]], [[np.array(0)]]))
        index.document_store.search = MagicMock(return_value=Document(page_content="1111", document_name="test.txt"))
        ret = index.similarity_search(["1111"], k=1)
        self.assertEqual(ret[0][0].page_content, "1111")

        index.similarity_search_by_vector = MagicMock(
            return_value=[[Document(page_content="1111", document_name="test.txt")]])
        ret = index.similarity_search_by_vector(query, k=1)
        self.assertEqual(ret[0][0].page_content, "1111")

        texts = ["4444", "5555", "6666"]
        index.add_texts("test-2.txt", texts, metadatas=[{"name": "sisisi"}, {"name": "wuwuwu"}, {"name": "liuliuliu"}])
        index.index.remove_ids = MagicMock(return_value=len(texts))
        index.delete("test.txt")
        index.save_local("./faiss.index")
        index2 = MindFAISS.load_local = MagicMock(return_value=index)
        index2.similarity_search_by_vector = MagicMock(
            return_value=[[Document(page_content="4444", document_name="test-2.txt")]])
        ret = index2.similarity_search_by_vector(query, k=1)
        self.assertEqual(ret[0][0].page_content, "4444")
        MindFAISS.clear_device()