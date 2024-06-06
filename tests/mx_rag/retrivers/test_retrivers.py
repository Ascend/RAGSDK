# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from transformers import is_torch_npu_available

if not is_torch_npu_available():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, "../../src/vectorstore/"))

from loguru import logger
from transformers import is_torch_npu_available

from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.retrievers import Retriever
from mx_rag.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage import Document, SQLiteDocstore

EMBEDDING_TEXT = """The unshare command creates new namespaces and then executes the specified program."""


class MyTestCase(unittest.TestCase):
    sql_db_file = "/tmp/sql.db"

    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

    def test_Retriever_npu(self):
        if not is_torch_npu_available():
            logger.info("skip npu case")
            return

        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore("/tmp/sql.db")
        logger.info("create emb done")
        MindFAISS.set_device(0)
        logger.info("set_device done")
        vector_store = MindFAISS(x_dim=1024, index_type="FLAT:L2", document_store=db, embed_func=emb.encode)
        vector_store.add_texts("unshare_desc.txt", [EMBEDDING_TEXT])
        logger.info("create MindFAISS done")
        r = Retriever(vector_store, score_threshold=0.5)

        def test_result(self):
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query)
            logger.info(f"relevant doc {x}")
            self.assertEqual(f"{EMBEDDING_TEXT}\n{query}", x)

        def test_result_with_prompt(self):
            prompt = "haha"
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query, prompt)
            logger.info(f"relevant doc {x}")
            self.assertEqual(f"{EMBEDDING_TEXT}\n{prompt}\n{query}", x)

        def test_no_result(self):
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query)
            logger.info(f"relevant doc {x}")
            self.assertEqual(query, x)

        def test_no_result_with_prompt(self):
            prompt = "haha"
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query, prompt)
            logger.info(f"relevant doc {x}")
            self.assertEqual(f"{query}", x)

        test_result(self)
        test_result_with_prompt(self)
        test_no_result(self)
        test_no_result_with_prompt(self)

    def test_Retriever(self):
        if is_torch_npu_available():
            logger.info("skip none npu case")
            return

        def embed_func(texts):
            return np.random.random((1, 1024))

        db = SQLiteDocstore("sql.db")
        MindFAISS.DEVICES = MagicMock()
        vector_store = MindFAISS(x_dim=1024, index_type="FLAT:L2", document_store=db, embed_func=embed_func)
        vector_store.similarity_search = MagicMock(
            return_value=[(Document(page_content="this is a test", document_name="test.txt"), 0.5)])

        r = Retriever(vector_store, score_threshold=0.5)

        def test_result(self):
            vector_store.similarity_search = MagicMock(
                return_value=[(Document(page_content=EMBEDDING_TEXT, document_name="test.txt"), 0.5)])
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query)
            logger.info(f"relevant doc {x}")
            self.assertEqual(f"{EMBEDDING_TEXT}\n\n{query}", x)

        def test_result_with_prompt(self):
            vector_store.similarity_search = MagicMock(
                return_value=[(Document(page_content=EMBEDDING_TEXT, document_name="test.txt"), 0.5)])
            prompt = "haha"
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query, prompt)
            logger.info(f"relevant doc {x}")
            self.assertEqual(f"{EMBEDDING_TEXT}\n\n{prompt}\n\n{query}", x)

        def test_no_result(self):
            vector_store.similarity_search = MagicMock(
                return_value=[(Document(page_content=EMBEDDING_TEXT, document_name="test.txt"), 0.6)])
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query)
            logger.info(f"relevant doc {x}")
            self.assertEqual(query, x)

        def test_no_result_with_prompt(self):
            vector_store.similarity_search = MagicMock(
                return_value=[(Document(page_content=EMBEDDING_TEXT, document_name="test.txt"), 0.6)])
            prompt = "haha"
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            x = r.get_relevant_documents(query, prompt)
            logger.info(f"relevant doc {x}")
            self.assertEqual(f"{query}", x)

        test_result(self)
        test_result_with_prompt(self)
        test_no_result(self)
        test_no_result_with_prompt(self)


if __name__ == '__main__':
    unittest.main()
