# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from transformers import is_torch_npu_available

from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.knowledge import KnowledgeDB
from langchain_core.documents import Document

if not is_torch_npu_available():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, "../vectorstore/"))

from loguru import logger
from transformers import is_torch_npu_available

from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.retrievers import Retriever
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore

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
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2", load_local_index="./faiss.index")
        knowledge_db = KnowledgeDB(KnowledgeStore("./sql.db"), db, index, "test", white_paths=["/home"])
        knowledge_db.add_file("unshare_desc.txt", [EMBEDDING_TEXT], embed_func=emb.embed_texts)
        logger.info("create MindFAISS done")
        r = Retriever(index, document_store= db, score_threshold=0.5, embed_func=emb.embed_texts)

        def test_result(self):
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_result_with_prompt(self):
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_no_result(self):
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(query, docs[0].page_content)

        def test_no_result_with_prompt(self):
            prompt = "haha"
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(query, docs[0].page_content)

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

        shutil.disk_usage = MagicMock(return_value=(1, 1, 1000 * 1024 * 1024))
        db = SQLiteDocstore("sql.db")
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2", load_local_index="./faiss.index")

        r = Retriever(vector_store, document_store= db, score_threshold=0.5, embed_func=embed_func)

        def test_result(self):
            r._get_relevant_documents = MagicMock(
                return_value=[Document(page_content=EMBEDDING_TEXT, metadata={})])
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_result_with_prompt(self):
            r._get_relevant_documents = MagicMock(
                return_value=[Document(page_content=EMBEDDING_TEXT, metadata={})])
            prompt = "haha"
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_no_result(self):
            r._get_relevant_documents = MagicMock(
                return_value=[Document(page_content=EMBEDDING_TEXT, metadata={})])
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_no_result_with_prompt(self):
            r._get_relevant_documents = MagicMock(
                return_value=[Document(page_content=EMBEDDING_TEXT, metadata={})])
            prompt = "haha"
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents [{query}]")
            docs = r.get_relevant_documents(query)

            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        test_result(self)
        test_result_with_prompt(self)
        test_no_result(self)
        test_no_result_with_prompt(self)


if __name__ == '__main__':
    unittest.main()
