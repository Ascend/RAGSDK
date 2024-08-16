# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from transformers import is_torch_npu_available

if not is_torch_npu_available():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, "../vectorstore/"))

from loguru import logger
from mx_rag.knowledge.knowledge import KnowledgeStore
from langchain_core.documents import Document
from mx_rag.knowledge import KnowledgeDB
from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import MultiQueryRetriever
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore


class MyTestCase(unittest.TestCase):

    def test_MultiQueryRetriever_npu(self):
        if not is_torch_npu_available():
            return
        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore("/tmp/sql.db")
        logger.info("create emb done")
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2", load_local_index="./faiss.index")

        knowledge_db = KnowledgeDB(KnowledgeStore("./sql.db"), db, index, "test", white_paths=["/home"])
        knowledge_db.add_file("test_file.txt", ["this is a test"], embed_func=emb.embed_documents)
        logger.info("create MindFAISS done")
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", base_url="http://71.14.88.12:7890")

        r = MultiQueryRetriever(llm=llm, vector_store=index, document_store=db, embed_func=emb.embed_documents)
        doc = r.invoke("what is test?")

        self.assertEqual("this is a test", doc[0].page_content)

    @patch("mx_rag.retrievers.multi_query_retriever.MultiQueryRetriever._get_relevant_documents")
    def test_MultiQueryRetrieverBase(self, get_relevant_documents_mock):
        if is_torch_npu_available():
            return

        def embed_func(texts):
            return np.random.random((1, 1024))

        mind_llm = Text2TextLLM(model_name="chatglm2-6b-quant", base_url="http://127.0.0.1:7890")


        get_relevant_documents_mock.return_value = [Document(page_content="this is a test", metadata={})]
        shutil.disk_usage = MagicMock(return_value=(1, 1, 1000 * 1024 * 1024))
        db = SQLiteDocstore("sql.db")
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2", load_local_index="./faiss.index")

        r = MultiQueryRetriever(llm=mind_llm, vector_store=vector_store, document_store=db, embed_func=embed_func)

        doc = r.invoke("what is test?")
        logger.info(f"relevant doc {doc}")
        self.assertEqual("this is a test", doc[0].page_content)

    @patch("mx_rag.retrievers.multi_query_retriever.MultiQueryRetriever._get_relevant_documents")
    def test_MultiQueryRetrieverMulti(self, get_relevant_documents_mock):
        if is_torch_npu_available():
            return

        def embed_func(texts):
            return np.random.random((1, 1024))

        mind_llm = Text2TextLLM(model_name="chatglm2-6b-quant", base_url="http://127.0.0.1:7890")

        get_relevant_documents_mock.return_value = [Document(page_content="this is a test", metadata={})]
        db = SQLiteDocstore("sql.db")
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2", load_local_index="./faiss.index")

        r = MultiQueryRetriever(llm=mind_llm, vector_store=vector_store, document_store=db, embed_func=embed_func, k=10)
        doc = r.invoke("what is test?")
        logger.info(f"relevant doc {doc}")
        self.assertEqual("this is a test", doc[0].page_content)


if __name__ == '__main__':
    unittest.main()
