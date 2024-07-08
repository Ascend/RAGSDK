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
    sys.path.insert(0, os.path.join(cur_dir, "../vectorstore/"))

from loguru import logger
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.document.doc import Doc
from mx_rag.knowledge import KnowledgeDB
from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import MultiQueryRetriever
from mx_rag.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage import SQLiteDocstore, Document


class MyTestCase(unittest.TestCase):

    def test_MultiQueryRetriever_npu(self):
        if not is_torch_npu_available():
            return
        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore("/tmp/sql.db")
        logger.info("create emb done")
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2")

        vector_store = KnowledgeDB(KnowledgeStore("./sql.db"), db, index, "test", white_paths=["/home"])
        vector_store.add_file("test_file.txt", ["this is a test"], embed_func=emb.embed_texts)
        logger.info("create MindFAISS done")
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://71.14.88.12:7890")

        r = MultiQueryRetriever(llm, vector_store=vector_store, document_store= db, embed_func=emb.embed_texts)
        doc = r.get_relevant_documents("what is test?")

        self.assertEqual("this is a test", doc[0].page_content)

    def test_MultiQueryRetrieverBase(self):
        if is_torch_npu_available():
            return

        def embed_func(texts):
            return np.random.random((1, 1024))

        mind_llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://127.0.0.1:7890")
        mind_llm.chat = MagicMock(
            return_value="1. Test is a framework for testing and evaluating the quality of a product or service.\n"
                         "2. Test is a process of verifying that a product or service meets certain requirements.\n"
                         "3. Test is a type of software or application designed to simulate a real-world scenario.")
        db = SQLiteDocstore("sql.db")
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2")

        r = MultiQueryRetriever(mind_llm, vector_store=vector_store, document_store= db, embed_func=embed_func)
        r._get_relevant_documents = MagicMock(
            return_value=[Doc(page_content="this is a test", metadata={})])

        doc = r.get_relevant_documents("what is test?")
        logger.info(f"relevant doc {doc}")
        self.assertEqual("this is a test", doc[0].page_content)

    def test_MultiQueryRetrieverMulti(self):
        if is_torch_npu_available():
            return

        def my_side_effect():
            yield [[(Document(page_content="this is a test1", document_name="test1.txt"), 0.01)]]
            yield [[(Document(page_content="this is a test2", document_name="test2.txt"), 0.01)]]
            yield [[(Document(page_content="this is a test3", document_name="test3.txt"), 0.01)]]

        similarity_search_mock = MagicMock()
        similarity_search_mock.side_effect = my_side_effect()

        def embed_func(texts):
            return np.random.random((1, 1024))

        mind_llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://127.0.0.1:7890")
        mind_llm.chat = MagicMock(
            return_value="1. Test is a framework for testing and evaluating the quality of a product or service.\n"
                         "2. Test is a process of verifying that a product or service meets certain requirements.\n"
                         "3. Test is a type of software or application designed to simulate a real-world scenario.")
        db = SQLiteDocstore("sql.db")
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2")

        vector_store.similarity_search = similarity_search_mock

        r = MultiQueryRetriever(mind_llm, vector_store=vector_store, document_store=db, embed_func=embed_func, k=10)
        r._get_relevant_documents = MagicMock(
            return_value=[Doc(page_content="this is a test", metadata={})])
        doc = r.get_relevant_documents("what is test?")
        logger.info(f"relevant doc {doc}")
        self.assertEqual("this is a test", doc[0].page_content)


if __name__ == '__main__':
    unittest.main()
