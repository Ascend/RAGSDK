#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os
import unittest
from typing import List
from unittest.mock import MagicMock

from loguru import logger
from transformers import is_torch_npu_available
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.document.loader import DocxLoader
from mx_rag.knowledge import KnowledgeDB
from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever, MultiQueryRetriever
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.chain import SingleText2TextChain
from mx_rag.llm.llm_parameter import LLMParameterConfig


class MyTestCase(unittest.TestCase):
    sql_db_file = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/sql.db"))

    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

    def test_with_npu(self):
        if not is_torch_npu_available():
            return
        current_dir = os.path.dirname(os.path.realpath(__file__))

        loader = DocxLoader(os.path.realpath(os.path.join(current_dir, "../../data/test.docx")))
        spliter = RecursiveCharacterTextSplitter()
        res = loader.load_and_split(spliter)
        emb = TextEmbedding("/workspace/bge-large-zh/", 2)
        db = SQLiteDocstore(MyTestCase.sql_db_file)
        logger.info("create emb done")
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0],
                          load_local_index="./faiss.index")
        knowledge_store = KnowledgeStore(MyTestCase.sql_db_file)
        knowledge_store.add_knowledge(knowledge_name='test', user_id='Default')
        vector_store = KnowledgeDB(knowledge_store, db, index, "test", white_paths=["/home"], user_id='Default')
        vector_store.add_file("test.docx",
                              [d.page_content for d in res],
                              embed_func=emb.embed_documents,
                              metadatas=[d.metadata for d in res]
                                )

        logger.info("create MindFAISS done")
        llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct", base_url="http://70.255.71.175:3000", timeout=120)

        def test_rag_chain_npu_no_doc(self):
            r = Retriever(vector_store=vector_store, embed_func=emb.embed_documents, score_threshold=0.5)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            rag.source = True
            for response in rag.query("CANN是什么呢", LLMParameterConfig(max_tokens=1024, temperature=0.1,
                                                                         top_p=1.0, stream=True)):
                logger.trace(f"response {response}")
                self.assertTrue(len(response.get('source_documents', None)) == 0)

        test_rag_chain_npu_multi_doc_query_rewrite(self)


if __name__ == '__main__':
    unittest.main()
