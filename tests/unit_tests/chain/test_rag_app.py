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
from unittest.mock import MagicMock

from loguru import logger
from transformers import is_torch_npu_available
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.document.loader import DocxLoader
from mx_rag.knowledge import KnowledgeDB
from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore


class MyTestCase(unittest.TestCase):
    sql_db_file = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/sql.db"))

    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

    @unittest.skipUnless(is_torch_npu_available(), "NPU not available")  # noqa
    def test_with_npu(self):  # noqa
        current_dir = os.path.dirname(os.path.realpath(__file__))
        loader = DocxLoader(os.path.realpath(os.path.join(current_dir, "../../data/test.docx")))
        spliter = RecursiveCharacterTextSplitter()
        res = loader.load_and_split(spliter)
        emb = TextEmbedding("/workspace/bge-large-zh/", 2)
        db = SQLiteDocstore(MyTestCase.sql_db_file)
        logger.info("create emb done")
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0], load_local_index="./faiss.index")
        knowledge_store = KnowledgeStore(MyTestCase.sql_db_file)
        knowledge_store.add_knowledge(knowledge_name='test', user_id='Default')
        vector_store = KnowledgeDB(knowledge_store, db, index, "test", white_paths=["/home"], user_id='Default')
        vector_store.add_file(
            "test.docx",
            [d.page_content for d in res],
            embed_func=emb.embed_documents,
            metadatas=[d.metadata for d in res],
        )


if __name__ == '__main__':
    unittest.main()
