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

import unittest

from pymilvus import MilvusClient

from mx_rag.chain import ParallelText2TextChain
from mx_rag.embedding.local import TextEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class TestParallelText2TextChain(unittest.TestCase):
    def test_parallel_chain(self):
        dev = 0
        milvus_url: str = "http://127.0.0.1:19530"
        milvus_client = MilvusClient(milvus_url)
        vector_store = MilvusDB.create(client=milvus_client,
                                       x_dim=1024,
                                       collection_name="test_vector")

        emb = TextEmbedding("/home/data/bge-large-zh-v1.5", dev_id=dev)
        llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct",
                           base_url="http://127.0.0.1:8000/v1/chat/completions",
                           client_param=ClientParam(use_http=True))
        chunk_store = SQLiteDocstore(db_path="./sql.db")
        retriever = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents,
                              k=1, score_threshold=0.6)
        parallel_chain = ParallelText2TextChain(llm=llm, retriever=retriever)
        answer = parallel_chain.query(text="123456")
        self.assertEqual(answer["query"], "123456")


if __name__ == '__main__':
    unittest.main()
