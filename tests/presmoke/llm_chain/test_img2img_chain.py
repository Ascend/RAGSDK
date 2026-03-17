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

import numpy as np
from pymilvus import MilvusClient

from mx_rag.chain import Img2ImgChain
from mx_rag.embedding.service import CLIPEmbedding
from mx_rag.llm import Img2ImgMultiModel
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import MxDocument, SQLiteDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class TestImg2ImgChain(unittest.TestCase):
    def test_img2img_chain(self):
        milvus_url: str = "http://127.0.0.1:19530"
        img_emb = CLIPEmbedding.create(url="http://127.0.0.1:8000/encode_clip",
                                       client_param=ClientParam(use_http=True))
        client = MilvusClient(milvus_url, server_name="localhost")
        img_vector_store = MilvusDB.create(client=client,
                                           x_dim=1024,
                                           collection_name="test_vector")
        chunk_store = SQLiteDocstore(db_path="./sql.db")
        # 传入模拟数据
        mock_vector = np.array([img_emb.embed_query("查找小男孩图片")])
        img_vector_store.add([0], mock_vector)
        mock_doc = MxDocument(page_content="test", metadata={}, document_name="test_document")
        chunk_store.add([mock_doc], 0)

        img_retriever = Retriever(vector_store=img_vector_store, document_store=chunk_store,
                                  embed_func=img_emb.embed_documents, k=1, score_threshold=0.5)
        multi_model = Img2ImgMultiModel(model_name="sd",
                                        url="http://127.0.0.1:8000/img2img",
                                        client_param=ClientParam(use_http=True))
        img2img_chain = Img2ImgChain(multi_model=multi_model, retriever=img_retriever)
        llm_data = img2img_chain.query("查找小男孩图片",
                                       prompt="he is a knight, wearing armor, big sword in right hand. "
                                              "Blur the background, focus on the knight")
        self.assertEqual(llm_data, {})


if __name__ == '__main__':
    unittest.main()
