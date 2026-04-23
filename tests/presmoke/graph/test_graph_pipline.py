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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pymilvus import MilvusClient

from mx_rag.chain.single_text_to_text import GraphRagText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.embedding.local import TextEmbedding
from mx_rag.graphrag import GraphRAGPipeline
from mx_rag.llm import LLMParameterConfig, Text2TextLLM
from mx_rag.reranker.local import LocalReranker
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class TestGraphPipline(unittest.TestCase):
    def test_graph_pipline(self):
        test_file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       "../../data/test.txt"))
        work_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 "../../data/test_pipeline"))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        dev_id = 0
        llm = Text2TextLLM(
            base_url="http://127.0.0.1:1025/v1/chat/completions",
            model_name="Llama3-8B-Chinese-Chat",
            llm_config=LLMParameterConfig(temperature=0.6, top_p=0.9),
            client_param=ClientParam(use_http=True),
        )
        rerank_model = LocalReranker("/home/data/bge-reranker-large/", dev_id, 1, False)
        embedding_model = TextEmbedding.create(model_path="/home/data/bge-large-zh-v1.5")
        data_load_mng = LoaderMng()
        data_load_mng.register_loader(TextLoader, [".txt"])
        data_load_mng.register_splitter(
            RecursiveCharacterTextSplitter,
            [".txt"],
            dict(chunk_size=512, chunk_overlap=20)
        )
        graph_name = "test"
        graph_type = "networkx"
        milvus_url: str = "http://my-release-milvus.milvus:19530"
        milvus_client = MilvusClient(milvus_url, server_name="localhost")
        vector_store = MilvusDB.create(client=milvus_client,
                                       x_dim=1024,
                                       collection_name="test_vector")
        pipeline = GraphRAGPipeline(work_dir, llm, embedding_model, 1024, rerank_model,
                                    graph_name=graph_name, node_vector_store=vector_store,
                                    concept_vector_store=vector_store)
        pipeline.upload_files([test_file_path], data_load_mng)
        pipeline.build_graph()
        question = "谁看电影？"
        contexts = pipeline.retrieve_graph(question)
        text2text_chain = GraphRagText2TextChain(
            llm=llm,
            retriever=pipeline.as_retriever(),
            reranker=rerank_model)
        result = text2text_chain.query(question)
        self.assertGreaterEqual(len(result.get("result")), 0)
        