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
import shutil
import unittest

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pymilvus import MilvusClient

from mx_rag.cache import (CacheChainChat, EvictPolicy, MxRAGCache,
                          SimilarityCacheConfig)
from mx_rag.chain import SingleText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.embedding.local import TextEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore, upload_files
from mx_rag.llm import Text2TextLLM
from mx_rag.storage.document_store import MilvusDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class TestSimilarityCacheDemo(unittest.TestCase):
    def setUp(self):
        self.path_to_cache_save_folder = os.path.join(os.path.realpath(os.path.dirname(__file__)), "./rag_cache")
        if os.path.exists(self.path_to_cache_save_folder):
            shutil.rmtree(self.path_to_cache_save_folder)
        os.makedirs(self.path_to_cache_save_folder)

    def test_chain(self):
        dim = 1024
        milvus_url: str = "http://127.0.0.1:19530"
        client = MilvusClient(milvus_url, server_name="localhost")
        dev = 0
        embedding_path = "/home/data/bge-large-zh-v1.5"
        reranker_path = "/home/data/bge-reranker-large"
        llm_url: str = "http://127.0.0.1:8000/v1/chat/completions"
        model_name: str = "Llama3-8B-Chinese-Chat"
        file_path: str = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       "../../data/gaokao.txt"))
        similarity_config = SimilarityCacheConfig(
            vector_config={
                "client": client,
                "vector_type": "milvus_db",
                "x_dim": dim,
                "collection_name": "mxrag_cache",  # milvus db的标签
                "param": None
            },
            cache_config="sqlite",
            emb_config={
                "embedding_type": "local_text_embedding",
                "x_dim": dim,
                "model_path": embedding_path,  # emb 模型路径
                "dev_id": dev
            },
            similarity_config={
                "similarity_type": "local_reranker",
                "model_path": reranker_path,  # reranker 模型路径
                "dev_id": dev
            },
            retrieval_top_k=1,
            cache_size=100,
            auto_flush=100,
            similarity_threshold=0.2,
            data_save_folder=self.path_to_cache_save_folder,
            disable_report=True,
            eviction_policy=EvictPolicy.FIFO
        )
        # cache 初始化
        cache = MxRAGCache("similarity_cache", similarity_config)
        # 设置缓存每条的字符限制为4000个字符
        cache.set_cache_limit(10000)
        # 设置是否详细显示缓存过程
        cache.set_verbose(True)
        llm = Text2TextLLM(base_url=llm_url, model_name=model_name, client_param=ClientParam(use_http=True, timeout=60))
        vector_store = MilvusDB.create(client=client,
                                       x_dim=1024,
                                       collection_name="test_vector")
        chunk_store = MilvusDocstore(client, collection_name="test_chunk")
        loader_mng = LoaderMng()
        # 加载文档加载器，可以使用mxrag自有的，也可以使用langchain的
        loader_mng.register_loader(loader_class=TextLoader, file_types=[".txt", ".md"])
        # 加载文档切分器，使用langchain的
        loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                                     file_types=[".pdf", ".docx", ".txt", ".md"],
                                     splitter_params={"chunk_size": 750,
                                                      "chunk_overlap": 150,
                                                      "keep_separator": False
                                                      }
                                     )
        # 初始化知识管理关系数据库
        knowledge_store = KnowledgeStore(db_path="./sql.db")
        emb = TextEmbedding(embedding_path, dev)
        # 添加知识库
        knowledge_store.add_knowledge("test", "Default", "admin")
        # 初始化知识库管理
        knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                                   chunk_store=chunk_store,
                                   vector_store=vector_store,
                                   knowledge_name="test",
                                   user_id='Default',
                                   white_paths=["/home"])
        # 完成离线知识库构建,上传领域知识test.docx文档。
        upload_files(knowledge_db, [file_path],
                     loader_mng=loader_mng,
                     embed_func=emb.embed_documents,
                     force=True)
        # Step2在线问题答复,初始化检索器
        retriever = vector_store.as_retriever(document_store=chunk_store,
                                              embed_func=emb.embed_documents, k=3, score_threshold=0.1)
        text2text_chain = SingleText2TextChain(llm=llm, retriever=retriever)
        cache_chain = CacheChainChat(chain=text2text_chain, cache=cache)
        res1 = cache_chain.query("请描述2024年高考作文题目")
        res2 = cache_chain.query("2024年的高考作文题目?")
        # 运行不稳定
        self.assertEqual(res1.get("result")[:160], res2.get("result")[:160])


if __name__ == '__main__':
    unittest.main()
