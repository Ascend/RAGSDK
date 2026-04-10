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
import os.path
import unittest

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from paddle.base import libpaddle
from pymilvus import MilvusClient

from mx_rag.chain import SingleText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB
from mx_rag.knowledge.handler import upload_files
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import MilvusDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class TestQADemo(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./sql.db"):
            os.remove("./sql.db")

    def test_query(self):
        white_path: list[str] = ["/home","/workspace"]
        file_path: str = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       "../../data/gaokao.txt"))
        llm_url: str = "http://127.0.0.1:8000/v1/chat/completions"
        embedding_url: str = "http://127.0.0.1:8000/v1/embeddings"
        model_name: str = "Llama3-8B-Chinese-Chat"
        score_threshold: float = 0.5
        query: str = "高考作文语文题目"
        milvus_url: str = "http://my-release-milvus.milvus:19530"
        # 离线构建知识库,首先注册文档处理器
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
        emb = TEIEmbedding(url=embedding_url, client_param=ClientParam(use_http=True))
        # 初始化向量数据库
        milvus_client = MilvusClient(milvus_url)
        vector_store = MilvusDB.create(client=milvus_client,
                                       x_dim=1024,
                                       collection_name="test_vector")
        # 初始化文档chunk关系数据库
        chunk_store = MilvusDocstore(milvus_client, collection_name="test_chunk")
        # 初始化知识管理关系数据库
        knowledge_store = KnowledgeStore(db_path="./sql.db")
        # 添加知识库
        knowledge_store.add_knowledge("test", "Default", "admin")
        # 初始化知识库管理
        knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                                   chunk_store=chunk_store,
                                   vector_store=vector_store,
                                   knowledge_name="test",
                                   white_paths=white_path,
                                   user_id="Default"
                                   )
        upload_files(knowledge=knowledge_db,
                     files=[file_path],
                     loader_mng=loader_mng,
                     embed_func=emb.embed_documents,
                     force=True
                     )

        # 初始化Retriever检索器
        text_retriever = Retriever(vector_store=vector_store,
                                   document_store=chunk_store,
                                   embed_func=emb.embed_documents,
                                   k=3,
                                   score_threshold=score_threshold
                                   )

        # 配置text生成text大模型chain，具体ip端口请根据实际情况适配修改
        llm = Text2TextLLM(base_url=llm_url, model_name=model_name, client_param=ClientParam(use_http=True, timeout=60))
        text2text_chain = SingleText2TextChain(retriever=text_retriever, llm=llm)
        res = text2text_chain.query(query)
        self.assertEqual(res.get("result"), "模拟回复：高考作文语文题目 的相关内容...")


if __name__ == '__main__':
    unittest.main()
