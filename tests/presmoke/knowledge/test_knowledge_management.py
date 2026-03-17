# !/usr/bin/env python3
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
import pathlib
import unittest

from pymilvus import MilvusClient

from mx_rag.embedding.local import TextEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MilvusDB


class TestKnowledgeManagement(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./sql.db"):
            os.remove("./sql.db")

    def test_knowledge_management(self):
        # 设置向量检索使用的NPU卡
        dev = 0
        milvus_url: str = "http://127.0.0.1:19530"
        # 加载embedding模型
        embed_func = TextEmbedding("/home/data/bge-large-zh-v1.5", dev_id=dev)
        # 初始化向量数据库
        milvus_client = MilvusClient(milvus_url)
        vector_store = MilvusDB.create(client=milvus_client,
                                       x_dim=1024,
                                       collection_name="test_vector")
        # 初始化文档chunk 关系数据库
        chunk_store = SQLiteDocstore(db_path="./sql.db")
        # 初始化知识管理关系数据库
        knowledge_store = KnowledgeStore(db_path="./sql.db")
        # 添加知识库及管理员
        knowledge_store.add_knowledge(knowledge_name="test", user_id='Default', role='admin')
        # 初始化知识管理
        knowledge_db = KnowledgeDB(knowledge_store=knowledge_store, chunk_store=chunk_store, vector_store=vector_store,
                                   knowledge_name="test", user_id="Default", white_paths=["/home/"])
        test_file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       "../../data/test.txt"))
        file_path = pathlib.Path(test_file_path)
        knowledge_db.add_file(file=file_path,
                              texts=["test1", "test2"],
                              embed_func={"dense": embed_func.embed_documents},
                              metadatas=[{"source": "./test.txt"}, {"source": "./test.txt"}])
        documents = [document.document_name for document in knowledge_db.get_all_documents()]
        self.assertEqual(documents, ["test.txt"])
        self.assertTrue(knowledge_db.check_document_exist(doc_name=file_path.name))
        knowledge_db.delete_file(doc_name=file_path.name)
        knowledge_db.delete_all()
        self.assertFalse(knowledge_db.check_document_exist(doc_name=file_path.name))


if __name__ == '__main__':
    unittest.main()
