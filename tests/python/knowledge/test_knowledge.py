# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

SQL_PATH = "./sql.db"


class TestKnowledge(unittest.TestCase):
    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    def test_knowledge(self):
        with patch("mx_rag.storage.vectorstore.faiss_npu.MindFAISS") as MindFAISS:
            if __name__ == '__main__':
                from mx_rag.storage.vectorstore.faiss_npu import MindFAISS, SimilarityStrategy
                from mx_rag.storage.document_store import SQLiteDocstore
                from mx_rag.knowledge import KnowledgeDB, KnowledgeMgr
                from mx_rag.knowledge.knowledge import KnowledgeStore, KnowledgeMgrStore
                total = np.random.random((3, 1024))
                query = np.array([total[0]])

                def embed_func(texts):
                    if len(texts) > 1:
                        return total
                    return query

                os.system = MagicMock(return_value=0)
                index = MindFAISS(x_dim=1024, devs=[0], similarity_strategy=SimilarityStrategy.FLAT_L2,
                                  load_local_index="./faiss.index")
                db = SQLiteDocstore(SQL_PATH)
                current_dir = os.path.dirname(os.path.realpath(__file__))
                top_path = os.path.dirname(os.path.dirname(current_dir))
                knowledge = KnowledgeDB(KnowledgeStore(SQL_PATH), db, index, "test_knowledge", white_paths=[top_path, ])
                knowledge.add_file("test_file.txt", ["this is a test"], metadatas=[{"filepath": "xxx.file"}],
                                   embed_func=embed_func)
                knowledge.get_all_documents()
                knowledge.delete_file("demo.docx")
                knowledge.get_all_documents()
                knowledge_mgr = KnowledgeMgr(KnowledgeMgrStore(SQL_PATH))
                knowledge2 = KnowledgeDB(KnowledgeStore(SQL_PATH), db, index, "test2_knowledge",
                                         white_paths=[top_path, ])
                knowledge_mgr.register(knowledge)
                knowledge_mgr.register(knowledge2)
                knowledge_mgr.get_all()
                knowledge2.get_all_documents()
                # 有资产删除失败
                try:
                    knowledge_mgr.delete(knowledge)
                except Exception as err:
                    print(err)
                knowledge.delete_file(knowledge.get_all_documents())
                knowledge_mgr.delete(knowledge)
                # 删除知识库后再删除
                try:
                    knowledge_mgr.delete(knowledge)
                except Exception as err:
                    print(err)
                # 重复注册
                knowledge_mgr.register(knowledge)
                try:
                    knowledge_mgr.register(knowledge)
                except Exception as err:
                    print(err)
