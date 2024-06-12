
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock


class TestKnowledge(unittest.TestCase):
    def test_knowledge(self):
        with patch("mx_rag.vectorstore.MindFAISS") as MindFAISS:
            if __name__ == '__main__':
                from mx_rag.vectorstore.faiss_npu import MindFAISS
                from mx_rag.storage import SQLiteDocstore
                from mx_rag.knowledge import Knowledge, KnowledgeBase
                total = np.random.random((3, 1024))
                query = np.array([total[0]])

                def embed_func(texts):
                    if len(texts) > 1:
                        return total
                    return query

                MindFAISS.set_device(0)
                index = MindFAISS(x_dim=1024, index_type="FLAT:L2", auto_save_path="faiss.index")
                db = SQLiteDocstore("./sql.db")
                current_dir = os.path.dirname(os.path.realpath(__file__))
                top_path = os.path.dirname(os.path.dirname(current_dir))
                knowledge = Knowledge("./sql.db", db, index, "test_knowledge", white_paths=[top_path, ])
                knowledge.upload_files([os.path.join(top_path, "data/demo.docx")], embed_func)
                knowledge.upload_files([os.path.join(top_path, "data/demo.docx")], embed_func, force=True)
                knowledge.delete_files(["demo.docx"])
                knowledge.upload_dir(os.path.join(top_path, "data"), embed_func)
                knowledge.get_all_documents()
                knowledge.delete_files(["demo.docx"])
                knowledge.get_all_documents()
                knowledge_mgr = KnowledgeBase("./sql.db")
                knowledge2 = Knowledge("./sql.db", db, index, "test2_knowledge", white_paths=[top_path, ])
                knowledge_mgr.register(knowledge)
                knowledge_mgr.register(knowledge2)
                knowledge_mgr.get_all()
                knowledge2.get_all_documents()
                # 有资产删除失败
                try:
                    knowledge_mgr.delete(knowledge)
                except Exception as err:
                    print(err)
                knowledge.delete_files(knowledge.get_all_documents())
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

