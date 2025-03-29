# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import pathlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mx_rag.knowledge import KnowledgeDB
from mx_rag.knowledge.base_knowledge import KnowledgeError
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS

SQL_PATH = "./sql.db"


class TestKnowledge(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_file = os.path.realpath(os.path.join(current_dir, "../../data/test.md"))

    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    @patch("mx_rag.knowledge.KnowledgeDB._check_store_accordance")
    def test_knowledge(self, knowledge_db_mock):
        embeddings = np.concatenate([np.random.random((1, 1024))])

        def embed_func(texts):
            return embeddings.tolist()

        db = SQLiteDocstore(SQL_PATH)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        top_path = os.path.dirname(os.path.dirname(current_dir))
        vector_store = MagicMock(spec=MindFAISS)
        vector_store.add = MagicMock(return_value=None)
        knowledge_store = KnowledgeStore(SQL_PATH)
        knowledge_store.add_knowledge("test_knowledge", "user123", "admin")
        with self.assertRaises(KnowledgeError):
            knowledge_store.add_knowledge("test_knowledge", "user123", 'admin')
        self.assertEqual(knowledge_store.check_knowledge_exist("test_knowledge", "user123"), True)
        knowledge_store.add_usr_id_to_knowledge("test_knowledge", "user124", "admin")

        with self.assertRaises(KnowledgeError):
            knowledge_store.add_usr_id_to_knowledge("test_knowledge001", "user124", "admin")
        with self.assertRaises(KnowledgeError):
            knowledge_store.add_usr_id_to_knowledge("test_knowledge", "user124", "admin")

        knowledge = KnowledgeDB(knowledge_store, db, vector_store, "test_knowledge",
                                white_paths=[top_path, ], user_id='user123')
        knowledge.add_file(pathlib.Path(self.test_file), ["this is a test"], metadatas=[{"filepath": "xxx.file"}],
                           embed_func=embed_func)
        with self.assertRaises(KnowledgeError):
            knowledge.add_file(pathlib.Path(self.test_file), ["this is a test"],
                               metadatas=[{"filepath": "xxx.file"}, {"filepath": "yyy.file"}], embed_func=embed_func)

        self.assertEqual(knowledge.get_all_documents()[0].knowledge_name, "test_knowledge")
        self.assertEqual(knowledge.get_all_documents()[0].document_name, "test.md")

        knowledge_db_mock.return_value = None
        knowledge_db1 = KnowledgeDB(KnowledgeStore(SQL_PATH), db, vector_store, "test_knowledge",
                                    white_paths=[top_path, ], user_id="user123")
        self.assertEqual(knowledge_db1.get_all_documents()[0].knowledge_name, "test_knowledge")
        self.assertEqual(knowledge_db1.get_all_documents()[0].document_name, "test.md")
        self.assertEqual(knowledge_store.get_all_knowledge_name('user123'), ["test_knowledge"])

        # 删除文档后, 只剩下空的knowledge
        knowledge.delete_file("test.md")
        self.assertEqual(knowledge_store.get_all_knowledge_name('user123'), ["test_knowledge"])
        self.assertEqual(knowledge.get_all_documents(), [])
        self.assertEqual(knowledge_store.get_all_usr_role_by_knowledge("test_knowledge"),
                         {'user123': 'admin', 'user124': 'admin'})
        self.assertEqual(knowledge_store.get_all_usr_role_by_knowledge("test_knowledge001"), {})
        # 多个usr_id对knowledge关系删除
        knowledge_store.delete_usr_id_from_knowledge("test_knowledge", "user123", 'admin')
        # user_id和knowledge1对1时，不允许删除关系，使用delete_knowledge删除
        with self.assertRaises(KnowledgeError):
            knowledge_store.delete_usr_id_from_knowledge("test_knowledge", "user123", 'admin')
        with self.assertRaises(KnowledgeError):
            knowledge_store.delete_usr_id_from_knowledge("test_knowledge001", "user123", 'admin')

        knowledge_db1.delete_all()
        self.assertEqual(knowledge_store.get_all_knowledge_name('user123'), [])

if __name__ == '__main__':
    unittest.main()
