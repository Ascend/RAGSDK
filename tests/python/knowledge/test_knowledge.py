# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import shutil
import unittest
from unittest.mock import patch, Mock, MagicMock

import numpy as np

from mx_rag.chain.tree_text_to_text import TreeText2TextChain
from mx_rag.knowledge.knowledge import KnowledgeTreeDB, KnowledgeStore
from mx_rag.retrievers import TreeBuilderConfig

from mx_rag.storage.document_store import SQLiteDocstore

SQL_PATH = "./sql.db"


class TestKnowledge(unittest.TestCase):
    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    def test_knowledge(self):
        with patch("mx_rag.storage.vectorstore.faiss_npu.MindFAISS") as MindFAISS:
            if __name__ == '__main__':
                from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
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
                index = MindFAISS(x_dim=1024, devs=[0], index_type="FLAT:L2", load_local_index="./faiss.index")
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

    @patch("mx_rag.retrievers.tree_retriever.tree_builder.TreeBuilder.build_from_text")
    def test_knowledge_treeDB(self, mock_build_from_text):
        mock_build_from_text.return_value = None
        current_dir = os.path.dirname(os.path.realpath(__file__))
        top_path = os.path.dirname(os.path.dirname(current_dir))
        tree_builder_config = TreeBuilderConfig(tokenizer=Mock(), summarization_model=Mock(spec=TreeText2TextChain))
        shutil.disk_usage = MagicMock(return_value=(1, 1, 1000 * 1024 * 1024))
        knwoledge_tree_db = KnowledgeTreeDB(KnowledgeStore(SQL_PATH), chunk_store=SQLiteDocstore(SQL_PATH),
                                            knowledge_name="test_knowledge",
                                            white_paths=[top_path], tree_builder_config=tree_builder_config)
        knwoledge_tree_db.add_files(["filename.txt", "filename1.txt"], ["this is a test", "this is a test1"], None,
                                    [{"filepath": "xxx.file"}, {"filepath": "xxx.file"}])
        documents = knwoledge_tree_db.get_all_documents()
        self.assertEqual(['filename.txt', 'filename1.txt'], documents)
        knwoledge_tree_db.delete_file("filename.txt")
        self.assertTrue(knwoledge_tree_db.check_document_exist("filename1.txt"))
        knwoledge_tree_db.add_file("testName.txt", "", None, "")
        knwoledge_tree_db.check_document_exist("testName.txt")
