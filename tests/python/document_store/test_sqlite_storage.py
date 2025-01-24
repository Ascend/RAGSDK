# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
import os
from pathlib import Path
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.document_store import MxDocument

SQL_PATH = str(Path(__file__).parent.absolute() / "sql.db")


class TestSQLiteStorage(unittest.TestCase):
    def setUp(self):
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)
        self.db = SQLiteDocstore(SQL_PATH)

    def tearDown(self):
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    def test_sqlite_storage_add(self):
        # 对add函数入参进行校验测试
        doc = MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test")
        with self.assertRaises(ValueError):
            # 期望传入一个列表
            self.db.add(doc, 1)
            # 期望列表元素的类型为MxDocument
            self.db.add([0], 1)
        self.assertEqual(self.db.add([doc], 1), [1])

    def test_sqlite_storage_delete(self):
        # 不删除任何chunk，返回空列表
        self.assertEqual(self.db.delete(document_id=1), [])

    def test_sqlite_storage_search(self):
        # 对search函数入参进行校验测试
        with self.assertRaises(ValueError):
            # 期望传入一个整数
            self.db.search(-1)
        doc = MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test")
        self.db.add([doc], 1)
        chunk = self.db.search(1)
        self.assertEqual(chunk.page_content, "Hello mxRAG")
        self.db.delete(1)
        self.assertEqual(self.db.get_all_index_id(), [])

    def test_chunk_encrypt(self):
        def fack_encryt(value):
            return "fack_encryt"

        db = SQLiteDocstore(SQL_PATH, encrypt_fun=fack_encryt)
        doc = MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test")
        db.add([doc], 1)
        chunk = db.search(1)
        self.assertEqual(chunk.page_content, "fack_encryt")


if __name__ == '__main__':
    unittest.main()
