# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.document_store.base_storage import MxDocument

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
        with self.assertRaises(TypeError):
            # 期望传入一个列表
            self.db.add(doc)
            # 期望列表元素的类型为MxDocument
            self.db.add([0])

        # 传入空文档列表，应返回空列表
        self.assertEqual(self.db.add([doc]), [1])

    def test_sqlite_storage_delete(self):
        # 对delete函数入参进行校验测试
        with self.assertRaises(ValueError):
            # 期望doc_name非空
            self.db.delete(doc_name="")
            # 期望doc_name为字符串
            self.db.delete(0)
        # 传入不存在的doc_name，不删除任何chunk，返回空列表
        self.assertEqual(self.db.delete(doc_name="test1"), [])

    def test_sqlite_storage_search(self):
        # 对search函数入参进行校验测试
        with self.assertRaises(ValueError):
            # 期望传入一个整数
            self.db.search(-1)


if __name__ == '__main__':
    unittest.main()