# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import patch
from mx_rag.document.loader.load_excel import ExcelLoader


class TestExcelLoader(unittest.TestCase):

    def test_class_init_case(self):
        loader = ExcelLoader("./mx_rag/document/loader/files/test.xlsx")
        self.assertIsInstance(loader, ExcelLoader)

    @patch.object(ExcelLoader, '_load_xls')
    def test_call_load_xls(self, mock_load_xls):
        # Arrange
        loader = ExcelLoader("./mx_rag/document/loader/files/test.xls")
        loader.load()
        mock_load_xls.assert_called_once()

    def test_load_xls(self):
        loader = ExcelLoader("./mx_rag/document/loader/files/test.xls")
        docs = loader.load()
        self.assertEqual(2, len(docs))
        self.assertEqual("", docs[1].page_content)
        content = ('None:中文资讯;source:机器之心;link:入门 | 机器之心 (jiqizhixin.com);count\n（3月15日为例）:9篇;SUM:24篇;**;'
                   'None:None;source:量子位;link:https://www.zhihu.com/org/liang-zi-wei-48/posts;count\n（3月15日为例）'
                   ':4篇;SUM:None;**;None:None;source:新智元;link:https://www.zhihu.com/org/xin-zhi-yuan-88-3;count\n'
                   '（3月15日为例）:4篇;SUM:None;**;None:None;source:极客公园;link:行业资讯 | 极客公园 (geekpark.net);count\n'
                   '（3月15日为例）:7篇;SUM:None;**;None:英文文献;source:huggingFace;link:日报 - 拥抱脸部 ('
                   'huggingface.co);count\n（3月15日为例）:14篇;SUM:55篇;**;None:None;source:PaperWithCode;link:带代码的最'
                   '新论文 |带代码的论文 (paperswithcode.com);count\n（3月15日为例）:41篇;SUM:None;**;')
        self.assertEqual(content, docs[0].page_content)
        self.assertEqual(docs[0].metadata["sheet"], '不需要订阅')
        self.assertEqual(len(docs[0].metadata), 2)

    @patch.object(ExcelLoader, '_load_xlsx')
    def test_call_load_xlsx(self, mock_load_xlsx):
        # Arrange
        loader = ExcelLoader("./mx_rag/document/loader/files/test.xlsx")
        loader.load()
        mock_load_xlsx.assert_called_once()

    def test_load_xlsx(self):
        loader = ExcelLoader("./mx_rag/document/loader/files/test.xlsx")
        docs = loader.load()
        content = ('None:中文资讯;source:机器之心;link:入门 | 机器之心 (jiqizhixin.com);count\n（3月15日为例）:9篇;SUM:24篇;**;'
                   'None:None;source:量子位;link:https://www.zhihu.com/org/liang-zi-wei-48/posts;count\n（3月15日为例）'
                   ':4篇;SUM:None;**;None:None;source:新智元;link:https://www.zhihu.com/org/xin-zhi-yuan-88-3;count\n'
                   '（3月15日为例）:4篇;SUM:None;**;None:None;source:极客公园;link:行业资讯 | 极客公园 (geekpark.net);count\n'
                   '（3月15日为例）:7篇;SUM:None;**;None:英文文献;source:huggingFace;link:日报 - 拥抱脸部 ('
                   'huggingface.co);count\n（3月15日为例）:14篇;SUM:55篇;**;None:None;source:PaperWithCode;link:带代码的最'
                   '新论文 |带代码的论文 (paperswithcode.com);count\n（3月15日为例）:41篇;SUM:None;**;')

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, content)
        self.assertEqual(docs[0].metadata["sheet"], '不需要订阅')
        self.assertEqual(len(docs[0].metadata), 2)

    @patch.object(ExcelLoader, '_load_csv')
    def test_call_load_csv(self, mock_load_csv):
        # Arrange
        loader = ExcelLoader("./mx_rag/document/loader/files/test.csv")
        loader.load()
        mock_load_csv.assert_called_once()

    def test_load_csv(self):
        loader = ExcelLoader("./mx_rag/document/loader/files/test.csv")
        docs = loader.load()
        content = ('None:中文资讯;source:机器之心;link:入门|机器之心(jiqizhixin.com);count\n（3月15日为例）:9篇;SUM:24篇;**;'
                   'None:None;source:量子位;link:https://www.zhihu.com/org/liang-zi-wei-48/posts;count\n（3月15日为例）:'
                   '4篇;SUM:None;**;')
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, content)
        self.assertEqual(len(docs[0].metadata), 1)
