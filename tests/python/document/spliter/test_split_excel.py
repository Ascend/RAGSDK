# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest

from langchain.text_splitter import CharacterTextSplitter

from mx_rag.document.loader.excel_loader import ExcelLoader
from mx_rag.document.splitter import CharTextSplitter


class TestExcelSplit(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def compare_langchain_res(self, excel_doc):
        langchain_char_text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100, separator="\n")
        expect_split_text = langchain_char_text_splitter.split_text(excel_doc[0].page_content)

        mxrag_char_text_splitter = CharTextSplitter(chunk_size=512, chunk_overlap=100, separator="\n")
        result_split_text = mxrag_char_text_splitter.split_text(excel_doc[0].page_content)

        self.assertEqual(expect_split_text, result_split_text)

    def test_excel_split(self):
        loader = ExcelLoader("./data/test.xls")
        excel_doc = loader.load()
        self.compare_langchain_res(excel_doc)
