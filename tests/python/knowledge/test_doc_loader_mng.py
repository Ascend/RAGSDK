# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.document.loader import ExcelLoader
from mx_rag.knowledge.doc_loader_mng import LoaderMng, LoaderInfo, SplitterInfo


class LoaderMngTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data/test.xlsx"))

    def test_register_loader(self):
        loader_mng = LoaderMng()
        loader_mng.register_loader(ExcelLoader, [".xlsx"])
        loader_info = loader_mng.get_loader(".xlsx")
        loader = loader_info.loader_class(file_path=self.data_dir)
        self.assertIsInstance(loader, ExcelLoader)

    def test_register_splitter(self):
        loader_mng = LoaderMng()
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        splitter_info = loader_mng.get_splitter(".xlsx")
        splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
        self.assertIsInstance(splitter, RecursiveCharacterTextSplitter)

    def test_unregister_loader(self):
        loader_mng = LoaderMng()
        loader_mng.register_loader(ExcelLoader, [".xlsx"])
        loader_mng.unregister_loader(ExcelLoader)
        with self.assertRaises(KeyError):
            loader_mng.get_loader(".xlsx")

    def test_unregister_splitter(self):
        loader_mng = LoaderMng()
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        loader_mng.unregister_splitter(RecursiveCharacterTextSplitter)
        with self.assertRaises(KeyError):
            loader_mng.get_splitter(".xlsx")

    def test_get_loader(self):
        loader_mng = LoaderMng()
        loader_mng.register_loader(ExcelLoader, [".xlsx"])
        loader_info = loader_mng.get_loader(".xlsx")
        self.assertIsInstance(loader_info, LoaderInfo)

    def test_get_splitter(self):
        loader_mng = LoaderMng()
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        splitter_info = loader_mng.get_splitter(".xlsx")
        self.assertIsInstance(splitter_info, SplitterInfo)
