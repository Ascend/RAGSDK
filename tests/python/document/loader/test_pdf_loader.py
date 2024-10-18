#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest
from unittest import mock
import fitz
from fitz.fitz import EmptyFileError
from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.utils.file_check import FileCheckError


class TestPdfLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_class_init_case(self):
        loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"))
        self.assertIsInstance(loader, PdfLoader)

    def test_class_init_failed(self):
        with self.assertRaises(TypeError):
            loader = PdfLoader(os.path.join(self.data_dir, "link.docx"))
            loader.load()

    def test_load_size_zero_byte(self):
        with unittest.mock.patch('os.path.getsize') as mock_path_getsize:
            mock_path_getsize.return_value = 0  # 0 MByte
            with self.assertRaises(EmptyFileError):
                loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"))
                loader.load()

    # 打桩测试超过了pdf文件超过100M字节场景
    def test_load_size_over_limit(self):
        with unittest.mock.patch('os.path.getsize') as mock_path_getsize:
            mock_path_getsize.return_value = 101 * 1024 * 1024  # 101 MByte
            with self.assertRaises(FileCheckError):
                loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"))
                loader.load()

    # 打桩测试超过了pdf文件页数超过1000页场景
    def test_load_page_num_over_limit(self):
        with unittest.mock.patch('mx_rag.document.loader.pdf_loader.PdfLoader._get_pdf_page_count') \
                as mock_get_pdf_page_count:
            mock_get_pdf_page_count.return_value = 1001  # 1001页
            with self.assertRaises(ValueError):
                loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"))
                loader.load()

    def test_load(self):
        loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"))
        pdf_doc = loader.load()
        self.assertEqual(15, pdf_doc[0].metadata["page_count"])
        self.assertTrue(pdf_doc[0].metadata["source"].find("files/test.pdf"))

    def test_lazy_load(self):
        loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"))
        pdf_doc = loader.lazy_load()
        self.assertTrue(hasattr(pdf_doc, '__iter__'), "lazy_load 应返回一个迭代器")
        self.assertTrue(hasattr(pdf_doc, '__next__'), "lazy_load 应返回一个迭代器")
