#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import unittest
from unittest import mock
import fitz
from fitz.fitz import EmptyFileError
from paddle.base import libpaddle
from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.utils import Lang
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
        pdf_doc = list(loader.lazy_load())
        self.assertEqual(15, pdf_doc[0].metadata["page_count"])
        self.assertTrue(pdf_doc[0].metadata["source"].find("files/test.pdf"))

    def test_parser(self):
        loader = PdfLoader(os.path.join(self.data_dir, "test2.pdf"), enable_ocr=True, lang=Lang.CH)
        pdf_doc = list(loader.lazy_load())
        self.assertEqual(1, pdf_doc[0].metadata["page_count"])
        self.assertTrue(pdf_doc[0].metadata["source"].find("files/test2.pdf"))

        with mock.patch('mx_rag.document.loader.pdf_loader.PPStructure') as mock_score_scale:
            mock_score_scale.side_effect = Exception("Test other exception")
            loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"), enable_ocr=True, lang=Lang.EN)
            self.assertIsNone(loader.ocr_engine)

        with mock.patch('mx_rag.document.loader.pdf_loader.PPStructure') as mock_score_scale:
            mock_score_scale.side_effect = AssertionError("Test assertion error")
            loader = PdfLoader(os.path.join(self.data_dir, "test.pdf"), enable_ocr=True, lang=Lang.EN)
            self.assertIsNone(loader.ocr_engine)


if __name__ == '__main__':
    unittest.main()
