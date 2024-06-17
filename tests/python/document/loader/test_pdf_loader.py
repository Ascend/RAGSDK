#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest

from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.document.loader.pdf_loader import PdfLang


class TestPdfLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    os.environ['http_proxy'] = "http://p_atlas:proxy%40123@proxycn2.huawei.com:8080/"
    os.environ['https_proxy'] = "http://p_atlas:proxy%40123@proxycn2.huawei.com:8080/"

    def test_class_init_case(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
        self.assertIsInstance(loader, PdfLoader)

    def test_class_init_failed(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/link.docx"))
        self.assertIsInstance(loader, PdfLoader)

        pdf_doc = loader.load()
        self.assertEqual(pdf_doc, [])

    def test_load_size_zero_byte(self):
        with unittest.mock.patch('os.path.getsize') as mock_path_getsize:
            mock_path_getsize.return_value = 0  # 0 MByte
            loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
            pdf_doc = loader.load()
            self.assertEqual(pdf_doc, [])

    # 打桩测试超过了pdf文件超过100M字节场景
    def test_load_size_over_limit(self):
        with unittest.mock.patch('os.path.getsize') as mock_path_getsize:
            mock_path_getsize.return_value = 101 * 1024 * 1024  # 101 MByte
            loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
            pdf_doc = loader.load()
            self.assertEqual(pdf_doc, [])

    # 打桩测试超过了pdf文件页数超过1000页场景
    def test_load_page_num_over_limit(self):
        with unittest.mock.patch('mx_rag.document.loader.pdf_loader.PdfLoader._get_pdf_page_count') \
                as mock_get_pdf_page_count:
            mock_get_pdf_page_count.return_value = 1001  # 1001页
            loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
            pdf_doc = loader.load()
            self.assertEqual(pdf_doc, [])

    def test_load(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
        pdf_doc = loader.load()
        self.assertEqual(15, pdf_doc[0].metadata["page_count"])
        self.assertTrue(pdf_doc[0].metadata["source"].find("files/test.pdf"))

    def test_load_layout_en(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test_layout.pdf"),
                           layout_recognize=True, lang=PdfLang.EN)
        pdf_doc = loader.load()
        self.assertEqual(7, pdf_doc[0].metadata["page_count"])
        self.assertTrue(pdf_doc[0].metadata["source"].find("files/test.pdf"))
