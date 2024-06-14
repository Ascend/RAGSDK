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
