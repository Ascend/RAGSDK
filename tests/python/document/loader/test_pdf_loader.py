#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest

from mx_rag.document.loader.pdf_loader import PdfLoader


class TestPdfLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_class_init_case(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
        self.assertIsInstance(loader, PdfLoader)

    def test_load(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
        pdf_doc = loader.load()
        self.assertEqual(15, pdf_doc[0].metadata["page_count"])
        self.assertTrue(pdf_doc[0].metadata["source"].find("files/test.pdf"))
