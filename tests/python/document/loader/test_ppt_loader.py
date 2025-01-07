#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest
from unittest.mock import patch, MagicMock

from mx_rag.document.loader.ppt_loader import PowerPointLoader


class TestPPTLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_load(self):
        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"))
        ppt_doc = loader.load()
        self.assertEqual(ppt_doc[0].metadata["source"],
                         os.path.join(self.data_dir, "test.pptx"))

        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"), enable_ocr=True)
        ppt_doc = loader.load()
        self.assertEqual(ppt_doc[0].metadata["source"],
                         os.path.join(self.data_dir, "test.pptx"))

    def test_enable_ocr_false(self):
        # 禁用ocr进行图片识别
        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"), enable_ocr=False)
        ppt_doc = loader.load()
        self.assertEqual(ppt_doc[0].metadata["source"],
                         os.path.join(self.data_dir, "test.pptx"))

    def test_invalid_enable_ocr(self):
        with self.assertRaises(ValueError):
            _ = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"), enable_ocr=0)
