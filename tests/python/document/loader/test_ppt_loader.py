#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest
from unittest.mock import patch, MagicMock


class TestPPTLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_load(self):
        with patch("paddleocr.PaddleOCR") as PaddleOCR:
            from mx_rag.document.loader.ppt_loader import PowerPointLoader
            loader = PowerPointLoader(os.path.realpath(os.path.join(self.current_dir, "../../../data/test.pptx")))
            loader._load_image_text = MagicMock(return_value=["1111", "222"])
            ppt_doc = loader.load()
            self.assertTrue(ppt_doc[0].metadata["source"].find("files/test.pdf"))
