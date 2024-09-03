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

    def test_lazy_load(self):
        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"))
        ppt_doc = loader.lazy_load()
        self.assertTrue(hasattr(ppt_doc, '__iter__'), "lazy_load 应返回一个迭代器")
        self.assertTrue(hasattr(ppt_doc, '__next__'), "lazy_load 应返回一个迭代器")
