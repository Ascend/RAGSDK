#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import unittest
from unittest.mock import patch, MagicMock

from mx_rag.document.loader.image_loader import ImageLoader

class ImageLoaderTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_load(self):
        loader = ImageLoader(os.path.join(self.data_dir, "test.png"))
        png = loader.load()
        self.assertTrue(png[0].metadata, {"path": os.path.join(self.current_dir, "../../../data/test.png")})

