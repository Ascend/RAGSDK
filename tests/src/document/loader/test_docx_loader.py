# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest

from mx_rag.document.loader import DocxLoader


class DocxLoaderTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_load(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/demo.docx"))
        d = loader.load()
        self.assertEqual(1, len(d))

    def test_load_with_image(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/demo.docx"), image_inline=True)
        d = loader.load()
        self.assertEqual(1, len(d))


if __name__ == '__main__':
    unittest.main()
