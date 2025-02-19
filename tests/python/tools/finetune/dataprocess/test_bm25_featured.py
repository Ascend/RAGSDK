# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest

from mx_rag.tools.finetune.dataprocess.bm25_featured import bm25_featured


class TestBm25Featured(unittest.TestCase):

    def test_run_success(self):
        query = ['a']
        doc = ['a']
        scores = bm25_featured(query, doc)

        self.assertEqual(scores, [0])


if __name__ == '__main__':
    unittest.main()
