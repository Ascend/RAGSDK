#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest

from mx_rag.embedding.local import SparseEmbedding


class TestSparseEmbedding(unittest.TestCase):
    def test_sparse_embedding(self):
        dev_id = 0
        embed = SparseEmbedding.create(model_path="/home/data/bge-m3", dev_id=dev_id)
        embedding1 = embed.embed_documents(['abc', 'bcd'])
        self.assertEqual(len(embedding1), 2)
        self.assertIsInstance(embedding1[0], dict)
        embedding2 = embed.embed_query('abc')
        self.assertIsInstance(embedding2, dict)
        self.assertIn(1563, embedding2.keys())


if __name__ == '__main__':
    unittest.main()
