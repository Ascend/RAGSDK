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

from mx_rag.embedding import EmbeddingFactory
from mx_rag.utils import ClientParam


class TestEmbeddingFactory(unittest.TestCase):
    def test_embedding(self):
        dev_id = 0
        txt_embed = EmbeddingFactory.create_embedding(embedding_type="local_text_embedding",
                                                      model_path="/home/data/bge-large-zh-v1.5", dev_id=dev_id)
        self.assertEqual(len(txt_embed.embed_query("abc")), 1024)
        # 根据实际情况修改参数
        tei_embed = EmbeddingFactory.create_embedding(embedding_type="tei_embedding",
                                                      url="http://127.0.0.1:8000/v1/embeddings",
                                                      client_param=ClientParam(use_http=True))
        self.assertEqual(len(tei_embed.embed_query("abc")), 1024)

        img_embed = EmbeddingFactory.create_embedding(embedding_type="local_images_embedding", model_name="RN50",
                                                      model_path="/home/data/chinese-clip-rn50", dev_id=dev_id)
        self.assertEqual(len(img_embed.embed_query("abc")), 1024)


if __name__ == '__main__':
    unittest.main()
