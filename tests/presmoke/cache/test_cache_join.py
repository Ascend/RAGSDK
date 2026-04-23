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

import json
import os
import shutil
import unittest

from loguru import logger
from pymilvus import MilvusClient

from mx_rag.cache import (CacheConfig, EvictPolicy, MxRAGCache,
                          SimilarityCacheConfig)


class TestSimilarityCacheDemo(unittest.TestCase):
    def setUp(self):
        self.path_to_cache_save_folder = os.path.join(os.path.realpath(os.path.dirname(__file__)), "./rag_cache")
        if os.path.exists(self.path_to_cache_save_folder):
            shutil.rmtree(self.path_to_cache_save_folder)
        os.makedirs(self.path_to_cache_save_folder)

    def test_cache(self):
        dim = 1024
        milvus_url: str = "http://my-release-milvus.milvus:19530"
        client = MilvusClient(milvus_url, server_name="localhost")
        dev_id = 0
        cache_config = CacheConfig(
            cache_size=100,
            eviction_policy=EvictPolicy.LRU,
            data_save_folder=self.path_to_cache_save_folder
        )
        cache = MxRAGCache("memory_cache", cache_config)
        # 检查cache实例是否初始化成功
        cache_obj = cache.get_obj()
        if cache_obj is None:
            logger.error(f"cache init failed")
        similarity_config = SimilarityCacheConfig(
            vector_config={
                "client": client,
                "vector_type": "milvus_db",
                "x_dim": dim,
                "collection_name": "mxrag_cache",  # milvus db的标签
                "param": None
            },
            cache_config="sqlite",
            emb_config={
                "embedding_type": "local_text_embedding",
                "x_dim": dim,
                "model_path": "/home/data/bge-large-zh-v1.5",  # emb 模型路径
                "dev_id": 0
            },
            similarity_config={
                "similarity_type": "local_reranker",
                "model_path": "/home/data/bge-reranker-large",  # reranker 模型路径
                "dev_id": 0
            },
            retrieval_top_k=1,
            cache_size=100,
            auto_flush=100,
            similarity_threshold=0.70,
            data_save_folder=self.path_to_cache_save_folder,
            disable_report=True,
            eviction_policy=EvictPolicy.FIFO
        )
        # cache 初始化
        similarity_cache = MxRAGCache("similarity_cache", similarity_config)
        # 检查cache实例是否初始化成功
        smi_cache_obj = similarity_cache.get_obj()
        if smi_cache_obj is None:
            logger.error("similarity cache init failed")
        # 设置缓存级联
        cache.join(similarity_cache)
        # 设置缓存每条的字符限制为4000个字符
        cache.set_cache_limit(4000)
        # 设置是否详细显示缓存过程
        cache.set_verbose(True)
        # 手动更新缓存
        answer = json.dumps({"小明的爸爸是谁?": "小明的爸爸名字是大明"})
        cache.update("小明的爸爸是谁?", answer)
        # 精确匹配结果
        res = cache.search("小明的爸爸是谁?")
        self.assertEqual(res, answer)
        # 语义近似匹配结果
        res = cache.search("小明的爸爸叫什么名字?")
        self.assertEqual(res, answer)


if __name__ == '__main__':
    unittest.main()
