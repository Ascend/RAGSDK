#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest
from unittest.mock import MagicMock

from mx_rag.cache import CacheChainChat, MxRAGCache
from mx_rag.chain import SingleText2TextChain
from mx_rag.llm import LLMParameterConfig


def _convert_data_to_user(data):
    return "world!"


def _convert_data_to_cache(data):
    return "hello" + data


class TestCacheChain(unittest.TestCase):
    def test_query(self):
        cache = MagicMock(spec=MxRAGCache)
        text2text_chain = MagicMock(spec=SingleText2TextChain)
        cache_chain = CacheChainChat(cache, text2text_chain, _convert_data_to_cache, _convert_data_to_user)
        llm_config = MagicMock(spec=LLMParameterConfig)
        # json无法解析时直接返回原值
        cache.search.return_value = "return value test"
        result = cache_chain.query("test", llm_config)
        self.assertEqual(result, "return value test")
        # 能解析json调用_convert_data_to_user处理返回值
        cache.search.return_value = ('{"query": "\u9ad8\u8003\u9898\u76ee\u662f\u4ec0\u4e48\uff1f",'
                                     ' "result": "\u8bed\u6587\u5168\u56fd\u5377"}')
        result = cache_chain.query("test", llm_config)
        self.assertEqual(result, "world!")
        # search为None的情况
        cache.search.return_value = None
        text2text_chain.query.return_value = "大模型返回"
        cache.update.return_value = None
        result = cache_chain.query("test", llm_config)
        self.assertEqual(result, "大模型返回")