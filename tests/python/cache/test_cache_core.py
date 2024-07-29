# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from typing import Dict

from gptcache.manager import get_data_manager
from gptcache.similarity_evaluation import ExactMatchEvaluation
from gptcache import Cache

from mx_rag.cache.cache_core import MXRAGCache


def get_data(data: Dict[str, str], **kwargs) -> str:
    return data['prompt']


def init_memory_cache(cache_obj: Cache, cache_name: str):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    data_manager = get_data_manager(
        data_path=os.path.join(current_dir, f"{cache_name}_data_map.txt")
    )

    cache_obj.init(
        pre_func=get_data,
        similarity_evaluation=ExactMatchEvaluation(),
        data_manager=data_manager
    )


def remove_cache_file(cache_name: str):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_file_path = os.path.join(current_dir, f"{cache_name}_data_map.txt")

    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)


class TestCacheCore(unittest.TestCase):
    def test_core_init(self):
        mxrag_cache = MXRAGCache()
        self.assertIsInstance(mxrag_cache, MXRAGCache)

    def test_data_check(self):
        data = None
        self.assertFalse(MXRAGCache.data_check(data))

        data = 123
        self.assertFalse(MXRAGCache.data_check(data))

        data = ""
        self.assertFalse(MXRAGCache.data_check(data))

        data = "123"
        self.assertTrue(MXRAGCache.data_check(data))

    def test_cache_update(self):
        remove_cache_file("l1_cache")

        mxrag_cache = MXRAGCache()
        mxrag_cache.register_init_func("l1_cache", init_memory_cache, lazy_init=False)
        self.assertRaises(ValueError, mxrag_cache.update, "", "l1_cache", "")
        self.assertRaises(NameError, mxrag_cache.update, "hell world", "", "yes world")

        mxrag_cache.update("hello world", "l1_cache", "yes i am")
        answer = mxrag_cache.lookup("hello world", "l1_cache")
        self.assertEqual(answer, "yes i am")

    def test_cache_join(self):
        remove_cache_file("l1_cache")
        remove_cache_file("l2_cache")

        mxrag_cache = MXRAGCache(init_memory_cache)
        mxrag_cache.register_init_func("l1_cache", init_memory_cache, lazy_init=True)
        mxrag_cache.join("l1_cache", "l2_cache")  # l1 l2 cache will create this time

        answer = mxrag_cache.lookup("hello world", "l1_cache")
        self.assertEqual(answer, None)

        mxrag_cache.update("hello world", "l1_cache", "i am bob")
        answer = mxrag_cache.lookup("hello world", "l1_cache")
        self.assertEqual(answer, "i am bob")

        answer = mxrag_cache.lookup("hello world", "l2_cache")
        self.assertEqual(answer, "i am bob")

        mxrag_cache.unjoin("l1_cache", "l2_cache")

        mxrag_cache.update("hello world", "l1_cache", "tcp is so good")
        answer = mxrag_cache.lookup("hello world", "l1_cache")
        self.assertEqual(answer, "tcp is so good")

        answer = mxrag_cache.lookup("hello world", "l2_cache")
        self.assertEqual(answer, "i am bob")

    def test_cache_clear(self):
        remove_cache_file("l1_cache")

        mxrag_cache = MXRAGCache()
        mxrag_cache.register_init_func("l1_cache", init_memory_cache, lazy_init=False)

        mxrag_cache.update("hello world", "l1_cache", "i am bob")
        answer = mxrag_cache.lookup("hello world", "l1_cache")
        self.assertEqual(answer, "i am bob")

        mxrag_cache.clear()
        remove_cache_file("l1_cache")
        answer = mxrag_cache.lookup("hello world", "l1_cache")
        self.assertEqual(answer, None)

        remove_cache_file("l1_cache")
        remove_cache_file("l2_cache")


if __name__ == '__main__':
    unittest.main()
