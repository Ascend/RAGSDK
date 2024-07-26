# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List
import unittest
from unittest import mock
from unittest.mock import patch

from mx_rag.reranker.reranker import Reranker
from mx_rag.cache.cache_similarity import CacheSimilarity
from cache_mocker import MockerReranker


def mock_create_similarity(*args, **kwargs):
    return MockerReranker(0)


class TestCacheSimilarity(unittest.TestCase):
    def test_cache_similarity_init_exception(self):
        self.assertRaises(ValueError, CacheSimilarity.create, **{
            "similarity_type": 1234  # type error
        })

        self.assertRaises(KeyError, CacheSimilarity.create, **{
            "similarity_type": "xxxx"  # type error
        })

        self.assertRaises(KeyError, CacheSimilarity.create, **{
            "xxxx": "xxxx"  # type error
        })

    def test_cache_similarity(self):
        with patch('mx_rag.reranker.reranker_factory.RerankerFactory.create_reranker',
                   mock.Mock(side_effect=mock_create_similarity)):
            similarity = CacheSimilarity.create(similarity=1234)

            src_dict = {
                "question": "hello world",
            }

            cache_dict = {
                "question": "hello world",
            }
            self.assertEqual(similarity.evaluation(src_dict, cache_dict), 1)

            cache_dict = {
                "question": "hello",
            }

            self.assertEqual(similarity.evaluation(src_dict, cache_dict), 0)

    def test_cache_similarity_range(self):
        with patch('mx_rag.reranker.reranker_factory.RerankerFactory.create_reranker',
                   mock.Mock(side_effect=mock_create_similarity)):
            similarity = CacheSimilarity.create(similarity_type=1234)

            score_min, score_max = similarity.range()
            self.assertEqual(score_min, 0.0)
            self.assertEqual(score_max, 1.0)

            similarity = CacheSimilarity.create(similarity_type=1234, score_min=2.0, score_max=4.0)

            score_min, score_max = similarity.range()
            self.assertEqual(score_min, 2.0)
            self.assertEqual(score_max, 4.0)


if __name__ == '__main__':
    unittest.main()
