#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import random
from typing import Dict
import unittest
from unittest import mock
from unittest.mock import patch

from mx_rag.reranker.service import TEIReranker
from mx_rag.utils import ClientParam


class TestTEIReranker(unittest.TestCase):
    class Result:
        def __init__(self, success: bool, data: str):
            self.success = success
            self.data = data

    def test_request_success(self):
        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)
            response_data = []
            for i in range(len(data['texts'])):
                response_data.append({'index': i, 'score': random.random()})
            return TestTEIReranker.Result(True, json.dumps(response_data))

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            rerank = TEIReranker(url='https://localhost:8888', client_param=ClientParam(use_http=True))

            texts = ['我是小黑', '我是小红'] * 100
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (len(texts),))

            texts = ['我是小黑', '我是小红'] * 300
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (len(texts),))

    def test_empty_texts(self):
        rerank = TEIReranker(url='https://localhost:8888', client_param=ClientParam(use_http=True))

        texts = ["text"]
        scores = rerank.rerank(query='你好', texts=texts)
        self.assertEqual(scores.shape, (0,))

    def test_texts_too_long(self):
        rerank = TEIReranker(url='https://localhost:8888', client_param=ClientParam(use_http=True))

        texts = ['我是小黑', '我是小红'] * 500001
        with self.assertRaises(ValueError):
            rerank.rerank(query='你好', texts=texts)

    def test_request_failed(self):
        def mock_post(url: str, body: str, headers: Dict):
            return TestTEIReranker.Result(False, "")

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            rerank = TEIReranker(url='https://localhost:8888', client_param=ClientParam(use_http=True))

            texts = ['我是小黑', '我是小红'] * 300
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (0,))


if __name__ == '__main__':
    unittest.main()
