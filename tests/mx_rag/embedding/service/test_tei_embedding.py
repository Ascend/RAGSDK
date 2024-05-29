# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import Dict
import unittest
from unittest import mock

import numpy as np

from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils import RequestUtils


class TestTEIEmbedding(unittest.TestCase):
    class Result:
        def __init__(self, success: bool, data: str):
            self.success = success
            self.data = data

    class MockResponse:
        def __init__(self, content, status_code):
            self.content = content
            self.status_code = status_code

        def json(self):
            return json.loads(self.content)

    def test_request_success(self):
        test_embed_length = 1024

        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)
            response_data = []
            for i in range(len(data['inputs'])):
                response_data.append(np.random.rand(test_embed_length).tolist())
            return TestTEIEmbedding.Result(True, json.dumps(response_data))

        RequestUtils.post = mock.Mock(side_effect=mock_post)

        embed = TEIEmbedding(url='http://localhost:8888')

        texts = ['abc'] * 100
        encoded_texts = embed.encode(texts=texts)
        self.assertEqual(len(texts), len(encoded_texts))

        texts = ['abc'] * 1000
        encoded_texts = embed.encode(texts=texts)
        self.assertEqual(len(texts), len(encoded_texts))

    def test_empty_texts(self):
        embed = TEIEmbedding(url='http://localhost:8888')

        texts = []
        encoded_texts = embed.encode(texts=texts)
        self.assertEqual(len(texts), len(encoded_texts))

    def test_request_failed(self):
        def mock_post(*args, **kwargs):
            return TestTEIEmbedding.Result(False, "")

        RequestUtils.post = mock.Mock(side_effect=mock_post)

        embed = TEIEmbedding(url='http://localhost:8888')

        texts = ['abc'] * 100
        encoded_texts = embed.encode(texts=texts)
        self.assertEqual(0, len(encoded_texts))


if __name__ == '__main__':
    unittest.main()
