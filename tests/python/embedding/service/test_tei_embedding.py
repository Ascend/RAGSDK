# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import Dict
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np

from mx_rag.embedding.service import TEIEmbedding


class TestTEIEmbedding(unittest.TestCase):
    class Result:
        def __init__(self, success: bool, data: str):
            self.success = success
            self.data = data

    def test_request_success(self):
        test_embed_length = 1024

        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)
            response_data = []
            for i in range(len(data['inputs'])):
                response_data.append(np.random.rand(test_embed_length).tolist())
            return TestTEIEmbedding.Result(True, json.dumps(response_data))

        with patch('mx_rag.utils.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='http://localhost:8888')

            texts = ['abc'] * 100
            encoded_texts = embed.embed_texts(texts=texts)
            self.assertEqual(encoded_texts.shape, (len(texts), test_embed_length))

            texts = ['abc'] * 1000
            encoded_texts = embed.embed_texts(texts=texts)
            self.assertEqual(encoded_texts.shape, (len(texts), test_embed_length))

    def test_empty_texts(self):
        embed = TEIEmbedding(url='http://localhost:8888')

        texts = []
        encoded_texts = embed.embed_texts(texts=texts)
        self.assertEqual(encoded_texts.shape, (0,))

    def test_request_failed(self):
        def mock_post(*args, **kwargs):
            return TestTEIEmbedding.Result(False, "")

        with patch('mx_rag.utils.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='http://localhost:8888')

            texts = ['abc'] * 100
            encoded_texts = embed.embed_texts(texts=texts)
            self.assertEqual(encoded_texts.shape, (0,))


if __name__ == '__main__':
    unittest.main()
