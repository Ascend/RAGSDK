#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import random
from typing import Dict
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np

from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils import ClientParam


class TestTEIEmbedding(unittest.TestCase):
    class Result:
        def __init__(self, success: bool, data: str):
            self.success = success
            self.data = data

    def test_request_success(self):
        # 测试 /embed接口
        test_embed_length = 1024

        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)
            response_data = []
            for i in range(len(data['inputs'])):
                response_data.append(np.random.rand(test_embed_length).tolist())
            return TestTEIEmbedding.Result(True, json.dumps(response_data))

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='https://localhost:8888/embed', client_param=ClientParam(use_http=True))

            texts = ['abc'] * 100
            encoded_texts = embed.embed_documents(texts=texts)
            self.assertEqual(100, len(encoded_texts))

    def test_request_success_1(self):
        # 测试 /v1/embeddings接口
        test_embed_length = 1024

        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)

            response_data = {
                "object": "list",
                "data": [],
                "model": "bge-large-zh-v1.5",
                "usage": {
                    "prompt_tokens": 69,
                    "total_tokens": 69
                }
            }
            for i in range(len(data['input'])):
                response_data["data"].append(
                    {"object": "embedding", "embedding": np.random.rand(test_embed_length).tolist()})

            return TestTEIEmbedding.Result(True, json.dumps(response_data))

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='https://localhost:8888/v1/embeddings', client_param=ClientParam(use_http=True))

            texts = ['abc'] * 100

            encoded_texts = embed.embed_documents(texts=texts)

            self.assertEqual(100, len(encoded_texts))

    def test_request_success_2(self):
        # 测试 /embed_sparse接口
        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)

            response_data = []
            for input in data['inputs']:
                item = [{"index": random.randint(0,100), "value":random.uniform(1,100)} for j in range(len(input.split()))]
                response_data.append(item)
            return TestTEIEmbedding.Result(True, json.dumps(response_data))

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='https://localhost:8888/embed_sparse', client_param=ClientParam(use_http=True))

            texts = ["I like learn english.", "The capital of China is Beijing."]

            encoded_texts = embed.embed_documents(texts=texts)

            self.assertEqual(2, len(encoded_texts))

    def test_empty_texts(self):
        embed = TEIEmbedding(url='https://localhost:8888/embed', client_param=ClientParam(use_http=True))

        texts = []
        with self.assertRaises(ValueError):
            embed.embed_documents(texts=texts)

    def test_request_failed(self):
        def mock_post(*args, **kwargs):
            return TestTEIEmbedding.Result(False, "")

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='https://localhost:8888/embed', client_param=ClientParam(use_http=True))

            texts = ['abc'] * 100
            try:
                encoded_texts = embed.embed_documents(texts=texts)
            except Exception as e:
                self.assertEqual(f"{e}", "tei get response failed")


if __name__ == '__main__':
    unittest.main()
