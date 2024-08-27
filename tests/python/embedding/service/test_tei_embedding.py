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

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='https://localhost:8888')

            texts = ['abc'] * 100
            try:
                encoded_texts = embed.embed_documents(texts=texts)
            except Exception as e:
                self.assertEqual(f"{e}", "texts length equal 0")


            texts = ['abc'] * 1000
            try:
                encoded_texts = embed.embed_documents(texts=texts)
            except Exception as e:
                self.assertEqual(f"{e}", f'texts length greater than {TEIEmbedding.TEXT_MAX_LEN}')

    def test_empty_texts(self):
        embed = TEIEmbedding(url='https://localhost:8888')

        texts = []
        with self.assertRaises(ValueError):
            embed.embed_documents(texts=texts)


    def test_request_failed(self):
        def mock_post(*args, **kwargs):
            return TestTEIEmbedding.Result(False, "")

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            embed = TEIEmbedding(url='https://localhost:8888')

            texts = ['abc'] * 100
            try:
                encoded_texts = embed.embed_documents(texts=texts)
            except Exception as e:
                self.assertEqual(f"{e}", "tei get response failed")


if __name__ == '__main__':
    unittest.main()
