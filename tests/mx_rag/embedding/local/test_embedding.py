# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import random
from typing import Dict
import unittest
from unittest import mock

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from mx_rag.embedding.local import LocalEmbedding
import mx_rag.utils as m_utils


class TestLocalEmbedding(unittest.TestCase):
    class BatchEncoding(Dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                error_msg = f'Dict object has no attribute [{key}]'
                raise AttributeError(error_msg)

        def __setattr__(self, key, value):
            self[key] = value

        def to(self, *args):
            return self

    class Model:
        def __init__(self, test_embed_length):
            self.device = 'cpu'
            self.test_embed_length = test_embed_length

        def __call__(self, *args, **kwargs):
            input_ids = args[0]
            last_hidden_state = torch.rand(input_ids.shape + (self.test_embed_length,))
            return TestLocalEmbedding.BatchEncoding(last_hidden_state=last_hidden_state)

        def half(self):
            return self

        def to(self, *args):
            return self

        def eval(self):
            return self

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            batch_text = args[0]
            max_length = kwargs.pop('max_length', 512)
            rand_token_len = random.randint(1, max_length)
            input_ids = torch.rand((len(batch_text), rand_token_len))
            attention_mask = torch.ones((len(batch_text), rand_token_len))
            return TestLocalEmbedding.BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def test_encode_success_fp16_mean(self):
        m_utils.dir_check = mock.Mock()
        AutoModel.from_pretrained = mock.Mock(return_value=TestLocalEmbedding.Model(1024))
        AutoTokenizer.from_pretrained = mock.Mock(return_value=TestLocalEmbedding.Tokenizer())
        transformers.is_torch_npu_available = mock.Mock(return_value=False)

        embed = LocalEmbedding(model_name_or_path='/model/embedding',
                               pooling_method='mean')

        texts = ['test_txt'] * 100
        ret = embed.encode(texts=texts)
        self.assertEqual(len(ret), len(texts))
        self.assertEqual(len(ret[0]), 1024)

        texts = ['test_txt'] * 1000
        ret = embed.encode(texts=texts)
        self.assertEqual(len(ret), len(texts))
        self.assertEqual(len(ret[0]), 1024)

    def test_encode_success_fp32_cls(self):
        m_utils.dir_check = mock.Mock()
        AutoModel.from_pretrained = mock.Mock(return_value=TestLocalEmbedding.Model(512))
        AutoTokenizer.from_pretrained = mock.Mock(return_value=TestLocalEmbedding.Tokenizer())
        transformers.is_torch_npu_available = mock.Mock(return_value=True)

        embed = LocalEmbedding(model_name_or_path='/model/embedding',
                               use_fp16=False)

        texts = ['test_txt'] * 100
        ret = embed.encode(texts=texts)
        self.assertEqual(len(ret), len(texts))
        self.assertEqual(len(ret[0]), 512)

        texts = ['test_txt'] * 1000
        ret = embed.encode(texts=texts)
        self.assertEqual(len(ret), len(texts))
        self.assertEqual(len(ret[0]), 512)

    def test_encode_failed_invalid_pooling(self):
        m_utils.dir_check = mock.Mock()
        AutoModel.from_pretrained = mock.Mock(return_value=TestLocalEmbedding.Model(1024))
        AutoTokenizer.from_pretrained = mock.Mock(return_value=TestLocalEmbedding.Tokenizer())
        transformers.is_torch_npu_available = mock.Mock(return_value=True)

        embed = LocalEmbedding(model_name_or_path='/model/embedding',
                               pooling_method='no valid')

        texts = ['test_txt'] * 100
        self.assertRaises(NotImplementedError, embed.encode, texts=texts)


if __name__ == '__main__':
    unittest.main()
