# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import random
from typing import Dict
import unittest
from unittest.mock import patch

import torch

from mx_rag.embedding.local import TextEmbedding


class TestTextEmbedding(unittest.TestCase):
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
            return TestTextEmbedding.BatchEncoding(last_hidden_state=last_hidden_state)

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
            return TestTextEmbedding.BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_encode_success_fp16_mean(self,
                                      torch_avail_mock,
                                      tok_pre_mock,
                                      model_pre_mock,
                                      dir_check_mock):
        model_pre_mock.return_value = self.Model(1024)
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = False

        embed = TextEmbedding(model_path='/model/embedding',
                              pooling_method='mean')

        texts = ['test_txt'] * 100
        ret = embed.embed_texts(texts=texts)
        self.assertEqual(ret.shape, (len(texts), 1024))

        texts = ['test_txt'] * 1000
        ret = embed.embed_texts(texts=texts)
        self.assertEqual(ret.shape, (len(texts), 1024))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_encode_success_fp32_cls(self,
                                     torch_avail_mock,
                                     tok_pre_mock,
                                     model_pre_mock,
                                     dir_check_mock):
        model_pre_mock.return_value = self.Model(512)
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = True

        embed = TextEmbedding(model_path='/model/embedding',
                              use_fp16=False)

        texts = ['test_txt'] * 100
        ret = embed.embed_texts(texts=texts)
        self.assertEqual(ret.shape, (len(texts), 512))

        texts = ['test_txt'] * 1000
        ret = embed.embed_texts(texts=texts)
        self.assertEqual(ret.shape, (len(texts), 512))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_encode_success_with_last_hidden_state(self,
                                                   torch_avail_mock,
                                                   tok_pre_mock,
                                                   model_pre_mock,
                                                   dir_check_mock):
        model_pre_mock.return_value = self.Model(1024)
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = False

        embed = TextEmbedding(model_path='/model/embedding',
                              pooling_method='mean')

        texts = ['test_txt'] * 100
        ret, lhs = embed.embed_texts_with_last_hidden_state(texts=texts)
        self.assertEqual(ret.shape, (len(texts), 1024))
        self.assertEqual(lhs.shape, (len(texts), 1024))

        texts = ['test_txt'] * 1000
        ret, lhs = embed.embed_texts_with_last_hidden_state(texts=texts)
        self.assertEqual(ret.shape, (len(texts), 1024))
        self.assertEqual(lhs.shape, (len(texts), 1024))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_encode_failed_invalid_pooling(self,
                                           torch_avail_mock,
                                           tok_pre_mock,
                                           model_pre_mock,
                                           dir_check_mock):
        model_pre_mock.return_value = self.Model(1024)
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = True

        embed = TextEmbedding(model_path='/model/embedding',
                              pooling_method='no valid')

        texts = ['test_txt'] * 100
        self.assertRaises(NotImplementedError, embed.embed_texts, texts=texts)


if __name__ == '__main__':
    unittest.main()
