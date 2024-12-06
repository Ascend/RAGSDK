# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import random
import shutil
from typing import Dict
import unittest
from unittest.mock import patch

import torch

from mx_rag.reranker.local import LocalReranker


class TestLocalReranker(unittest.TestCase):
    model_path = "/model/reranker"

    def setUp(self) -> None:
        os.makedirs(self.model_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_path)

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
        def __init__(self):
            self.device = 'cpu'

        def __call__(self, *args, **kwargs):
            input_ids = kwargs.pop('input_ids')
            logits = torch.rand(len(input_ids), 1)
            return TestLocalReranker.BatchEncoding(logits=logits)

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
            return TestLocalReranker.BatchEncoding(input_ids=input_ids)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_rerank_success_fp16(self,
                                 torch_avail_mock,
                                 tok_pre_mock,
                                 model_pre_mock,
                                 dir_check_mock):
        model_pre_mock.return_value = TestLocalReranker.Model()
        tok_pre_mock.return_value = TestLocalReranker.Tokenizer()
        torch_avail_mock.return_value = True

        rerank = LocalReranker(model_path=self.model_path)
        texts = ['我是小黑', '我是小红'] * 100
        ret = rerank.rerank(query='你好', texts=texts)

        self.assertEqual(ret.shape, (len(texts),))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_rerank_success_fp32(self,
                                 torch_avail_mock,
                                 tok_pre_mock,
                                 model_pre_mock,
                                 dir_check_mock):
        model_pre_mock.return_value = TestLocalReranker.Model()
        tok_pre_mock.return_value = TestLocalReranker.Tokenizer()
        torch_avail_mock.return_value = False

        rerank = LocalReranker(model_path=self.model_path,
                               use_fp16=False)
        texts = ['我是小黑', '我是小红'] * 100
        ret = rerank.rerank(query='你好', texts=texts)

        self.assertEqual(ret.shape, (len(texts),))


if __name__ == '__main__':
    unittest.main()
