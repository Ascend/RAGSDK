# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import patch

import numpy as np

from mx_rag.tools.finetune.dataprocess.reranker_featured import reranker_featured


class TestRerankerFeatured(unittest.TestCase):

    @patch("mx_rag.reranker.local.LocalReranker.__init__")
    @patch("mx_rag.reranker.local.LocalReranker.rerank")
    def test_run_success(self, fake_rerank, fake_init):
        def f_reranker(query: str, texts: list[str]):
            return np.array([1] * len(texts))

        fake_init.return_value = None
        fake_rerank.side_effect = f_reranker

        scores = reranker_featured("test_rerank_name", ["query"], ["doc"])
        self.assertEqual(scores, [1])


if __name__ == '__main__':
    unittest.main()
