# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
from unittest.mock import patch

import numpy as np

from mx_rag.tools.finetune.dataprocess.mine_hard_negative import MineHardNegative


class TestMineHn(unittest.TestCase):

    @patch("mx_rag.utils.file_check.FileCheck.check_input_path_valid")
    @patch("mx_rag.utils.file_check.SecFileCheck.check")
    @patch("mx_rag.embedding.local.TextEmbedding.__init__")
    @patch("mx_rag.embedding.local.TextEmbedding.embed_documents")
    def test_run_success(self, fake_embed_documents, fake_embed_init, fake_check, fake_check_path):
        def embed_text(texts: list[str]):
            return np.array([[1] * 1024] * len(texts))

        fake_embed_documents.side_effect = embed_text
        fake_embed_init.return_value = None

        train_datas = []
        for i in range(1000):
            train_data = {'query': f'q{i}', 'pos': [f'p{i}_1', f'p{i}_2'], 'neg': [f'neg{i}_1', f'neg{i}_2']}
            train_datas.append(train_data)

        mhn = MineHardNegative("test_embed_path")
        mhn.find_knn_neg(train_datas, [0, 2], 2)


if __name__ == '__main__':
    unittest.main()
