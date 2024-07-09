# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import patch

from mx_rag.tools.finetune.train.train_embedding import train_embedding


class TestTrainEmbedding(unittest.TestCase):
    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("mx_rag.utils.file_check.SecFileCheck.check")
    @patch("subprocess.run")
    def test_run_success(self, fake_run, fake_check, fake_dir_check):
        fake_run.return_value = self.Result(0)
        train_embedding("test_reranker_path", "test_new_reranker_path", "dataset_path")


if __name__ == '__main__':
    unittest.main()
