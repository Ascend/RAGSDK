# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from unittest.mock import patch

from mx_rag.tools.finetune.generator import EvalDataGenerator


class TestEvalDataGenerator(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(current_dir, "../../../../data/")
        self.eval_data_generator = EvalDataGenerator(None, self.file_path, "")
        if os.path.exists(os.path.join(self.file_path, "eval_data.jsonl")):
            os.remove(os.path.join(self.file_path, "eval_data.jsonl"))

    @patch("mx_rag.utils.FileCheck.dir_check")
    @patch("mx_rag.tools.finetune.generator.eval_data_generator.SecFileCheck.check")
    @patch("mx_rag.utils.FileCheck.check_path_is_exist_and_valid")
    @patch("mx_rag.tools.finetune.generator.eval_data_generator.generate_qa_embedding_pairs")
    @patch("mx_rag.tools.finetune.generator.eval_data_generator.BaseGenerator._feature_qd_pair")
    @patch("mx_rag.tools.finetune.generator.eval_data_generator.BaseGenerator._prefer_qd_pair")
    def test_generate_eval_data(self, prefer_qd_mock, feature_qd_mock, generate_qa_mock, path_check_mock, eval_file_check_mock, dir_check_mock):
        prefer_qd_mock.return_value = ['q1', 'q2'], ['content1', 'content2']
        feature_qd_mock.return_value = ['q1', 'q2'], ['content1', 'content2']
        generate_qa_mock.return_value = {"content1": ["q1"], "content2": ["q2"]}
        self.eval_data_generator.generate_eval_data()