# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from unittest.mock import patch

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.generator import EvalDataGenerator
from mx_rag.utils import ClientParam


class TestEvalDataGenerator(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(current_dir, "../../../../data/")
        client_param = ClientParam(use_http=True, timeout=120)
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", client_param=client_param)
        self.eval_data_generator = EvalDataGenerator(llm, self.file_path)
        if os.path.exists(os.path.join(self.file_path, "evaluate_data.jsonl")):
            os.remove(os.path.join(self.file_path, "evaluate_data.jsonl"))
 
    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("mx_rag.utils.file_check.FileCheck.check_path_is_exist_and_valid")
    @patch("mx_rag.utils.file_check.SecFileCheck.check")
    @patch("mx_rag.tools.finetune.dataprocess.generate_qa_embedding_pairs")
    def test_generate_eval_data(self, generate_qa_mock, path_check_mock, eval_file_check_mock, dir_check_mock):
        generate_qa_mock.return_value = {"content1": ["q1"], "content2": ["q2"]}
        self.eval_data_generator.generate_evaluate_data(["test"])

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    def test_generate_origin_document(self, dir_check_mock):
        result = self.eval_data_generator.generate_origin_document("/test")
        self.assertEqual([], result)