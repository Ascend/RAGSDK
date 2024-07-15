# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from unittest.mock import patch

from mx_rag.tools.finetune.generator import TrainDataGenerator



class TestTrainDataGenerator(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(current_dir, "../../data/")
        self.train_data_generator = TrainDataGenerator(None, "", self.file_path, "", "")

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("mx_rag.tools.finetune.generator.train_data_generator.TrainDataGenerator._generate_origin_document")
    def test_process_origin_document(self, generate_mock, dir_check_mock):
        generate_mock.return_value = []
        result = self.train_data_generator.process_origin_document()
        self.assertEqual([], result)

    @patch("mx_rag.tools.finetune.generator.train_data_generator.TrainDataGenerator._generate_qd_pairs")
    def test_generate_coarsest_qd_pairs(self, generate_qd_pairs_mock):
        mock_value = [], []
        generate_qd_pairs_mock.return_value = mock_value
        result = self.train_data_generator.generate_coarsest_qd_pairs([], 1)
        self.assertEqual(mock_value, result)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    def test_save_train_data_and_rewrite(self, dir_check_mock):
        with self.assertRaises(Exception):
            result = self.train_data_generator.save_train_data_and_rewrite([], [], 1)
