# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from unittest.mock import patch

from mx_rag.utils.file_check import FileCheckError
from mx_rag.utils.file_operate import write_jsonl_to_file, read_jsonl_from_file


class TestFileOperate(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(current_dir, "../../data/finetune.jsonl")
        self.datas = [{"query": "question1", "pos": ["doc content1"]}, {"query": "question2", "pos": ["doc content2"]}]

    def test_write_jsonl_to_file_exceed_limit(self):
        with self.assertRaises(Exception):
            write_jsonl_to_file(self.datas, self.file_path, 1)
        with self.assertRaises(FileCheckError):
            write_jsonl_to_file(self.datas, self.file_path)

    def test_read_jsonl_from_file_exception(self):
        with self.assertRaises(FileCheckError):
            read_jsonl_from_file(self.file_path)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("mx_rag.utils.file_check.FileCheck.check_input_path_valid")
    def test_write_and_read_jsonl_file(self, dir_check_mock, check_input_mock):
        path = os.path.realpath(self.file_path)
        write_jsonl_to_file(self.datas, path)
        datas = read_jsonl_from_file(path)
        self.assertEqual(self.datas, datas)
