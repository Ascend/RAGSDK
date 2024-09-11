# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
from unittest.mock import Mock, patch

from mx_rag.cache import QAGenerate, QAGenerationConfig, MarkDownParser, HTMLParser


class TestQAGenerate(unittest.TestCase):

    def test_generate_qas_length_not_equal(self):
        config = QAGenerationConfig(['title1', 'title2'], ['content1'], Mock(), Mock())
        qa_generate = QAGenerate(config)
        with self.assertRaises(ValueError):
            qa_generate.generate_qa()

    @patch("mx_rag.cache.QAGenerate.generate_qa")
    def test_generate_qas_no_qas(self, mock_generate_qas):
        config = QAGenerationConfig(['title1', 'title2'], ['content1'], Mock(), Mock())
        qa_generate = QAGenerate(config)
        mock_generate_qas.return_value = []
        result = qa_generate.generate_qa()
        self.assertEqual(result, [])

    @patch("mx_rag.cache.QAGenerate._split_html_text")
    @patch("mx_rag.cache.QAGenerate._generate_qa_from_html")
    def test_generate_qas_with_qas(self, generate_mock, split_mock):
        config = QAGenerationConfig(['title1', 'title2'], ['content1', 'content2'], Mock(), Mock())
        qa_generate = QAGenerate(config)
        generate_mock.return_value = ["q1?参考段落:answer1", "q2?参考段落:answer2"]
        split_mock.return_value = "text"
        result = qa_generate.generate_qa()
        self.assertEqual(result, {'q1?': 'answer1', 'q2?': 'answer2'})

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    def test_markdown_parse(self, dir_check_mock):
        # 创建MarkDownParser实例
        test_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data"))
        parser = MarkDownParser(test_dir)
        # 调用parse方法
        titles, contents = parser.parse()
        # 验证结果
        self.assertEqual(titles, ['test.md'])
        self.assertEqual(contents, ['# Test Tile\n\nthis is a test'])

    @patch("mx_rag.cache.HTMLParser.parse")
    def test_html_parse(self, parse_mock):
        html_parser = HTMLParser(["https://127.0.0.1"])
        parse_mock.return_value = [], []
        titles, contents = html_parser.parse()
        self.assertEqual(titles, [])
        self.assertEqual(contents, [])


if __name__ == '__main__':
    unittest.main()
