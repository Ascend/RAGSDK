# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
from unittest.mock import patch

from langchain_core.prompts import PromptTemplate
from mx_rag.llm import Text2TextLLM
from mx_rag.summary.summary import Summary


class TestSummary(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        self.llm = Text2TextLLM(base_url="http://127.0.0.1:1025/v1/chat/completions", model_name="qianwen-7b",
                                timeout=60,
                                use_http=True)

    def teardown_method(self, method):
        pass

    def mock_summary(self, text: str, prompt: PromptTemplate) -> str:
        return text[0:len(text) - 1]

    def test_summarize_none_str(self):
        su = Summary(llm=self.llm)
        res = True
        try:
            su.summarize([""])
        except Exception as e:
            res = False

        self.assertEqual(res, False)

    @patch("mx_rag.summary.summary.Summary._summarize")
    def test_summarize(self, summary_mock):
        summary_mock.side_effect = self.mock_summary
        su = Summary(llm=self.llm)
        res = su.summarize(["aaa", "1b2", "2c3", "ddd"], not_summarize_threshold=2)
        self.assertEqual(len(res), len(res))
        self.assertEqual(res[1], "1b")
        self.assertEqual(res[2], "2c")

    def test_split_summary_by_threshold(self):
        su = Summary(llm=self.llm)
        res = su._split_summary_by_threshold(["aaa"], merge_threshold=1024)
        self.assertSequenceEqual(res, [(0, 0)])

        res = su._split_summary_by_threshold(["aaa", "1b2"], merge_threshold=1024)
        self.assertSequenceEqual(res, [(0, 1)])

        res = su._split_summary_by_threshold(["aaa", "b"*1000, "2c3"*20], merge_threshold=1024)
        self.assertSequenceEqual(res, [(0, 1), (2, 2)])

        res = su._split_summary_by_threshold(["aaa", "2"*1020, "2c3", "ddd", "67"], merge_threshold=1024)
        self.assertSequenceEqual(res, [(0, 1), (2, 4)])

        res = su._split_summary_by_threshold(["11"*1024, "22"*1024, "111111"*1024, "33", "33"], merge_threshold=1024)
        self.assertSequenceEqual(res, [(0, 0), (1, 1), (2, 2), (3, 4)])

        res = su._split_summary_by_threshold(["11111"*1024, "33", "33"], merge_threshold=1024)
        self.assertSequenceEqual(res, [(0, 0), (1, 2)])

    @patch("mx_rag.summary.summary.Summary._summarize")
    def test_merge_text_summarize(self, summary_mock):
        summary_mock.side_effect = self.mock_summary
        su = Summary(llm=self.llm)

        res = su.merge_text_summarize(["aaa"], merge_threshold=1024, not_summarize_threshold=4)
        self.assertSequenceEqual(res, "aaa")

        res = su.merge_text_summarize(["aaa"], merge_threshold=1024, not_summarize_threshold=2)
        self.assertSequenceEqual(res, "aa")

        res = su.merge_text_summarize(["aa", "eb2", "hc34"], merge_threshold=1024, not_summarize_threshold=2)
        self.assertSequenceEqual(res, "aa\n\neb2\n\nhc3")
