#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import unittest
from unittest.mock import patch

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess.llm_preferred import llm_preferred, SCORING_QD_PROMPT


class TestLlmPreferred(unittest.TestCase):

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_run_success(self, chat):
        chat.return_value = "得分是0.13"
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        scores = llm_preferred(llm, ["query"], ["document"], SCORING_QD_PROMPT)

        self.assertEqual(scores, [0.13])

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_run_border_success(self, chat):
        chat.return_value = "得分是1分"
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        scores = llm_preferred(llm, ["query"], ["document"], SCORING_QD_PROMPT)

        self.assertEqual(scores, [1.0])

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_make_request_type_error(self, chat):
        chat.side_effect = TypeError("Invalid argument type")
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        scores = llm_preferred(llm, ["query"], ["document"], SCORING_QD_PROMPT)

        self.assertEqual(scores, [0.0])

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_make_request_timeout_error(self, chat):
        chat.side_effect = TimeoutError("Request timed out")
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        scores = llm_preferred(llm, ["query"], ["document"], SCORING_QD_PROMPT)

        self.assertEqual(scores, [0.0])

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_make_request_generic_exception(self, chat):
        chat.side_effect = Exception("Unknown error occurred")
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        scores = llm_preferred(llm, ["query"], ["document"], SCORING_QD_PROMPT)

        self.assertEqual(scores, [0.0])


if __name__ == '__main__':
    unittest.main()
