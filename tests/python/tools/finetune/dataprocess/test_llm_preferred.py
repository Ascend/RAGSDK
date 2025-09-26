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
        chat.return_value = "得分是62.13"
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        scores = llm_preferred(llm, ["query"], ["document"], SCORING_QD_PROMPT)

        self.assertEqual(scores, [62.13])


if __name__ == '__main__':
    unittest.main()
