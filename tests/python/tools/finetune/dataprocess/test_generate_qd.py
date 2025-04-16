# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import patch

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess.generate_qd import generate_qa_embedding_pairs, GENERATE_QD_PROMPT


class TestGenerateQD(unittest.TestCase):

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_run_success(self, chat):
        chat.return_value = "question?"
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        qd = generate_qa_embedding_pairs(llm, ["hello"], GENERATE_QD_PROMPT, 1)
        self.assertEqual(qd["hello"], ["question?"])


if __name__ == '__main__':
    unittest.main()
