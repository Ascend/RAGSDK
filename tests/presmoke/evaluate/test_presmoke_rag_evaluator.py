#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest
from unittest.mock import MagicMock, patch

from ragas.evaluation import EvaluationResult

from mx_rag.embedding.local import TextEmbedding
from mx_rag.evaluate import RAGEvaluator
from mx_rag.llm import Text2TextLLM


class TestRAGEvaluator(unittest.TestCase):
    @patch("mx_rag.evaluate.rag_evaluator.evaluate", autospec=True)
    def test_rag_evaluator(self, evaluate_mock):
        dataset = {
            "user_input": ["世界上最高的山峰是哪座？"],
            "response": ["珠穆朗玛峰"],
            "retrieved_contexts": [["世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8848米。"]],
        }
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        scores = [{"answer_relevancy": 0.09, "faithfulness": 0.5}]
        result = MagicMock(spec=EvaluationResult)
        result.scores = scores
        evaluate_mock.return_value = result
        output = evaluator.evaluate(metrics=["answer_relevancy", "faithfulness"], dataset=dataset, language="chinese")
        self.assertIn("answer_relevancy", output)
        self.assertIn("faithfulness", output)

    def test_rag_evaluator_empty_metrics(self):
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        dataset = {"user_input": ["test"], "response": ["test"], "retrieved_contexts": [["test"]]}
        with self.assertRaises(ValueError):
            evaluator.evaluate(metrics=[], dataset=dataset, language="english")


if __name__ == '__main__':
    unittest.main()
