# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import Mock, patch
from mx_rag.graphrag.qa_base_model import QABaseModel, GenerationEvaluationStrategy
from mx_rag.utils.common import Lang


class TestQABaseModel(unittest.TestCase):
    def setUp(self):
        """Set up a mock LLM and QABaseModel instance."""
        self.mock_llm = Mock()
        self.mock_llm_config = Mock()
        self.qa_model = QABaseModel(self.mock_llm, self.mock_llm_config, metric="generation")

    def test_init(self):
        """Test initialization of QABaseModel."""
        self.assertEqual(self.qa_model.llm, self.mock_llm)
        self.assertEqual(self.qa_model.llm_config, self.mock_llm_config)
        self.assertEqual(self.qa_model.metric, "generation")
        self.assertIsInstance(self.qa_model.evaluation_strategy, GenerationEvaluationStrategy)

    @patch.object(QABaseModel, "_plain_generate")
    def test_generate(self, mock_plain_generate):
        """Test the generate method."""
        mock_plain_generate.return_value = ["response1", "response2"]
        questions = ["question1", "question2"]

        result = self.qa_model.generate(questions)

        mock_plain_generate.assert_called_once_with(questions)
        self.assertEqual(result, ["response1", "response2"])

    @patch.object(GenerationEvaluationStrategy, "evaluate")
    def test_evaluate(self, mock_evaluate):
        """Test the evaluate method."""
        mock_evaluate.return_value = ["evaluation1", "evaluation2"]
        questions = ["question1", "question2"]
        answers = ["answer1", "answer2"]
        responses = ["response1", "response2"]

        result = self.qa_model.evaluate(questions, answers, responses, Lang.EN)

        mock_evaluate.assert_called_once_with(questions, answers, responses, Lang.EN)
        self.assertEqual(result, ["evaluation1", "evaluation2"])

    def test_select_evaluation_strategy(self):
        """Test the _select_evaluation_strategy method."""
        strategy = self.qa_model._select_evaluation_strategy()
        self.assertIsInstance(strategy, GenerationEvaluationStrategy)

        # Test unsupported metric
        self.qa_model.metric = "unsupported_metric"
        strategy = self.qa_model._select_evaluation_strategy()
        self.assertIsNone(strategy)

    @patch("mx_rag.graphrag.qa_base_model.LLM_PLAIN_TEMPLATE", "{question}")
    def test_plain_generate(self):
        """Test the _plain_generate method."""
        self.mock_llm.chat.side_effect = ["response1", "response2"]
        questions = ["question1", "question2"]

        result = self.qa_model._plain_generate(questions)

        expected_prompts = ["question1", "question2"]
        self.mock_llm.chat.assert_any_call(
            expected_prompts[0],
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions as simply as possible."
                }
            ],
            llm_config=self.mock_llm_config
        )
        self.mock_llm.chat.assert_any_call(
            expected_prompts[1],
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions as simply as possible."
                }
            ],
            llm_config=self.mock_llm_config
        )
        self.assertEqual(result, ["response1", "response2"])
