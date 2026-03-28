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

from mx_rag.corag.evaluator import CoRagEvaluator
from mx_rag.corag.config import CoRagBaseConfig
from mx_rag.llm.text2text import Text2TextLLM


class TestCoRagEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up mock dependencies for testing."""
        self.mock_base_llm = MagicMock(spec=Text2TextLLM)
        self.mock_config = MagicMock(spec=CoRagBaseConfig)
        self.mock_config.base_llm = self.mock_base_llm
        self.mock_config.final_llm = None
        self.mock_config.judge_llm = None
        self.mock_config.sub_answer_llm = None
        self.mock_config.retrieve_api_url = "http://example.com/retrieve"
        self.mock_config.num_threads = 4
        self.mock_config.max_path_length = 3
        
        self.evaluator = CoRagEvaluator(self.mock_config)

    def test_init(self):
        """Test CoRagEvaluator initialization."""
        self.assertEqual(self.evaluator.base_llm, self.mock_base_llm)
        self.assertEqual(self.evaluator.final_llm, None)
        self.assertEqual(self.evaluator.judge_llm, None)
        self.assertEqual(self.evaluator.sub_answer_llm, None)
        self.assertEqual(self.evaluator.retrieve_api_url, "http://example.com/retrieve")
        self.assertEqual(self.evaluator.num_threads, 4)
        self.assertEqual(self.evaluator.max_path_length, 3)

    def test_unicode_escape_to_char(self):
        """Test unicode escape to character conversion."""
        self.assertEqual(
            CoRagEvaluator._unicode_escape_to_char('\\u4e2d\\u6587'),
            '中文')
        self.assertEqual(
            CoRagEvaluator._unicode_escape_to_char('Hello\\u0020World'),
            'Hello World')
        self.assertEqual(
            CoRagEvaluator._unicode_escape_to_char('Normal text'),
            'Normal text')

    def test_split_sentences(self):
        """Test sentence splitting functionality."""
        text = "Hello world. How are you? I'm fine!"
        sentences = CoRagEvaluator._split_sentences(text)
        self.assertEqual(sentences, ["Hello world.", "How are you?", "I'm fine!"])
        
        # Test with empty text
        self.assertEqual(CoRagEvaluator._split_sentences(""), [])
        
        # Test with only whitespace
        self.assertEqual(CoRagEvaluator._split_sentences("   \t   \n   "), [])

    def test_extract_question(self):
        """Test question extraction from data items."""
        # Test with 'question' field
        item = {"question": "What is the capital of France?"}
        self.assertEqual(
            CoRagEvaluator._extract_question(item),
            "What is the capital of France?"
        )
        
        # Test with other question field names
        item = {"user_query": "What is the capital of Germany?"}
        self.assertEqual(
            CoRagEvaluator._extract_question(item),
            ""
        )
        # Test with empty item
        self.assertEqual(CoRagEvaluator._extract_question({}), "")
        # Test with no question field
        self.assertEqual(CoRagEvaluator._extract_question({"answer": "Paris"}), "")

    def test_check_hit(self):
        """Test hit checking functionality."""
        retrieved_docs = [
            "Paris is the capital city of France.",
            "France is a country in Europe."
        ]
        golden_facts = ["Paris is the capital", "France in Europe"]
        
        hits, total = self.evaluator._check_hit(retrieved_docs, golden_facts)
        self.assertEqual(hits, 1)
        self.assertEqual(total, 2)
        
        # Test with no hits
        retrieved_docs = ["London is a city in England."]
        hits, total = self.evaluator._check_hit(retrieved_docs, golden_facts)
        self.assertEqual(hits, 0)
        self.assertEqual(total, 2)
        
        # Test with empty inputs
        hits, total = self.evaluator._check_hit([], golden_facts)
        self.assertEqual(hits, 0)
        self.assertEqual(total, 2)
        
        hits, total = self.evaluator._check_hit(retrieved_docs, [])
        self.assertEqual(hits, 0)
        self.assertEqual(total, 0)

    def test_get_golden_facts_musique_format(self):
        """Test golden facts extraction for MuSiQue format."""
        item = {
            "paragraphs": [
                {
                    "is_supporting": True,
                    "paragraph_text": "Paris is the capital of France. It is a beautiful city."
                },
                {
                    "is_supporting": False,
                    "paragraph_text": "London is the capital of England."
                }
            ]
        }
        
        facts = self.evaluator._get_golden_facts(item)
        self.assertEqual(
            facts,
            ["Paris is the capital of France.", "It is a beautiful city."]
        )

    def test_get_golden_facts_hotpotqa_format(self):
        """Test golden facts extraction for HotpotQA format."""
        item = {
            "context": [
                ["France", ["Paris is the capital of France.", "France is in Europe."]],
                ["Germany", ["Berlin is the capital of Germany."]]
            ],
            "supporting_facts": [
                ["France", 0],
                ["Germany", 0]
            ]
        }
        
        facts = self.evaluator._get_golden_facts(item)
        self.assertEqual(
            facts,
            ["Paris is the capital of France.", "Berlin is the capital of Germany."]
        )

    @patch('mx_rag.corag.evaluator.search_by_retrieve_api')
    def test_naive_retrieve(self, mock_search):
        """Test naive retrieval functionality."""
        # Mock search results
        mock_search.return_value = [
            {"text": "Paris is the capital of France."},
            {"content": "France is a country in Europe."},
            {"contents": "Europe is a continent."},
            "Direct string result"
        ]
        
        results = self.evaluator._naive_retrieve("What is the capital of France?", 3)
        self.assertEqual(len(results), 3)
        self.assertIn("Paris is the capital of France.", results)
        self.assertIn("France is a country in Europe.", results)
        self.assertIn("Europe is a continent.", results)
        
        # Test exception handling
        mock_search.side_effect = Exception("Search error")
        results = self.evaluator._naive_retrieve("What is the capital of France?", 3)
        self.assertEqual(results, [])

    @patch('mx_rag.corag.evaluator.get_generate_intermediate_answer_prompt')
    def test_generate_naive_prediction(self, mock_get_prompt):
        """Test naive prediction generation."""
        mock_get_prompt.return_value = "test prompt"
        self.mock_base_llm.chat.return_value = "Paris"
        
        prediction = self.evaluator._generate_naive_prediction(
            "What is the capital of France?",
            ["Paris is the capital of France."]
        )
        
        self.assertEqual(prediction, "Paris")
        mock_get_prompt.assert_called_once()
        self.mock_base_llm.chat.assert_called_once_with("test prompt")

    def test_aggregate_metrics(self):
        """Test metrics aggregation functionality."""
        results = [
            {
                "time": [1.0, 0.5],
                "is_correct": True,
                "naive_is_correct": False,
                "corag_recall": {"hits": 2, "total": 3},
                "naive_recall": {"hits": 1, "total": 3}
            },
            {
                "time": [0.8, 0.4],
                "is_correct": False,
                "naive_is_correct": True,
                "corag_recall": {"hits": 1, "total": 2},
                "naive_recall": {"hits": 1, "total": 2}
            }
        ]
        
        summary = CoRagEvaluator._aggregate_metrics(results, True, True)
        
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["corag_accuracy"], 0.5)
        self.assertEqual(summary["naive_accuracy"], 0.5)
        self.assertEqual(summary["corag_correct_count"], 1)
        self.assertEqual(summary["naive_correct_count"], 1)
        self.assertEqual(summary["avg_path_time"], 0.9)
        self.assertEqual(summary["avg_time"], 0.45)
        self.assertEqual(summary["corag_micro_recall"], 3/5)
        self.assertEqual(summary["naive_micro_recall"], 2/5)


if __name__ == '__main__':
    unittest.main()