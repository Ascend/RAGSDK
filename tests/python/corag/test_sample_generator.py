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

from mx_rag.corag.sample_generator import SampleGenerator
from mx_rag.corag.config import CoRagBaseConfig
from mx_rag.corag.corag_agent import ReasoningPath
from mx_rag.llm.text2text import Text2TextLLM


class TestSampleGenerator(unittest.TestCase):
    def setUp(self):
        """Set up mock dependencies for testing."""
        self.mock_base_llm = MagicMock(spec=Text2TextLLM)
        self.mock_config = MagicMock(spec=CoRagBaseConfig)
        self.mock_config.base_llm = self.mock_base_llm
        self.mock_config.final_llm = None
        self.mock_config.sub_answer_llm = None
        self.mock_config.judge_llm = None
        self.mock_config.retrieve_api_url = "http://example.com/retrieve"
        self.mock_config.num_threads = 4
        self.mock_config.max_path_length = 3
        
        self.generator = SampleGenerator(self.mock_config)

    def test_init(self):
        """Test SampleGenerator initialization."""
        self.assertEqual(self.generator.base_llm, self.mock_base_llm)
        self.assertEqual(self.generator.final_llm, None)
        self.assertEqual(self.generator.sub_answer_llm, None)
        self.assertEqual(self.generator.judge_llm, None)
        self.assertEqual(self.generator.retrieve_api_url, "http://example.com/retrieve")
        self.assertEqual(self.generator.num_threads, 4)
        self.assertEqual(self.generator.max_path_length, 3)

    @patch('mx_rag.corag.sample_generator.get_generate_subquery_prompt')
    @patch('mx_rag.corag.sample_generator.get_generate_intermediate_answer_prompt')
    @patch('mx_rag.corag.sample_generator.get_generate_final_answer_prompt')
    def test_generate_samples(self, mock_final_prompt, mock_intermediate_prompt, mock_subquery_prompt):
        """Test sample generation from valid paths."""
        # Mock prompts
        mock_subquery_prompt.return_value = "subquery prompt"
        mock_intermediate_prompt.return_value = "intermediate prompt"
        mock_final_prompt.return_value = "final prompt"
        
        # Test valid path
        path = {
            "query": "What is the capital of France?",
            "steps": [
                {
                    "subquery": "Which city is the capital of France?",
                    "subanswer": "Paris",
                    "documents": ["Paris is the capital of France."]
                }
            ],
            "generated_final_answer": "The capital of France is Paris."
        }
        
        samples = SampleGenerator._generate_samples(path)
        
        # Check number of samples (1 subquery + 1 subanswer + 1 final answer)
        self.assertEqual(len(samples), 3)
        
        # Check subquery generation sample
        self.assertEqual(samples[0]["type"], "subquery_generation")
        self.assertEqual(len(samples[0]["messages"]), 2)
        self.assertEqual(samples[0]["messages"][0]["role"], "user")
        self.assertEqual(samples[0]["messages"][1]["role"], "assistant")
        
        # Check subanswer generation sample
        self.assertEqual(samples[1]["type"], "subanswer_generation")
        self.assertEqual(len(samples[1]["messages"]), 2)
        
        # Check final answer generation sample
        self.assertEqual(samples[2]["type"], "final_answer_generation")
        self.assertEqual(len(samples[2]["messages"]), 2)
        
        # Test with empty path
        self.assertEqual(SampleGenerator._generate_samples({}), [])
        
        # Test with empty steps
        path_empty_steps = {
            "query": "What is the capital of France?",
            "steps": [],
            "generated_final_answer": "The capital of France is Paris."
        }
        self.assertEqual(SampleGenerator._generate_samples(path_empty_steps), [])

    @patch('mx_rag.corag.sample_generator.check_answer')
    def test_generate_paths(self, mock_check_answer):
        """Test path generation functionality."""
        # Mock CoRagAgent and its methods
        mock_agent = MagicMock()
        mock_rag_path = MagicMock(spec=ReasoningPath)
        mock_rag_path.subqueries = ["Which city is the capital of France?"]
        mock_rag_path.subanswers = ["Paris"]
        mock_rag_path.documents = [["Paris is the capital of France."]]
        mock_agent.sample_path.return_value = mock_rag_path
        mock_agent.generate_final_answer.return_value = "Paris"
        
        # Mock check_answer to return True
        mock_check_answer.return_value = True
        
        # Test item with query and answer
        item = {
            "id": "123",
            "query": "What is the capital of France?",
            "answer": ["Paris"]
        }
        
        paths = self.generator._generate_paths(item, mock_agent, n_samples=1)
        
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["id"], "123")
        self.assertEqual(paths[0]["query"], "What is the capital of France?")
        self.assertEqual(paths[0]["answers"], ["Paris"])
        self.assertEqual(paths[0]["generated_final_answer"], "Paris")
        self.assertEqual(len(paths[0]["steps"]), 1)
        
        # Test with no query
        item_no_query = {"answer": ["Paris"]}
        paths = self.generator._generate_paths(item_no_query, mock_agent, n_samples=1)
        self.assertEqual(paths, [])
        
        # Test with incorrect answer
        mock_check_answer.return_value = False
        paths = self.generator._generate_paths(item, mock_agent, n_samples=1)
        self.assertEqual(paths, [])

    @patch('mx_rag.corag.sample_generator.SampleGenerator._generate_paths')
    @patch('mx_rag.corag.sample_generator.SampleGenerator._generate_samples')
    @patch('mx_rag.corag.sample_generator.SampleGenerator._create_agent')
    def test_process_item(self, mock_create_agent, mock_generate_samples, mock_generate_paths):
        """Test item processing functionality."""
        # Mock dependencies
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        
        mock_path = {
            "query": "What is the capital of France?",
            "steps": [],
            "generated_final_answer": "Paris"
        }
        mock_generate_paths.return_value = [mock_path]
        
        mock_samples = [{"type": "test_sample"}]
        mock_generate_samples.return_value = mock_samples
        
        # Test item processing
        item = {"query": "What is the capital of France?", "answer": ["Paris"]}
        samples = self.generator._process_item(item, n_samples=1)
        
        self.assertEqual(samples, mock_samples)
        mock_create_agent.assert_called_once()
        mock_generate_paths.assert_called_once_with(item, mock_agent, 1)
        mock_generate_samples.assert_called_once_with(mock_path)
        
        # Test with no valid paths
        mock_generate_paths.return_value = []
        samples = self.generator._process_item(item, n_samples=1)
        self.assertEqual(samples, [])
        mock_generate_samples.assert_called_once_with(mock_path)  

if __name__ == '__main__':
    unittest.main()