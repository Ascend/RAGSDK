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

from mx_rag.corag.corag_agent import (
    _process_subquery,
    ReasoningPath,
    CoRagAgent
)
from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.utils import ClientParam


class TestCoRagAgent(unittest.TestCase):
    def test_process_subquery(self):
        """Test subquery processing function."""
        # Test with reasoning block
        subquery, reasoning = _process_subquery(
            '<reasoning>Let me think about this...</reasoning> What is the capital of France?'
        )
        self.assertEqual(subquery, "What is the capital of France?")
        self.assertEqual(reasoning, "Let me think about this...")
        
        # Test with old think block format
        subquery, reasoning = _process_subquery(
            '<think>Let me think about this...</think> What is the capital of France?'
        )
        self.assertEqual(subquery, "What is the capital of France?")
        self.assertIsNone(reasoning)  # Old format not captured as reasoning
        
        # Test with step prefix
        subquery, reasoning = _process_subquery(
            'Step 1: What is the capital of France?'
        )
        self.assertEqual(subquery, "What is the capital of France?")
        self.assertIsNone(reasoning)
        
        # Test with intermediate query prefix
        subquery, reasoning = _process_subquery(
            'Intermediate query 1: What is the capital of France?'
        )
        self.assertEqual(subquery, "What is the capital of France?")
        self.assertIsNone(reasoning)
        
        # Test with surrounding quotes
        subquery, reasoning = _process_subquery(
            '"What is the capital of France?"'
        )
        self.assertEqual(subquery, "What is the capital of France?")
        self.assertIsNone(reasoning)

    def test_rag_path_dataclass(self):
        """Test ReasoningPath data class initialization and structure."""
        # Test default initialization
        rag_path = ReasoningPath(original_query="What is the capital of France?")
        self.assertEqual(rag_path.original_query, "What is the capital of France?")
        self.assertEqual(rag_path.subqueries, [])
        self.assertEqual(rag_path.subanswers, [])
        self.assertEqual(rag_path.document_ids, [])
        self.assertEqual(rag_path.reasoning_steps, [])
        self.assertEqual(rag_path.documents, [])
        
        # Test with custom values
        rag_path = ReasoningPath(
            original_query="What is the capital of France?",
            subqueries=["Which country is Paris in?"],
            subanswers=["France"],
            document_ids=[["doc1"]],
            reasoning_steps=["Let me think..."],
            documents=[["Paris is the capital of France."]]
        )
        self.assertEqual(rag_path.original_query, "What is the capital of France?")
        self.assertEqual(rag_path.subqueries, ["Which country is Paris in?"])
        self.assertEqual(rag_path.subanswers, ["France"])
        self.assertEqual(rag_path.document_ids, [["doc1"]])
        self.assertEqual(rag_path.reasoning_steps, ["Let me think..."])
        self.assertEqual(rag_path.documents, [["Paris is the capital of France."]])

    @patch('mx_rag.corag.corag_agent.RequestUtils')
    def test_corag_agent_init(self, mock_request_utils_class):
        """Test CoRagAgent initialization."""
        # Create mock LLMs
        mock_base_llm = MagicMock(spec=Text2TextLLM)
        mock_final_llm = MagicMock(spec=Text2TextLLM)
        mock_sub_answer_llm = MagicMock(spec=Text2TextLLM)

        # Test with only base LLM
        agent = CoRagAgent(
            base_llm=mock_base_llm,
            retrieve_api_url="http://example.com/retrieve"
        )

        self.assertEqual(agent.base_llm, mock_base_llm)
        self.assertEqual(agent.retrieve_api_url, "http://example.com/retrieve")
        self.assertIsNone(agent.final_llm)
        self.assertIsNone(agent.sub_answer_llm)
        self.assertIsInstance(agent.client_param, ClientParam)
        self.assertIsNotNone(agent._client)

        # Test with all LLMs and custom client_param
        mock_client_param = ClientParam(use_http=True)
        agent = CoRagAgent(
            base_llm=mock_base_llm,
            retrieve_api_url="http://example.com/retrieve",
            final_llm=mock_final_llm,
            sub_answer_llm=mock_sub_answer_llm,
            client_param=mock_client_param
        )

        self.assertEqual(agent.base_llm, mock_base_llm)
        self.assertEqual(agent.retrieve_api_url, "http://example.com/retrieve")
        self.assertEqual(agent.final_llm, mock_final_llm)
        self.assertEqual(agent.sub_answer_llm, mock_sub_answer_llm)
        self.assertEqual(agent.client_param, mock_client_param)

    @patch('mx_rag.corag.corag_agent.RequestUtils')
    @patch('mx_rag.corag.corag_agent.get_generate_subquery_prompt')
    @patch('mx_rag.corag.corag_agent.get_generate_intermediate_answer_prompt')
    def test_sample_path(self, mock_get_intermediate_prompt, mock_get_subquery_prompt, mock_request_utils_class):
        """Test sample_path method with mocks."""
        # Setup mocks
        mock_base_llm = MagicMock(spec=Text2TextLLM)

        # Mock LLM responses
        mock_base_llm.chat.side_effect = [
            "What is the capital of France?",  # Subquery
            "Paris",  # Intermediate answer
        ]
        mock_base_llm.llm_config = LLMParameterConfig()

        # Mock RequestUtils and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = '[{"text": "Paris is the capital city of France.", "id": "doc1"}]'
        mock_client.post.return_value = mock_response
        mock_request_utils_class.return_value = mock_client

        # Mock prompts (just return simple strings for testing)
        mock_get_subquery_prompt.return_value = "subquery prompt"
        mock_get_intermediate_prompt.return_value = "intermediate prompt"

        # Create agent and call sample_path
        agent = CoRagAgent(
            base_llm=mock_base_llm,
            retrieve_api_url="http://example.com/retrieve"
        )

        rag_path = agent.sample_path(
            query="What is the capital of France?",
            task_desc="Answer geography questions",
            max_path_length=1
        )

        # Verify interactions
        self.assertEqual(len(rag_path.subqueries), 1)
        self.assertEqual(len(rag_path.subanswers), 1)
        self.assertEqual(len(rag_path.document_ids), 1)
        self.assertEqual(len(rag_path.documents), 1)

        # Check that LLM was called
        self.assertEqual(mock_base_llm.chat.call_count, 2)

        # Check that RequestUtils post was called
        mock_client.post.assert_called_once()

        # Check that prompts were generated
        mock_get_subquery_prompt.assert_called_once()
        mock_get_intermediate_prompt.assert_called_once()

if __name__ == '__main__':
    unittest.main()