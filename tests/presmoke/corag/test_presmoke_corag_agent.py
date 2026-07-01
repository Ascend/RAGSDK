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

from mx_rag.corag import CoRagAgent
from mx_rag.llm import Text2TextLLM
from mx_rag.llm.llm_parameter import LLMParameterConfig


class TestCoRagAgent(unittest.TestCase):
    @patch('mx_rag.corag.corag_agent.RequestUtils')
    @patch('mx_rag.corag.corag_agent.get_generate_subquery_prompt')
    @patch('mx_rag.corag.corag_agent.get_generate_intermediate_answer_prompt')
    def test_corag_agent_sample_path(
        self, mock_get_intermediate_prompt, mock_get_subquery_prompt, mock_request_utils_class
    ):
        mock_base_llm = MagicMock(spec=Text2TextLLM)
        mock_base_llm.chat.side_effect = [
            "What is the capital of France?",
            "Paris",
        ]
        mock_base_llm.llm_config = LLMParameterConfig()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = '[{"text": "Paris is the capital city of France.", "id": "doc1"}]'
        mock_client.post.return_value = mock_response
        mock_request_utils_class.return_value = mock_client

        mock_get_subquery_prompt.return_value = "subquery prompt"
        mock_get_intermediate_prompt.return_value = "intermediate prompt"

        agent = CoRagAgent(base_llm=mock_base_llm, retrieve_api_url="http://127.0.0.1:8000/retrieve")

        rag_path = agent.sample_path(
            query="What is the capital of France?", task_desc="Answer geography questions", max_path_length=1
        )

        self.assertEqual(len(rag_path.subqueries), 1)
        self.assertEqual(len(rag_path.subanswers), 1)

    @patch('mx_rag.corag.corag_agent.RequestUtils')
    def test_corag_agent_init(self, mock_request_utils_class):
        mock_base_llm = MagicMock(spec=Text2TextLLM)
        agent = CoRagAgent(base_llm=mock_base_llm, retrieve_api_url="http://127.0.0.1:8000/retrieve")
        self.assertEqual(agent.base_llm, mock_base_llm)
        self.assertEqual(agent.retrieve_api_url, "http://127.0.0.1:8000/retrieve")


if __name__ == '__main__':
    unittest.main()
