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
from unittest.mock import MagicMock

from mx_rag.corag.config import CoRagBaseConfig, DEFAULT_TASK_DESC, DEFAULT_SAMPLE_TASK_DESC
from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.utils import ClientParam


class TestConfig(unittest.TestCase):
    def test_default_constants(self):
        """Test default constants are properly defined."""
        self.assertIsInstance(DEFAULT_TASK_DESC, str)
        self.assertIn("search tools", DEFAULT_TASK_DESC)
        self.assertIn("sub-queries", DEFAULT_TASK_DESC)
        self.assertEqual(DEFAULT_SAMPLE_TASK_DESC, "answer multi-hop questions")

    def test_corag_base_config_init(self):
        """Test CoRagBaseConfig initialization."""
        # Create mock LLM instances
        mock_base_llm = MagicMock(spec=Text2TextLLM)
        mock_final_llm = MagicMock(spec=Text2TextLLM)
        mock_sub_answer_llm = MagicMock(spec=Text2TextLLM)
        mock_judge_llm = MagicMock(spec=Text2TextLLM)
        
        # Test initialization with default parameters
        config = CoRagBaseConfig(
            base_llm=mock_base_llm,
            retrieve_api_url="http://example.com/retrieve"
        )

        self.assertEqual(config.base_llm, mock_base_llm)
        self.assertEqual(config.retrieve_api_url, "http://example.com/retrieve")
        self.assertEqual(config.num_threads, 8)  # Default value
        self.assertEqual(config.max_path_length, 3)  # Default value
        self.assertIsInstance(config.client_param, ClientParam)

        # Test initialization with custom parameters
        mock_client_param = ClientParam(use_http=True)
        config = CoRagBaseConfig(
            base_llm=mock_base_llm,
            retrieve_api_url="http://example.com/retrieve",
            num_threads=16,
            max_path_length=5,
            final_llm=mock_final_llm,
            sub_answer_llm=mock_sub_answer_llm,
            judge_llm=mock_judge_llm,
            client_param=mock_client_param
        )
        self.assertEqual(config.base_llm, mock_base_llm)
        self.assertEqual(config.retrieve_api_url, "http://example.com/retrieve")
        self.assertEqual(config.num_threads, 16)
        self.assertEqual(config.max_path_length, 5)
        self.assertEqual(config.final_llm, mock_final_llm)
        self.assertEqual(config.sub_answer_llm, mock_sub_answer_llm)
        self.assertEqual(config.judge_llm, mock_judge_llm)
        self.assertEqual(config.client_param, mock_client_param)


if __name__ == '__main__':
    unittest.main()