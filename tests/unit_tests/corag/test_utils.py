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

from mx_rag.corag.utils import (
    normalize_retrieve_api_results,
    normalize_text,
    check_answer,
    check_answer_with_llm_judge,
    ThreadSafeCounter,
    truncate_long_text_by_char
)
from mx_rag.llm.text2text import Text2TextLLM


class TestUtils(unittest.TestCase):
    def test_normalize_retrieve_api_results(self):
        """Test graph API results normalization."""
        # Test with dict containing 'chunks' key
        self.assertEqual(
            normalize_retrieve_api_results({'chunks': [1, 2, 3]}),
            [1, 2, 3]
        )
        
        # Test with dict containing other keys
        self.assertEqual(
            normalize_retrieve_api_results({'data': ['a', 'b']}),
            ['a', 'b']
        )
        
        # Test with empty dict
        self.assertEqual(normalize_retrieve_api_results({}), [])
        
        # Test with list
        self.assertEqual(
            normalize_retrieve_api_results([{'key': 'value'}]),
            [{'key': 'value'}]
        )
        
        # Test with non-dict, non-list
        self.assertEqual(normalize_retrieve_api_results("string"), [])

    def test_normalize_text(self):
        """Test text normalization function."""
        self.assertEqual(
            normalize_text("Hello, World!"),
            "hello world"
        )
        
        self.assertEqual(
            normalize_text("The quick brown fox jumps over the lazy dog."),
            "quick brown fox jumps over lazy dog"
        )
        
        self.assertEqual(
            normalize_text("  Multiple   spaces   here  "),
            "multiple spaces here"
        )
        
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(""), "")

    def test_check_answer(self):
        """Test answer checking function."""
        self.assertTrue(
            check_answer("The capital of France is Paris.", ["Paris"])
        )
        
        self.assertFalse(
            check_answer("London", ["Paris", "Berlin"])
        )
        
        self.assertFalse(
            check_answer("", ["Paris"])
        )
        
        self.assertFalse(
            check_answer("Paris", [""])
        )

    @patch('mx_rag.corag.utils.get_evaluate_answer_prompt')
    def test_check_answer_with_llm_judge(self, mock_get_prompt):
        """Test LLM-based answer checking function."""
        # Mock LLM and prompt
        mock_llm = MagicMock(spec=Text2TextLLM)
        mock_get_prompt.return_value = "test prompt"
        
        # Test with YES response
        mock_llm.chat.return_value = "YES"
        self.assertTrue(
            check_answer_with_llm_judge(
                "Paris", ["Paris"], "What is the capital of France?", mock_llm
            )
        )
        
        # Test with NO response
        mock_llm.chat.return_value = "NO"
        self.assertFalse(
            check_answer_with_llm_judge(
                "London", ["Paris"], "What is the capital of France?", mock_llm
            )
        )
        
        # Test with exception fallback
        mock_llm.chat.side_effect = Exception("LLM error")
        self.assertTrue(
            check_answer_with_llm_judge(
                "Paris", ["Paris"], "What is the capital of France?", mock_llm
            )
        )
        
        # Test with empty prediction
        self.assertFalse(
            check_answer_with_llm_judge(
                "", ["Paris"], "What is the capital of France?", mock_llm
            )
        )

    def test_thread_safe_counter(self):
        """Test thread-safe counter class."""
        counter = ThreadSafeCounter(initial_value=5)

        # Test increment
        self.assertEqual(counter.increment(), 6)
        self.assertEqual(counter.increment(3), 9)

        # Test reset
        self.assertEqual(counter.reset(), 0)
        self.assertEqual(counter.increment(), 1)

    def test_truncate_long_text_by_char(self):
        """Test text truncation function."""
        # Test with short text (no truncation needed)
        short_text = "Short text"
        self.assertEqual(truncate_long_text_by_char(short_text, 100), short_text)

        # Test with long English text
        long_text = "a" * 200
        result = truncate_long_text_by_char(long_text, 50)
        self.assertEqual(len(result), 100)  # English text * 2

        # Test with long Chinese text
        long_chinese = "中" * 200
        result = truncate_long_text_by_char(long_chinese, 50)
        self.assertEqual(len(result), 50)  # Chinese text * 1

        # Test with empty text
        self.assertEqual(truncate_long_text_by_char("", 50), "")

        # Test with None
        self.assertEqual(truncate_long_text_by_char(None, 50), None)


if __name__ == '__main__':
    unittest.main()