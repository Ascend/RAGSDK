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

from mx_rag.corag.prompts import (
    get_generate_subquery_prompt,
    get_generate_intermediate_answer_prompt,
    get_generate_final_answer_prompt,
    get_evaluate_answer_prompt
)


class TestPrompts(unittest.TestCase):
    def test_get_generate_subquery_prompt(self):
        """Test subquery generation prompt function."""
        # Test with empty past interactions
        prompt = get_generate_subquery_prompt(
            query="What is the capital of France?",
            past_subqueries=[],
            past_subanswers=[],
            task_desc="Answer geography questions"
        )
        
        self.assertIn("main question", prompt)
        self.assertIn("No previous interactions", prompt)
        self.assertIn("Answer geography questions", prompt)
        self.assertIn("What is the capital of France?", prompt)
        
        # Test with past interactions
        prompt = get_generate_subquery_prompt(
            query="What is the capital of France?",
            past_subqueries=["Which country is Paris in?"],
            past_subanswers=["France"],
            task_desc="Answer geography questions"
        )
        
        self.assertIn("Step 1: Which country is Paris in?", prompt)
        self.assertIn("Answer 1: France", prompt)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            get_generate_subquery_prompt(
                query="What is the capital of France?",
                past_subqueries=["Which country is Paris in?", "What is France's capital?"],
                past_subanswers=["France"],
                task_desc="Answer geography questions"
            )

    def test_get_generate_intermediate_answer_prompt(self):
        """Test intermediate answer generation prompt function."""
        # Test with documents
        prompt = get_generate_intermediate_answer_prompt(
            subquery="What is the capital of France?",
            documents=["Paris is the capital city of France.", "France is a country in Europe."]
        )
        
        self.assertIn("Using only the information provided in the following", prompt)
        self.assertIn("Document 1:", prompt)
        self.assertIn("Paris is the capital city of France.", prompt)
        self.assertIn("Document 2:", prompt)
        self.assertIn("France is a country in Europe.", prompt)
        self.assertIn("What is the capital of France?", prompt)
        
        # Test with empty documents
        prompt = get_generate_intermediate_answer_prompt(
            subquery="What is the capital of France?",
            documents=[]
        )
        
        self.assertIn("Reference Documents", prompt)
        # Should handle empty documents gracefully
        self.assertIn("What is the capital of France?", prompt)

    def test_get_generate_final_answer_prompt(self):
        """Test final answer generation prompt function."""
        # Test with documents and past interactions
        prompt = get_generate_final_answer_prompt(
            original_query="What is the capital of France?",
            interaction_queries=["Which country is Paris in?"],
            interaction_answers=["France"],
            task_instructions="Answer geography questions",
            reference_docs=["Paris is the capital city of France."]
        )
        
        self.assertIn("comprehensive final answer", prompt)
        self.assertIn("REFERENCE MATERIALS", prompt)
        self.assertIn("Reference Document 1", prompt)
        self.assertIn("Paris is the capital city of France.", prompt)
        self.assertIn("INTERACTION HISTORY", prompt)
        self.assertIn("[Subquery 1] Which country is Paris in?", prompt)
        self.assertIn("[Response 1] France", prompt)
        self.assertIn("Answer geography questions", prompt)
        self.assertIn("What is the capital of France?", prompt)
        
        # Test without documents
        prompt = get_generate_final_answer_prompt(
            original_query="What is the capital of France?",
            interaction_queries=["Which country is Paris in?"],
            interaction_answers=["France"],
            task_instructions="Answer geography questions",
            reference_docs=None
        )
        
        self.assertIn("No Reference Materials", prompt)
        
        # Test without past interactions
        prompt = get_generate_final_answer_prompt(
            original_query="What is the capital of France?",
            interaction_queries=[],
            interaction_answers=[],
            task_instructions="Answer geography questions"
        )
        
        self.assertIn("No previous interactions", prompt)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            get_generate_final_answer_prompt(
                original_query="What is the capital of France?",
                interaction_queries=["Which country is Paris in?", "What is France's capital?"],
                interaction_answers=["France"],
                task_instructions="Answer geography questions"
            )

    def test_get_evaluate_answer_prompt(self):
        """Test answer evaluation prompt function."""
        prompt = get_evaluate_answer_prompt(
            query="What is the capital of France?",
            prediction="Paris",
            gt_text="Paris"
        )
        
        self.assertIn("expert evaluator", prompt)
        self.assertIn("Question: What is the capital of France?", prompt)
        self.assertIn("Ground truth answer(s): Paris", prompt)
        self.assertIn("Predicted answer: Paris", prompt)
        self.assertIn("Respond with only \"YES\"", prompt)
        
        # Test with multiple ground truths
        prompt = get_evaluate_answer_prompt(
            query="What is the capital of France?",
            prediction="Paris",
            gt_text="Paris or France's capital"
        )
        
        self.assertIn("Ground truth answer(s): Paris or France's capital", prompt)


if __name__ == '__main__':
    unittest.main()