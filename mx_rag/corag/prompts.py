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

from typing import List, Optional


def get_generate_subquery_prompt(query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str) -> str:
    if len(past_subqueries) != len(past_subanswers):
        raise ValueError("past_subqueries and past_subanswers must have the same length")
    
    past = ''
    for idx, (subquery, subanswer) in enumerate(zip(past_subqueries, past_subanswers)):
        past += f"""Step {idx+1}: {subquery}
Answer {idx + 1}: {subanswer}\n"""
    past = past.strip()

    prompt = f"""As an assistant using a search engine to iteratively answer the main question, 
create a new, simple follow-up question based on the previous 
interactions that can help you progress toward a complete answer.

If previous answers aren't helpful, you may rephrase or 
break down the main question. Keep your follow-up question simple and clear.

### Previous Interaction History
{past or 'No previous interactions'}

### Task Overview
{task_desc}

### Main Question
{query}

Please provide only the follow-up question without any additional explanation or text."""

    return prompt


def get_generate_intermediate_answer_prompt(subquery: str, documents: List[str]) -> str:
    context = ''
    for idx, doc in enumerate(documents):
        context += f"""Document {idx+1}:\n{doc}\n\n"""

    prompt = f"""Using only the information provided in the following 
documents, please answer the given question.

IMPORTANT: Do not invent any facts or information 
not explicitly stated in the documents.

If there is no relevant information in the documents 
to answer the question, respond with "No relevant information available".

### Reference Documents
{context.strip()}

### Question
{subquery}

Please provide a direct and concise answer without any additional explanation."""

    return prompt


def get_generate_final_answer_prompt(
        original_query: str, interaction_queries: List[str], interaction_answers: List[str], task_instructions: str,
        reference_docs: Optional[List[str]] = None
) -> str:

    if len(interaction_queries) != len(interaction_answers):
        raise ValueError("interaction_queries and interaction_answers must have the same length")
    
    # Build interaction history
    interaction_history = ''
    for idx, (subquery, subanswer) in enumerate(zip(interaction_queries, interaction_answers)):
        interaction_history += f"""[Subquery {idx+1}] {subquery}
[Response {idx+1}] {subanswer}\n"""
    interaction_history = interaction_history.strip()

    # Build reference context
    reference_context = ''
    if reference_docs:
        for idx, doc in enumerate(reference_docs):
            reference_context += f"""[Reference Document {idx+1}]\n{doc}\n\n"""

    prompt = f"""You are tasked with synthesizing a comprehensive final answer 
for the following question based on the provided interaction history 
and any additional reference documents.

Important: The intermediate responses were generated automatically 
and might contain errors or inconsistencies. Please carefully verify 
all information before including it in your final answer.

---

{'REFERENCE MATERIALS' if reference_docs else 'No Reference Materials'}
{reference_context.strip() or 'No additional reference documents provided.'}

---

INTERACTION HISTORY
{interaction_history or 'No previous interactions available.'}

---

TASK INSTRUCTIONS
{task_instructions}

---

MAIN QUESTION
{original_query}

---

Respond with an appropriate answer only, 
do not explain yourself or output anything else."""
    return prompt


def get_evaluate_answer_prompt(query: str, prediction: str, gt_text: str) -> str:
    prompt = f"""You are an expert evaluator. Determine if the predicted answer correctly answers the question.

    Question: {query}

    Ground truth answer(s): {gt_text}

    Predicted answer: {prediction}

    Evaluate whether the predicted answer is correct. Consider:
    1. Semantic equivalence (same meaning even if wording differs)
    2. Factual correctness
    3. Completeness (if multiple answers are expected)

    Respond with only "YES" if the answer is correct, or "NO" if it is incorrect. Do not provide any explanation."""

    return prompt
