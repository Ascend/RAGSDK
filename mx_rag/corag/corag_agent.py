#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

import re
import threading
import json
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from loguru import logger

from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.corag.utils import normalize_retrieve_api_results, truncate_long_text_by_char
from mx_rag.utils.url import RequestUtils
from mx_rag.utils import ClientParam
from mx_rag.corag.prompts import (
    get_generate_subquery_prompt,
    get_generate_intermediate_answer_prompt,
    get_generate_final_answer_prompt,
)


def _process_subquery(input_subquery: str) -> Tuple[str, Optional[str]]:
    # 限制输入长度，防止ReDoS攻击
    if len(input_subquery) > 10000:
        return input_subquery.strip(), None

    # Extract reasoning blocks from the subquery
    # 使用更严格的模式，避免过度回溯
    reasoning_pattern = r'<reasoning>([^<]*)</reasoning>'
    reasoning_match = re.search(reasoning_pattern, input_subquery)
    reasoning_content = reasoning_match.group(1).strip() if reasoning_match else None

    # Remove reasoning blocks from the original text
    processed_subquery = re.sub(reasoning_pattern, '', input_subquery)
    # Also handle original think block format for backward compatibility
    processed_subquery = re.sub(r'<think>([^<]*)</think>', '', processed_subquery)

    processed_subquery = processed_subquery.strip()
    # Remove surrounding quotes if present
    if processed_subquery.startswith('"') and processed_subquery.endswith('"'):
        processed_subquery = processed_subquery[1:-1]
    # Remove step prefix if present (both old and new formats)
    processed_subquery = re.sub(r'^(Step|Intermediate query) \d+: ', '', processed_subquery)

    return processed_subquery, reasoning_content


@dataclass
class ReasoningPath:
    """
    表示CoRAG推理路径的类，包含原始查询、子查询、子答案、文档ID、思考过程和文档列表。
    """

    original_query: str
    subqueries: List[str] = field(default_factory=list)
    subanswers: List[str] = field(default_factory=list)
    document_ids: List[List[str]] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    documents: List[List[str]] = field(default_factory=list)


class CoRagAgent:
    def __init__(
        self,
        base_llm: Text2TextLLM,
        retrieve_api_url: str,
        final_llm: Optional[Text2TextLLM] = None,
        sub_answer_llm: Optional[Text2TextLLM] = None,
        retrieve_top_k: int = 5,
        client_param: ClientParam = ClientParam(),
    ):
        self.base_llm = base_llm
        self.final_llm = final_llm
        self.sub_answer_llm = sub_answer_llm
        self.retrieve_api_url = retrieve_api_url
        self.retrieve_top_k = retrieve_top_k
        self.client_param = client_param

        self._client = RequestUtils(client_param=self.client_param)

        self.lock = threading.Lock()

    def sample_path(self, query: str, task_desc: str, max_path_length: int = 3) -> ReasoningPath:
        """
        生成CoRAG推理路径，通过迭代生成子查询，收集子答案和相关文档，构建完整的推理过程。
        通过控制LLM调用次数和子查询数量，确保生成的路径在合理范围内。
        """
        # Initialize or use provided interaction history
        interaction_queries: List[str] = []
        interaction_answers: List[str] = []
        retrieved_doc_ids: List[List[str]] = []
        retrieved_docs: List[List[str]] = []
        thought_process: List[str] = []

        # Configure LLM parameters and call limits
        original_temp = self.base_llm.llm_config.temperature
        llm_call_count = 0
        max_allowed_calls = 4 * max_path_length

        # Generate additional subqueries until reaching max path length or call limit
        while len(interaction_queries) < max_path_length and llm_call_count < max_allowed_calls:
            llm_call_count += 1

            # Generate follow-up query prompt
            followup_prompt = get_generate_subquery_prompt(
                query=query,
                past_subqueries=interaction_queries,
                past_subanswers=interaction_answers,
                task_desc=task_desc,
            )

            # Ensure prompt fits within token limits
            followup_prompt = truncate_long_text_by_char(
                followup_prompt, max_token_length=self.base_llm.llm_config.max_tokens
            )

            # Generate and process subquery
            generated_subquery = self.base_llm.chat(query=followup_prompt)
            processed_query, step_reasoning = _process_subquery(generated_subquery)

            # Check for duplicate queries and adjust temperature if needed
            with self.lock:
                # Always reset to original temperature first
                self.base_llm.llm_config.temperature = original_temp

                if processed_query in interaction_queries:
                    # Increase temperature for more diverse results if duplicate found
                    self.base_llm.llm_config.temperature = max(original_temp, 0.7)
                    continue

            # Get answer and related documents for the new subquery
            query_answer, current_doc_ids, current_docs = self._get_subanswer_and_doc_ids(subquery=processed_query)

            # Update interaction history
            interaction_queries.append(processed_query)
            interaction_answers.append(query_answer)
            retrieved_doc_ids.append(current_doc_ids)
            retrieved_docs.append(current_docs)
            thought_process.append(step_reasoning)

        # Create and return the complete reasoning path
        complete_path = ReasoningPath(
            original_query=query,
            subqueries=interaction_queries,
            subanswers=interaction_answers,
            document_ids=retrieved_doc_ids,
            documents=retrieved_docs,
            reasoning_steps=thought_process,
        )
        return complete_path

    def generate_final_answer(self, rag_path: ReasoningPath, task_description: str) -> str:
        """
        基于完整的推理路径生成最终答案。

        该方法接收一个包含完整查询和推理历史的 ReasoningPath 对象，
        结合任务描述和可选的参考文档，通过 LLM 生成最终的综合答案。

        Args:
            rag_path: 包含查询和推理历史的 ReasoningPath 对象
            task_description: 任务的详细描述

        Returns:
            生成的最终答案字符串
        """
        # Generate final answer prompt based on the reasoning path and context
        final_prompt = get_generate_final_answer_prompt(
            original_query=rag_path.original_query,
            interaction_queries=rag_path.subqueries or [],
            interaction_answers=rag_path.subanswers or [],
            task_instructions=task_description,
            reference_docs=rag_path.documents or [],
        )

        # Determine which LLM to use for final answer generation
        answer_llm = self.final_llm if self.final_llm is not None else self.base_llm

        # Ensure the prompt fits within the token limits of the selected LLM
        final_prompt = truncate_long_text_by_char(final_prompt, max_token_length=answer_llm.llm_config.max_tokens)

        # Generate and return the final comprehensive answer
        return answer_llm.chat(query=final_prompt)

    def _get_subanswer_and_doc_ids(self, subquery: str) -> Tuple[str, List, List[str]]:
        """这段代码的主要功能是根据子查询从数据源检索相关文档，并生成一个子答案。
        它还处理了消息的截断和聊天客户端的调用，最终返回子答案和相关的文档信息。
        """
        documents = []
        doc_ids = []
        retriever_results = []
        if self.retrieve_api_url:
            request_body = {"query": subquery, "top_k": self.retrieve_top_k}
            request_body["stream"] = False
            response = self._client.post(
                url=self.retrieve_api_url, body=json.dumps(request_body), headers={"Content-Type": "application/json"}
            )
            if response.success:
                try:
                    data = json.loads(response.data)
                    retriever_results = normalize_retrieve_api_results(data)
                except json.JSONDecodeError as e:
                    logger.error(f"response content cannot convert to json format: {e}")
                    retriever_results = []
                except Exception as e:
                    logger.error(f"unexpected error while parsing JSON response. Error: {e}")
                    retriever_results = []

            for res in retriever_results:
                if isinstance(res, str):
                    documents.append(res)
                    doc_ids.append('graph_chunk')
                elif isinstance(res, dict):
                    content = res.get('contents') or res.get('content') or res.get('text') or str(res)
                    documents.append(content)
                    doc_ids.append(str(res.get('id') or res.get('doc_id') or 'graph_chunk'))
        prompt = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        client = self.sub_answer_llm if self.sub_answer_llm else self.base_llm
        prompt = truncate_long_text_by_char(prompt, max_token_length=client.llm_config.max_tokens)
        subanswer: str = client.chat(query=prompt)
        return subanswer, doc_ids, documents
