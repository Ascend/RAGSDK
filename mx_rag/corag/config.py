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

from typing import Optional
from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.utils.common import validate_params, MAX_URL_LENGTH

# Constants
DEFAULT_TASK_DESC = (
    "You are a helpful assistant that can use search tools to solve complex multi-step questions. "
    "When you receive a question, you should decompose it into several simple sub-queries. "
    "After receiving the retrieved context for each sub-query, provide a sub-answer. "
    "Finally, give the final answer based on all information. "
    "Follow the format strictly: SubQuery, SubAnswer, and Final Answer. /no_think"
)

DEFAULT_SAMPLE_TASK_DESC = "answer multi-hop questions"


class CoRagBaseConfig:
    """CoRAG 基础配置类，包含共享的核心参数。
    
    Attributes:
        base_llm: 基础LLM实例，用于生成子查询和答案。
        retrieve_api_url: 检索API的URL地址。
        num_threads: 并行处理的线程数，默认为8。
        max_path_length: 最大路径长度，默认为3。
        judge_llm: 判断LLM实例，用于评估答案正确性（可选）。
        final_llm: 最终答案生成LLM实例（可选）。
        sub_answer_llm: 子答案生成LLM实例（可选）。

    """
    
    
    def __init__(
        self,
        base_llm: Text2TextLLM,
        retrieve_api_url: str,
        num_threads: int = 8,
        max_path_length: int = 3,
        final_llm: Optional[Text2TextLLM] = None,
        sub_answer_llm: Optional[Text2TextLLM] = None,
        judge_llm: Optional[Text2TextLLM] = None,
    ):
        self.base_llm = base_llm
        self.retrieve_api_url = retrieve_api_url
        self.num_threads = num_threads
        self.max_path_length = max_path_length
        self.final_llm = final_llm
        self.sub_answer_llm = sub_answer_llm
        self.judge_llm = judge_llm
