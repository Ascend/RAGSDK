# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re

from tqdm import tqdm
from loguru import logger
from langchain.prompts import PromptTemplate

LLM_PREFERRED_MAX_LEN = 10000
SCORING_QD_PROMPT = """您的任务是评估给定问题与文档之间的相关性。相关性评分应该在0到1之间，其中1表示非常相关，0表示不相关。评分应该基于文档内容回答问题的直接程度。

请仔细阅读问题和文档，然后基于以下标准给出一个相关性评分：
- 如果文档直接回答了问题，给出接近1的分数。
- 如果文档与问题相关，但不是直接回答，给出一个介于0和1之间的分数，根据相关程度递减。
- 如果文档与问题不相关，给出0。

例如：
问题：小明昨天吃了什么饭？
文档：小明昨天和朋友出去玩，还聚了餐，吃的海底捞，真是快乐的一天。
因为文档直接回答了问题的内容，因此给出0.99的分数

问题：小红学习成绩怎么样？
文档：小红在班上上课积极，按时完成作业，帮助同学，被老师评为了班级积极分子。
文档中并没有提到小红的学习成绩，只是提到了上课积极，按时完成作业，因此给出0.10的分数

请基于上述标准，为以下问题与文档对给出一个相关性评分，评分分数保留小数点后2位数：
问题: {query}
文档: {doc}

"""


def llm_preferred(llm, query_list: list[str], doc_list: list[str]):
    """大模型打分"""

    if len(query_list) > LLM_PREFERRED_MAX_LEN or len(doc_list) > LLM_PREFERRED_MAX_LEN:
        logger.error(f"llm_preferred's inputs len should not bigger than {LLM_PREFERRED_MAX_LEN}")
        return []

    if len(query_list) != len(doc_list):
        logger.error(f"llm_preferred's query_list and doc_list has different length")
        return []

    scoring_qd_prompt = PromptTemplate(input_variables=["query", "doc"], template=SCORING_QD_PROMPT)
    score_list = []
    for query, doc in tqdm(list(zip(query_list, doc_list))):
        prompt = scoring_qd_prompt.format(query=query, doc=doc)
        result = llm.chat(prompt, max_tokens=512)
        # 使用正则表达式提取相关性评分中的小数
        match = re.search(r"\d+\.\d+", result)
        score = float(0)
        if match:
            score = float(match.group())
        score_list.append(score)

    return score_list
