# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re

from langchain.prompts import PromptTemplate
from tqdm import tqdm
from loguru import logger

from mx_rag.llm import Text2TextLLM

GENERATE_QA_MAX_LEN = 10000
GENERATE_QA_PROMPT = """阅读文章，生成一个相关的问题，例如：
文章：气候变化对海洋生态系统造成了严重的影响，其中包括海洋温度上升、海平面上升、酸化等问题。这些变化对海洋生物种群分布、生态圈的稳定性以及渔业等方面都产生了深远影响。在全球变暖的背景下，保护海洋生态系统已经成为当务之急。 
问题：气候变化对海洋生态系统的影响主要体现在哪些方面？
文章：零售业是人工智能应用的另一个重要领域。通过数据分析和机器学习算法，零售商可以更好地了解消费者的购买行为、趋势和偏好。人工智能技术可以帮助零售商优化库存管理、推荐系统、市场营销等方面的工作，提高销售额和客户满意度。
问题：人工智能是如何帮助零售商改善客户体验和销售业绩的？
请仿照样例对以下文章提{question_number}个相关问题：

文章：{doc}

输出格式为以下，按照问题1，问题2...进行编号，冒号后面不要再出现数字编号：
问题1：...
...

"""


def generate_qa_embedding_pairs(llm: Text2TextLLM, doc_list: list[str], question_number: int = 1):
    """使用大模型生成问题对"""

    if len(doc_list) > GENERATE_QA_MAX_LEN:
        logger.error(f"generate_qa_embedding_pairs's inputs len should not bigger than {GENERATE_QA_MAX_LEN}")
        return {}

    generate_qa_prompt = PromptTemplate(
        input_variables=["doc", "question_number"],
        template=GENERATE_QA_PROMPT,
    )

    doc_queries = {}
    for doc_content in tqdm(doc_list):
        prompt = generate_qa_prompt.format(
            doc=doc_content,
            question_number=question_number
        )
        result = llm.chat(prompt, max_tokens=512)
        rs_list = [re.sub(r"^[^：]+：", "", item) for item in result.split("\n")]
        doc_queries[doc_content] = rs_list

    return doc_queries
