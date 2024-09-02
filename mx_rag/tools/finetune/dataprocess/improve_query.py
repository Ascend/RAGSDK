# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from tqdm import tqdm
from loguru import logger

from mx_rag.tools.finetune.instruction import RuleComplexInstructionRewriter
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params

IMPROVE_QUERY_MAX_LEN = 10000


@validate_params(
    old_query_list=dict(validator=lambda x: 0 < len(x) <= IMPROVE_QUERY_MAX_LEN),
)
def improve_query(llm: Text2TextLLM, old_query_list: list[str]):
    """问题重写"""

    new_query_list = []
    for query in tqdm(old_query_list):
        rewriter = RuleComplexInstructionRewriter()
        prompt = rewriter.get_rewrite_prompts(query, "更改指令语言风格")
        result = llm.chat(prompt)
        new_query_list.append(result)

    return new_query_list
