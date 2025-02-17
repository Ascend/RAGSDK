# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm
from loguru import logger

from mx_rag.tools.finetune.instruction import RuleComplexInstructionRewriter
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils.common import validate_params, validata_list_str, TEXT_MAX_LEN, STR_MAX_LEN

IMPROVE_QUERY_MAX_LEN = 10000
MAX_TOKENS = 512


@validate_params(
    llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
    old_query_list=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                        message="param must meets: Type is List[str], "
                                "list length range [1, 1000 * 1000], str length range [1, 128 * 1024 * 1024]")
)
def improve_query(llm: Text2TextLLM, old_query_list: list[str]):
    """问题重写"""

    if len(old_query_list) > IMPROVE_QUERY_MAX_LEN:
        logger.error(f"improve_query's inputs len should not bigger than {IMPROVE_QUERY_MAX_LEN}")
        return []

    new_query_list = multi_processing(llm, old_query_list)

    return new_query_list


def multi_processing(llm, query_list):
    logger.info('start to multi process improve query')

    # 使用 partial 传递固定参数
    make_request_partial = partial(make_request, llm)
    with ThreadPoolExecutor() as executor:
        answers = list(tqdm(executor.map(make_request_partial, query_list), total=len(query_list)))
    # 使用正则表达式提取相关性评分中的小数
    logger.info('end to multi process improve query')
    return answers


def make_request(llm, query):
    rewriter = RuleComplexInstructionRewriter()
    prompt = rewriter.get_rewrite_prompts(query, "更改指令语言风格")
    llm_config = LLMParameterConfig(max_tokens=MAX_TOKENS)
    try:
        response = llm.chat(prompt, llm_config=llm_config)
    except Exception:
        logger.error(f"llm chat failed")
        response = ''
    return response
