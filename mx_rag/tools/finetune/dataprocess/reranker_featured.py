# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from loguru import logger
from tqdm import tqdm

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import validate_params, validata_list_str, TEXT_MAX_LEN, STR_MAX_LEN

RERANKER_FEATURED_MAX_LEN = 10000


@validate_params(
    reranker=dict(validator=lambda x: isinstance(x, Reranker),
                  message="param must be instance of LocalReranker"),
    query_list=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                    message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                            "str length range [1, 128 * 1024 * 1024]"),
    doc_list=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                  message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                          "str length range [1, 128 * 1024 * 1024]")
)
def reranker_featured(reranker: LocalReranker, query_list: list[str], doc_list: list[str]):
    """重排模型打分"""

    if len(query_list) > RERANKER_FEATURED_MAX_LEN or len(doc_list) > RERANKER_FEATURED_MAX_LEN:
        logger.error(f"reranker_featured inputs len should not bigger than {RERANKER_FEATURED_MAX_LEN}")
        return []

    if len(query_list) != len(doc_list):
        logger.error(f"reranker_featured query_list and doc_list has different len")
        return []

    score_list = []

    for query, doc in tqdm(zip(query_list, doc_list), total=len(query_list)):
        scores = reranker.rerank(query, [doc])
        score_list.append(scores[0] if scores.size > 0 else 0)

    return score_list
