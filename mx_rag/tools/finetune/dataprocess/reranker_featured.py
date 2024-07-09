# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from loguru import logger

from mx_rag.reranker.local import LocalReranker

RERANKER_FEATURED_MAX_LEN = 10000


def reranker_featured(model_path: str, query_list: list[str], doc_list: list[str], dev_id: int = 0):
    """重排模型打分"""

    if len(query_list) > RERANKER_FEATURED_MAX_LEN or len(doc_list) > RERANKER_FEATURED_MAX_LEN:
        logger.error(f"reranker_featured inputs len should not bigger than {RERANKER_FEATURED_MAX_LEN}")
        return []

    if len(query_list) != len(doc_list):
        logger.error(f"reranker_featured query_list and doc_list has different len")
        return []

    reranker = LocalReranker(model_path, dev_id=dev_id)

    score_list = []

    for query, doc in zip(query_list, doc_list):
        scores = reranker.rerank(query, [doc])
        score_list.append(scores[0] if scores.size > 0 else 0)

    return score_list
