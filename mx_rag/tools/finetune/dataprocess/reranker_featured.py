# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from loguru import logger

from mx_rag.reranker.local import LocalReranker
from mx_rag.utils.common import validate_params, MAX_DEVICE_ID

RERANKER_FEATURED_MAX_LEN = 10000


@validate_params(
    query_list=dict(validator=lambda x: 0 < len(x) <= RERANKER_FEATURED_MAX_LEN,
                    message="param length range (0, 10000]"),
    doc_list=dict(validator=lambda x: 0 < len(x) <= RERANKER_FEATURED_MAX_LEN,
                  message="param length range (0, 10000]"),
    dev_id=dict(validator=lambda x: 0 <= x <= MAX_DEVICE_ID, message="param value range [0, 63]")
)
def reranker_featured(model_path: str, query_list: list[str], doc_list: list[str], dev_id: int = 0):
    """重排模型打分"""

    if len(query_list) != len(doc_list):
        logger.error(f"reranker_featured query_list and doc_list has different len")
        return []

    reranker = LocalReranker(model_path, dev_id=dev_id)

    score_list = []

    for query, doc in zip(query_list, doc_list):
        scores = reranker.rerank(query, [doc])
        score_list.append(scores[0] if scores.size > 0 else 0)

    return score_list
