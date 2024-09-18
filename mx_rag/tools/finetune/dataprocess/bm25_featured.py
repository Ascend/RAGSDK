# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from loguru import logger
from mx_rag.utils.common import validate_params

BM25_FEATURED_MAX_LEN = 10000


@validate_params(
    query_list=dict(validator=lambda x: 0 < len(x) <= BM25_FEATURED_MAX_LEN, message="param length range (0, 10000]"),
    doc_list=dict(validator=lambda x: 0 < len(x) <= BM25_FEATURED_MAX_LEN, message="param length range (0, 10000]")
)
def bm25_featured(query_list: list[str], doc_list: list[str]):
    """bm25对文档对打分"""

    if len(query_list) != len(doc_list):
        logger.error(f"bm25_featured query_list and doc_list has different len")
        return []

    score_list = []

    def chinese_tokenizer(text):
        return list(jieba.cut(text))

    # 对每个文档进行分词
    tokenized_doc = [chinese_tokenizer(doc) for doc in doc_list]
    bm25 = BM25Okapi(tokenized_doc)
    for index, query in enumerate(tqdm(query_list, desc="bm25 sort", disable=len(query_list) < 128)):
        tokenized_query = chinese_tokenizer(query)
        doc_scores = bm25.get_scores(tokenized_query)
        score_list.append(doc_scores[index])
    return score_list
