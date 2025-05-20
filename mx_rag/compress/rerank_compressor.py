# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import List, Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mx_rag.compress import PromptCompressor
from mx_rag.reranker import Reranker
from mx_rag.utils.common import validate_params, \
    BOOL_TYPE_CHECK_TIP, MAX_PAGE_CONTENT, MAX_QUERY_LENGTH, \
    STR_TYPE_CHECK_TIP, CALLABLE_TYPE_CHECK_TIP


class RerankCompressor(PromptCompressor):
    @validate_params(
        reranker=dict(validator=lambda x: isinstance(x, Reranker),
                      message="param must be instance of Reranker"),
        splitter_func=dict(validator=lambda x: isinstance(x, Callable) or x is None,
                           message="param must be Callable[[str], List[str]] function or None"),
    )
    def __init__(self,
                 reranker: Reranker,
                 splitter_func: Callable[[str], List[str]] = None
                 ):
        self.reranker = reranker
        self.splitter_func = splitter_func

    @staticmethod
    def _ranked_texts(sentences_list, sorted_idx, target_rate, context_reorder):
        # 压缩策略：按照排序，优先保留相似性高的句子，直到达到目标
        reserved_ctx_ids = []
        context_sentences_lens = [len(t) for t in sentences_list]
        context_sentences_len_sum = sum(context_sentences_lens)
        target_sentences_len = target_rate * context_sentences_len_sum

        r_set = set()
        for idx, _ in sorted_idx:
            if idx not in r_set:
                reserved_ctx_ids.append(idx)
                r_set.add(idx)

            target_sentences_len -= context_sentences_lens[idx]
            if target_sentences_len < 0:
                break
        if not context_reorder:
            reserved_ctx_ids = sorted(reserved_ctx_ids)
        compressed_text = ''.join([sentences_list[i] for i in reserved_ctx_ids])
        return compressed_text

    @staticmethod
    def _split_text(text: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=0,
            separators=["\n", ""],
            keep_separator=True
        )
        sentences_list = text_splitter.split_text(text=text)
        return sentences_list

    @validate_params(
        context=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PAGE_CONTENT,
                     message=f"param must be str, and length range [1, {MAX_PAGE_CONTENT}]"),
        question=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_QUERY_LENGTH,
                      message=STR_TYPE_CHECK_TIP + f", and length range [1, {MAX_QUERY_LENGTH}]"),
        target_rate=dict(validator=lambda x: isinstance(x, float) and 0 < x < 1,
                         message="param must be float, and value range (0, 1)"),
        context_reorder=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def compress_texts(self,
                       context: str,
                       question: str,
                       target_rate: float = 0.6,
                       context_reorder: bool = False):
        if self.splitter_func is None:
            self.splitter_func = self._split_text
        # 文本切分
        sentences_list = self.splitter_func(context)
        # 句子排序
        ranker_result = self.reranker.rerank(query=question, texts=sentences_list)
        sorted_idx = sorted(enumerate(ranker_result), key=lambda x: x[1], reverse=True)
        return self._ranked_texts(sentences_list, sorted_idx, target_rate, context_reorder)
