#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Union, Dict, Iterator, Callable
import json

from mx_rag.chain import Chain
from mx_rag.cache import MxRAGCache
from mx_rag.utils.common import validate_params, MAX_QUERY_LENGTH
from mx_rag.llm.llm_parameter import LLMParameterConfig


def _default_data_convert(data):
    return data


class CacheChainChat(Chain):
    """
    功能描述:
        适配cache的chain 对用户提供chain和cache的能力，当cache无法命中时，访问大模型
        更新cache

    Attributes:
        _cache: RAGCache
        _chain: 同大模型对话的模块
    """

    @validate_params(
        cache=dict(validator=lambda x: isinstance(x, MxRAGCache), message="param must be instance of MxRAGCache"),
        chain=dict(validator=lambda x: isinstance(x, Chain), message="param must be instance of Chain"),
        convert_data_to_cache=dict(validator=lambda x: isinstance(x, Callable),
                                   message="param must be callable function"),
        convert_data_to_user=dict(validator=lambda x: isinstance(x, Callable),
                                  message="param must be callable function")
    )
    def __init__(self,
                 cache: MxRAGCache,
                 chain: Chain,
                 convert_data_to_cache=_default_data_convert,
                 convert_data_to_user=_default_data_convert):
        self._cache = cache
        self._chain = chain
        self._convert_data_to_cache = convert_data_to_cache
        self._convert_data_to_user = convert_data_to_user

    @validate_params(
        text=dict(validator=lambda x: 0 < len(x) <= MAX_QUERY_LENGTH, message="param length range (0, 128*1024*1024]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig) or x is None,
                        message="param must be None or LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(), *args, **kwargs) \
            -> Union[Dict, Iterator[Dict]]:
        """
        MXRAGCache 根据verbose标志 用于表示是否记录日志。

        Args:
            llm_config: 大模型参数
            text: 用户问题
        Return:
            ans: 用户答案
        """
        cache_ans = self._cache.search(query=text)
        # 缓存存入为什么格式返回什么格式，可能不是json格式的
        if cache_ans is not None:
            try:
                answer = json.loads(cache_ans)
                if answer.get("query"):
                    answer["query"] = text
                return self._convert_data_to_user(answer)
            except Exception:
                return cache_ans

        ans = self._chain.query(text, llm_config)

        result = ans
        # 如果是 stream对象需要通过迭代的方式把内容都读取完才能cache
        if isinstance(ans, Iterator):
            result = None
            for res in ans:
                result = res

        self._cache.update(query=text, answer=json.dumps(self._convert_data_to_cache(result)))
        return result
