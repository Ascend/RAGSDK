# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Any, Union, Dict, Iterator
import json

from mx_rag.chain import Chain
from mx_rag.cache.cache_core import MxRAGCache
from mx_rag.utils.common import validate_params, MAX_QUERY_LENGTH


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
        cache=dict(validator=lambda x: isinstance(x, MxRAGCache)),
        chain=dict(validator=lambda x: isinstance(x, Chain))
    )
    def __init__(self,
                 cache: MxRAGCache,
                 chain: Chain,
                 convert_data_to_cache=_default_data_convert,
                 convert_data_to_user=_default_data_convert,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self._cache = cache
        self._chain = chain
        self._convert_data_to_cache = convert_data_to_cache
        self._convert_data_to_user = convert_data_to_user

    @validate_params(
        text=dict(validator=lambda x: 0 < len(x) <= MAX_QUERY_LENGTH),
    )
    def query(self, text: str, *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        """
        MXRAGCache 根据verbose标志 用于表示是否记录日志。

        Args:
            text: 用户问题
        Return:
            ans: 用户答案
        """
        cache_ans = self._cache.search(query=text)
        if cache_ans is not None:
            return self._convert_data_to_user(json.loads(cache_ans))

        ans = self._chain.query(text, *args, **kwargs)

        result = ans
        # 如果是 stream对象需要通过迭代的方式把内容都读取完才能cache
        if isinstance(ans, Iterator):
            result = None
            for res in ans:
                result = res

        self._cache.update(query=text, answer=json.dumps(self._convert_data_to_cache(result)))
        return result
