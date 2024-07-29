# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
MXRAGCache 核心功能类
该类主要是给RAG框架提供数据缓存的能力，包括以下功能
1、缓存实例的构造(get_cache, new_cache)
2、缓存的查询(search)，更新(update)，以及刷新(flush)
3、缓存的级联功能(join)
"""
from typing import Any

from gptcache.core import Cache
from loguru import logger

from mx_rag.cache.cache_config import CacheConfig
from mx_rag.cache.cache_api import init_mxrag_cache


def _default_dump(data: Any) -> str:
    return data


def _default_load(data: str) -> Any:
    return data


class MxRAGCache:
    verbose: bool = False

    def __init__(self,
                 cache_name: str,
                 config: CacheConfig):
        self.cache_name = cache_name
        self.cache_obj = Cache()

        init_mxrag_cache(self.cache_obj, self.cache_name, config)

    @staticmethod
    def _update(
            llm_data, update_cache_func, *args, **kwargs
    ) -> None:
        """When updating cached data, do nothing, because currently only cached queries are processed"""
        from gptcache.adapter.api import _update_cache_callback
        _update_cache_callback(llm_data, update_cache_func, *args, **kwargs)
        return llm_data

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls.verbose = verbose

    def search(self, query: str):
        """
        MXRAGCache 查询缓存

        Args:
            query: 需要被查询的缓存问题
        Return:
            answer: 如果命中则为缓存问题，未命中则返回None
        """
        from gptcache.adapter.api import adapt, _cache_data_converter

        def llm_handle_none(*llm_args, **llm_kwargs) -> None:
            """Do nothing on a cache miss"""
            return None

        answer = adapt(
            llm_handle_none,
            _cache_data_converter,
            self._update,
            prompt=query,
            cache_obj=self.cache_obj
        )

        if answer is not None:
            self._verbose_log("Hit!")
        else:
            self._verbose_log("Miss!")
        return answer

    def update(self, query: str, answer: str):
        """
        MXRAGCache 更新缓存

        Args:
            query: 需要被缓存的用户问题
            answer: 需要被缓存的用户答案
        Return:
            None
        """
        from gptcache.adapter.api import adapt, _cache_data_converter

        def llm_handle(*llm_args, **llm_kwargs):
            return answer

        adapt(
            llm_handle,
            _cache_data_converter,
            self._update,
            cache_skip=True,
            prompt=query,
            cache_obj=self.cache_obj
        )
        self._verbose_log("Update!")

    def flush(self):
        """
        MXRAGCache 强制将缓存数据从内存刷新到磁盘

        Return:
            None
        """
        self.cache_obj.flush()
        self._verbose_log("Flush!")

    def get_obj(self):
        """
        MXRAGCache 获得gpt缓存示例，用于兼容langchain等 RAG开源框架

        Return:
            gptcache
        """
        return self.cache_obj

    def join(self, next_cache):
        """
        MXRAGCache 缓存级联

        Args:
            next_cache: 下级缓存
        Return:
            None
        """
        self.cache_obj.next_cache = next_cache.get_obj()

    def _verbose_log(self, log_str: str):
        """
        MXRAGCache 根据verbose标志 用于表示是否记录日志。

        Args:
            log_str: 日志信息
        Return:
            None
        """
        if self.verbose:
            logger.info(f"{self.cache_name}: " + log_str)
