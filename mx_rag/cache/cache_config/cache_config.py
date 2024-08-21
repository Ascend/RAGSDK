# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
MXRAGCache 配置功能类
提供对外的配置参数，CacheConfig继承子gptCache的Config类，默认为memory_cache
SimilarityCacheConfig 继承CacheConfig提供 语义相似cache
"""
import os
from enum import Enum
from typing import Dict, Any, Union

from gptcache.config import Config
from gptcache.embedding.base import BaseEmbedding
from gptcache.manager import CacheBase, VectorBase
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation import SimilarityEvaluation

from mx_rag.utils.file_operate import check_disk_free_space

FREE_SPACE_LIMIT = 1024 * 1024 * 1024


class EvictPolicy(Enum):
    """
    功能描述:
        缓存替换策略

    Attributes:
        LRU(Least Recently Used):替换最近最少使用的缓存
        LFU(Least Frequently Used):替换最不常使用的缓存
        FIFO(First In First Out):替换最先被调入cache的缓存
        RR(Random Replacement):随机替换缓存
    """
    LRU: str = 'LRU'
    LFU: str = 'LFU'
    FIFO: str = 'FIFO'
    RR: str = 'RR'


def _get_default_save_folder():
    return "/usr/local/Ascend/mxRag/cache_save_folder"


class CacheConfig(Config):
    """
    功能描述:
        CacheConfig 继承子gptcache的Config，扩展了gptcache的参数

    Attributes:
        config_type: (str) 表明缓存类型，默认为memory_cache_config
        cache_size: (int) 缓存大小，单位是条
        eviction_policy: (EvictPolicy) 替换策略，包含(LRU, LFU, FIFO, RR)
        pre_emb_func: (Callable[[data: Dict[str, Any], **_: Dict[str, Any]], Any]) embedding前的适配函数
        data_save_folder: (str) 缓存数据存储路径
        **kwargs: 配置基类的参数
    """

    def __init__(self,
                 cache_size: int,
                 eviction_policy: EvictPolicy = EvictPolicy.LRU,
                 pre_emb_func=get_prompt,
                 data_save_folder: str = _get_default_save_folder(),
                 min_free_space: int = FREE_SPACE_LIMIT,
                 **kwargs):
        super().__init__(**kwargs)
        self.config_type = "memory_cache_config"
        self.cache_size = cache_size
        self.eviction_policy = eviction_policy
        self.pre_emb_func = pre_emb_func
        self.data_save_folder = data_save_folder
        self.min_free_space = min_free_space

        similarity_threshold = kwargs.get("similarity_threshold", 0.8)
        if not (1 >= similarity_threshold >= 0):
            raise ValueError("similarity_threshold must 0 ~ 1 range")

        if not isinstance(self.cache_size, int):
            raise TypeError("cache_size type error")

        if self.cache_size <= 0:
            raise ValueError("cache_size should greater than zero")

        if not isinstance(self.eviction_policy, EvictPolicy):
            raise TypeError("eviction_policy type error")

        if not isinstance(self.data_save_folder, str):
            raise TypeError("data_save_folder type error")

        if check_disk_free_space(os.path.dirname(self.data_save_folder), self.min_free_space):
            raise Exception("Insufficient remaining space, please clear disk space")


class SimilarityCacheConfig(CacheConfig):
    """
    功能描述:
        SimilarityCacheConfig 继承自CacheConfig，在CacheConfig基础上扩展了语义相似缓存参数

    Attributes:
        config_type: (str) 表明缓存类型，similarity_cache_config
        vector_config: Union[VectorBase, Dict[str, Any]] 向量数据库配置参数，当为VectorBase类型时，则需要由用户构建
                        当为Dict类型时由内部构建
        cache_config: Union[CacheBase, str] 缓存数据库配置参数，当为CacheBase类型时，则需要由用户构建
                        当为str类型时由内部构建
        emb_config: Union[BaseEmbedding, Dict[str, Any]] embedding 配置参数，当为BaseEmbedding，则需要由用户构建
                        当为Dict类型时 由内部构建
        similarity_config: Union[SimilarityEvaluation, Dict[str, Any]] 相似度 配置参数，当为SimilarityEvaluation
                        则需要用户构建，当为Dict时则内部构建
        retrieval_top_k: int 检索时的TOPK参数
        clean_size: int 每次添加满的时候删除的元素个数 1 表示每次删除一个
        **kwargs: 配置基类的参数
    """

    def __init__(self,
                 retrieval_top_k: int = 1,
                 clean_size: int = 1,
                 vector_config: Union[VectorBase, Dict[str, Any]] = None,
                 cache_config: Union[CacheBase, str] = None,
                 emb_config: Union[BaseEmbedding, Dict[str, Any]] = None,
                 similarity_config: Union[SimilarityEvaluation, Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.config_type = "similarity_cache_config"
        self.vector_config = vector_config
        self.cache_config = cache_config
        self.emb_config = emb_config
        self.similarity_config = similarity_config
        self.retrieval_top_k = retrieval_top_k
        self.clean_size = clean_size

        if isinstance(self.cache_config, str) and self.cache_config != "sqlite":
            raise ValueError("cache_config only support sqlite now.")

        if not isinstance(self.retrieval_top_k, int):
            raise TypeError("retrieval_top_k type error")

        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must greater equal than zero")

        if not isinstance(self.clean_size, int):
            raise TypeError("clean size type error")

        if not (0 < self.clean_size <= self.cache_size):
            raise ValueError("clean size value range is (0, cache_size]")
