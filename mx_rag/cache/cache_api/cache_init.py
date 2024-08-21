# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
MXRAGCache 提供的对外API，用于初始化MXRAGCache的参数，在MXRAGCache运行之前进行调用
"""
import os
from typing import Dict

import cachetools
from gptcache import Cache
from gptcache.manager.scalar_data import CacheBase
from gptcache.similarity_evaluation import ExactMatchEvaluation
from loguru import logger

from mx_rag.cache.cache_config import CacheConfig, SimilarityCacheConfig, EvictPolicy
from mx_rag.cache.cache_similarity import CacheSimilarity
from mx_rag.cache.cache_storage import CacheVecStorage
from mx_rag.cache.cache_emb import CacheEmb
from mx_rag.utils.file_check import FileCheck


def _get_data_save_file(data_save_folder: str, cache_name: str, memory_only: bool = False):
    """
    功能描述:
        内部接口，根据用户配置创建缓存数据目录和文件

    Args:
        data_save_folder:str 缓存存储目录 由用户提供，需要符合安全标准
        cache_name:str cache实例的键值
        memory_only:bool 是否是memory_cache缓存 如果是则只需要创建data_map文件
    Return:
        vector_save_file:str similarity cache向量数据库缓存文件
        sql_save_file:str similarity cache缓存数据缓存文件
        data_save_file:str memory cache的缓存文件
    """
    FileCheck.check_input_path_valid(data_save_folder)

    if not os.path.exists(data_save_folder):
        os.makedirs(data_save_folder)
        logger.info(f"creat cache data save folder {data_save_folder}")
    else:
        logger.info(f"find cache data save folder {data_save_folder}")

    file_prefix = cache_name

    vector_save_file = ""
    sql_save_file = ""
    data_save_file = ""
    if not memory_only:
        vector_save_file = os.path.join(data_save_folder, f"{file_prefix}_vector_cache_file.index")
        logger.info(f"vector cache data save file : {vector_save_file}")

        sql_save_file = os.path.join(data_save_folder, f"{file_prefix}_sql_cache_file.db")
        logger.info(f"sql cache data save file : {sql_save_file}")
    else:
        data_save_file = os.path.join(data_save_folder, f"{file_prefix}_data_map.txt")
        logger.info(f"data map save file : {data_save_file}")

    return vector_save_file, sql_save_file, data_save_file


# 初始化语义近似 cache
def _init_mxrag_similar_cache(cache_obj: Cache, cache_name: str, config: SimilarityCacheConfig):
    """
    功能描述:
        内部接口，根据SimilarityCacheConfig 初始化指定cache_name的cache实例，为语义相关cache

    Args:
        cache_obj:str 缓存存储目录 由用户提供，需要符合安全标准
        cache_name:str cache实例的键值
        config:SimilarityCacheConfig 语义相似缓存配置数据
    Return:
        None
    """
    from gptcache.manager import get_data_manager
    from gptcache.adapter.api import init_similar_cache

    vector_save_file, sql_save_file, _ = _get_data_save_file(config.data_save_folder, cache_name)

    config.vector_config["vector_save_file"] = vector_save_file
    config.vector_config["top_k"] = config.retrieval_top_k
    vector_base = CacheVecStorage.create(**config.vector_config) \
        if isinstance(config.vector_config, Dict) else config.vector_config

    cache_base = CacheBase(config.cache_config, sql_url=f'{config.cache_config}:///{sql_save_file}') \
        if isinstance(config.cache_config, str) else config.cache_config

    similarity = CacheSimilarity.create(**config.similarity_config) \
        if isinstance(config.similarity_config, Dict) else config.similarity_config

    embedding = CacheEmb.create(**config.emb_config) if isinstance(config.emb_config, Dict) \
        else config.emb_config

    data_manager = get_data_manager(
        cache_base=cache_base,
        vector_base=vector_base,
        max_size=config.cache_size,
        eviction=config.eviction_policy.value,
        clean_size=config.clean_size
    )

    init_similar_cache(
        pre_func=config.pre_emb_func,
        cache_obj=cache_obj,
        data_manager=data_manager,
        embedding=embedding,
        evaluation=similarity,
        config=config
    )


def _init_mxrag_memory_cache(cache_obj: Cache, cache_name: str, config: CacheConfig):
    """
    功能描述:
        内部接口，根据CacheConfig 初始化指定cache_name的cache实例，为memory_cache only

    Args:
        cache_obj:str 缓存存储目录 由用户提供，需要符合安全标准
        cache_name:str cache实例的键值
        config:CacheConfig memory_cache 缓存配置数据
    Return:
        None
    """
    from gptcache.manager.data_manager import MapDataManager
    from gptcache.adapter.api import init_similar_cache

    _, _, data_save_file = _get_data_save_file(config.data_save_folder, cache_name, True)

    evict_policy_memory_map = {
        EvictPolicy.LRU.value: cachetools.LRUCache,
        EvictPolicy.LFU.value: cachetools.LFUCache,
        EvictPolicy.FIFO.value: cachetools.FIFOCache,
        EvictPolicy.RR.value: cachetools.RRCache
    }

    data_manager = MapDataManager(data_save_file,
                                  config.cache_size,
                                  evict_policy_memory_map.get(config.eviction_policy.value, cachetools.LRUCache))

    init_similar_cache(
        pre_func=config.pre_emb_func,
        cache_obj=cache_obj,
        data_manager=data_manager,
        embedding=CacheEmb(skip_emb=True),
        evaluation=ExactMatchEvaluation(),
        config=config
    )


def init_mxrag_cache(cache_obj: Cache, cache_name: str, config):
    """
    功能描述:
        对外接口，根据config 初始化指定cache_name的cache实例

    Args:
        cache_obj:str 缓存存储目录 由用户提供，需要符合安全标准
        cache_name:str cache实例的键值
        config:CacheConfig/SimilarityCacheConfig 缓存配置数据
    Return:
        None
    Raises:
        ValueError: 当配置数据不在有效范围内时
    """
    if config.config_type == "similarity_cache_config":
        _init_mxrag_similar_cache(cache_obj, cache_name, config)
    elif config.config_type == "memory_cache_config":
        _init_mxrag_memory_cache(cache_obj, cache_name, config)
    else:
        raise ValueError("config type not support. ")
