# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
MXRAGCache 核心功能类
该类主要是给RAG框架提供数据缓存的能力，包括以下功能
1、缓存实例的构造(get_cache, new_cache)
2、缓存的查询(lookup)，更新(update)，以及清理(clear)
3、缓存的级联功能(join, unjoin)
"""
from typing import Optional, Callable, Any, Dict, cast

from langchain_core.caches import BaseCache
from loguru import logger


def _default_dump(data: Any) -> str:
    return data


def _default_load(data: str) -> Any:
    return data


class MXRAGCache(BaseCache):
    """
    功能描述:
        MXRAGCache 类，作为cache的核心，为RAG框架提供lookup, update, clear 和 cache申请释放操作缓存的方法，
        适配langchain和 mxrag

    Attributes:
        default_init_cache_func: 默认cache构造函数，由用户指定, 默认值为None,如果用户不指定 则使用默认cache构造
        init_cache_func: 特殊cache构造函数， 由用户注册，如果用户对应的cache没有注册，则使用default_init_cache_func
        load_ans_func: 适配 RAG框架的方法，将缓存的结果转换为RAG框架能处理的格式，由用户注册，如果用户不注册，则使用默认函数_default_load
        dump_ans_func/dump_query_func: 适配 RAG框架的方法，将RAG框架缓存的请求转换为MXRAGCache能处理的格式，由用户注册，如果用户不注册，则使用默认函数_default_dump
        cache_dict: 将cache_name作为键值，用于查询cache实例的字典
        cache_join_table: 将cache_name作为键值，用于查询下一级cache实例的字典
        verbose: 是否显示缓存过程中的日志，默认关闭。
    """

    def __init__(self, default_init_cache_func: Callable[[Any, str], None] = None,
                 verbose: bool = False):
        self.default_init_cache_func = default_init_cache_func
        self.init_cache_func: Dict[str, Callable[[Any, str], None]] = {}
        self.load_ans_func = _default_load
        self.dump_ans_func = _default_dump
        self.dump_query_func = _default_dump
        self.cache_dict: Dict[str, Any] = {}
        self.cache_join_table: Dict[str, str] = {}
        self.verbose = verbose

    @staticmethod
    def data_check(data: str) -> bool:
        """
        用于检查 来自RAG 框架请求数据是否合法的函数，MXRAGCache目前只能处理字符串对象

        Args:
            data:(str) 来自RAG 框架的请求数据
        Return:
            bool True:表示校验合法 False:表示校验非法
        """
        if data is None:
            logger.error("cache data is None error.")
            return False

        if not isinstance(data, str):
            logger.error("cache data type error.")
            return False

        if data == "":
            logger.error("cache data empty error.")
            return False
        return True

    def lookup(self, prompt: Any, llm_string: str):
        """
        MXRAGCache 对外提供的缓存查询函数。
        step1: 调用 dump_query_func将RAG 框架传递的参数进行适配，转换为内部可处理的数据query
        step2: 调用 内部函数_lookup进行查询
        step3: 调用 load_ans_func 将内部缓存数据转换为RAG框架可以处理的格式

        Args:
            prompt: (Any) 来自RAG框架查询的键值，任意类型
            llm_string: (str) cache的键值，指定查询的cache
        Return:
            answer: (str) 缓存结果
        """
        query = self.dump_query_func(prompt)
        ans = self._lookup(query, llm_string)
        return self.load_ans_func(ans) if ans is not None else None

    def update(self, prompt: Any, llm_string: str, return_val: Any) -> None:
        """
        MXRAGCache 对外提供的缓存更新函数。
        step1: 调用 dump_query_func将RAG 框架传递的参数进行适配，转换为内部可处理的数据query
        step2: 调用 dump_ans_func将RAG 框架传递的参数进行适配，转换为内部可处理的数据ans
        step3: 调用 _update 进行缓存更新

        Args:
            prompt: (Any) 来自RAG框架查询的键值，任意类型
            llm_string: (str) cache的键值，指定查询的cache
            return_val: (Any) 需要更新的缓存结果
        Return:
            None
        """
        query = self.dump_query_func(prompt)
        ans = self.dump_ans_func(return_val)
        self._update(query, llm_string, ans)

    def clear(self, **kwargs: Any) -> None:
        """
        MXRAGCache 的缓存刷新函数，会将缓存数据进行flush 刷新到磁盘文件，以便于下次使用

        Args:
            **kwargs
        Return:
            None
        """
        from gptcache import Cache

        for cache in self.cache_dict.values():
            cache = cast(Cache, cache)
            cache.flush()

        self.cache_dict.clear()

    def join(self, cache_name: str, next_cache_name: str) -> None:
        """
        MXRAGCache的 缓存级联函数，将cache_name和next_cache_name进行级联，这样
        cache_name对应的cache就会成为一个二级缓存

        Args:
            cache_name: (str) 第一级缓存的名字
            next_cache_name: (str) 第二级缓存的名字
        Return:
            None
        """
        if cache_name in self.cache_join_table:
            logger.warning(f"{cache_name} already have sub_cache will overwrite")

        self.cache_join_table[cache_name] = next_cache_name

    def unjoin(self, cache_name: str, next_cache_name: str) -> None:
        """
        MXRAGCache 的 失能级联函数，将cache_name的子cache(next_cache_name)断链

        Args:
            cache_name:  (str) 第一级缓存的名字
            next_cache_name: (str) 第二级缓存的名字
        Return:
            None
        Raises:
            IndexError: cache并不存在下级缓存时
            ValueError: 当传递的next_cache并不是当前cache的下级缓存时
        """
        if cache_name not in self.cache_join_table:
            raise IndexError(f"{cache_name} not join. ")

        if self.cache_join_table[cache_name] != next_cache_name:
            raise ValueError(f"{cache_name} not join with {next_cache_name} so cant unjoin. ")

        self.cache_join_table.pop(cache_name)

    def register_init_func(self, cache_name: str, init_cache_func: Callable[[Any, str], None], lazy_init: bool):
        """
        MXRAGCache 注册cache初始化函数，可以提供懒惰初始化功能。

        Args:
            cache_name: cache的键值
            init_cache_func: cache对应的初始化函数
            lazy_init: 是否懒惰初始化，如果为true则在使用该cache时才会进行初始化
        Return:
            None
        """
        if cache_name in self.init_cache_func:
            self._verbose_log(f"{cache_name} init func already register so will replace old. ")

        self.init_cache_func[cache_name] = init_cache_func
        if not lazy_init:
            self._new_cache(cache_name)

    def unregister_init_func(self, cache_name: str):
        """
        MXRAGCache 去注册cache初始化函数

        Args:
            cache_name: cache的键值
        Return:
            None
        """
        if cache_name not in self.init_cache_func:
            self._verbose_log(f"{cache_name} init func not register. ")
            return

        self.init_cache_func.pop(cache_name)

    def set_llm_adapter_api(self,
                            load_ans_func: Callable[[str], Any],
                            dump_ans_func: Callable[[Any], str],
                            dump_query_func: Callable[[Any], str]):
        """
        MXRAG 提供的设置适配RAG框架API的函数

        Args:
            load_ans_func: 适配命中cache的缓存结果转换为RAG框架可处理的函数
            dump_ans_func: 适配更新cache的缓存结果转换为内部可处理的函数
            dump_query_func: 适配查询/更新cache promt转换为内部可以处理的函数
        Return:
            None
        """
        if load_ans_func is not None:
            self.load_ans_func = load_ans_func

        if dump_ans_func is not None:
            self.dump_ans_func = dump_ans_func

        if dump_query_func is not None:
            self.dump_query_func = dump_query_func

    def _lookup(self, query: str, cache_name: str):
        """
        MXRAGCache 内部_lookup函数，根据query在指定的cache_name对应的cache实例中查询
        step1: 检查query数据合法性
        step2: 根据cache_name得到cache实例
        step3: 调用gptcache get方法进行查询缓存结果
        step4: 如果缓存结果ans 为None 则表示缓存未命中，如果cache_name存在下一级cache，则
               进行下一级缓存查询
        step5: 如果缓存结果ans 不为None，则表示缓存命中。
        step6: 返回缓存命中结果

        Args:
            query: 查询的键值
            cache_name: cache的键值
        Return:
            answer: cache的缓存结果
        Raises:
            ValueError: 当数据校验不通过时则抛出该异常
            NameError: 当cache_name不合法时则抛出该异常
        """
        from gptcache.adapter.api import get

        if not self.data_check(query):
            raise ValueError("input check failed. ")

        if cache_name == "":
            raise NameError(f"cache_name:{cache_name} is wrong. ")

        _cache_obj = self._get_cache(cache_name)
        ans = get(query, cache_obj=_cache_obj)

        if ans is not None:
            self._verbose_log(f"{cache_name} Hit!. ")
        else:
            self._verbose_log(f"{cache_name} Miss!. ")
            if cache_name in self.cache_join_table:
                ans = self._lookup_next(query, cache_name)

        return ans

    def _update(self, query: str, cache_name: str, answer: str) -> None:
        """
        MXRAGCache 内部_update函数，根据query在指定cache_name对应的缓存实例中更新数据为answer
        step1: 校验query和answer的数据合法性
        step2: 如果cache 存在下一级缓存则首先更新下一级缓存
        step3: 更新本级缓存

        Args:
            query: 查询的键值
            cache_name: cache的键值
            answer: 需要更新的数据
        Return:
            None
        Raises:
            ValueError: 当数据校验不通过时则抛出该异常
            NameError: 当cache_name不合法时则抛出该异常
        """
        if not self.data_check(query) or not self.data_check(answer):
            raise ValueError("input check failed. ")

        if cache_name == "":
            raise NameError(f"cache_name:{cache_name} is wrong. ")

        if cache_name in self.cache_join_table:
            self._update(query, self.cache_join_table[cache_name], answer)
        self._update_cache(query, cache_name, answer)

    def _lookup_next(self, query: str, cache_name: str) -> Optional[str]:
        """
        MXRAGCache 查询下级缓存的函数
        step1: 根据cache_name在cache_join_table中得到下一级缓存的键值
        step2: 根据下一级缓存的键值去递归调用_lookup查询数据
        step3: 如果命中 则更新当前缓存的数据，并将数据返回给上级缓存

        Args:
            query: 查询的键值
            cache_name: cache的键值
        Return:
            如果命中 则返回answer (str) 如果没有命中则返回None
        """
        next_cache_name = self.cache_join_table[cache_name]
        ans = self._lookup(query, next_cache_name)
        if ans is not None:
            self._update_cache(query, cache_name, ans)
        return ans

    def _update_cache(self, query: str, cache_name: str, ans: str) -> None:
        """
        MXRAGCache 更新本级cache的内部函数

        Args:
            query: 查询的键值
            cache_name: cache的键值
            ans: 需要被更新的缓存数据
        Return:
            None
        """
        from gptcache.adapter.api import put

        _cache_obj = self._get_cache(cache_name)
        put(query, ans, cache_obj=_cache_obj)

        self._verbose_log(f"{cache_name} Update!. ")

    def _new_cache(self, cache_name: str) -> Any:
        """
        MXRAGCache 构造cache的函数
        step1: 首先根据cache_name 查看是否有用户指定的cache构造方法(init_cache_func)，如果有则使用该方法进行构造
        step2: 其次查看用户是否设置了默认cache构造方法，如果有则使用该方法进行构造
        step3: 最后使用系统默认构造方法构造cache

        Args:
            cache_name: 需要被构造的cache的键值
        Return:
            缓存实例
        """
        from gptcache import Cache
        from gptcache.manager.factory import get_data_manager
        from gptcache.processor.pre import get_prompt

        _cache = Cache()

        if self.init_cache_func is not None \
                and cache_name in self.init_cache_func:

            self.init_cache_func[cache_name](_cache, cache_name)  # type: ignore[call-arg]
        elif self.default_init_cache_func is not None:
            self._verbose_log(f"because {cache_name} init func not register use default init func instead")

            self.default_init_cache_func(_cache, cache_name)
        else:
            self._verbose_log(f"because {cache_name} init func not register and no "
                              f"default init func. so use memory init func")

            _cache.init(
                pre_embedding_func=get_prompt,
                data_manager=get_data_manager(data_path=cache_name),
            )

        self.cache_dict[cache_name] = _cache
        return _cache

    def _verbose_log(self, log_str: str):
        """
        MXRAGCache 根据verbose标志 用于表示是否记录日志。

        Args:
            log_str: 日志信息
        Return:
            None
        """
        if self.verbose:
            logger.info(log_str)

    def _get_cache(self, cache_name: str) -> Any:
        """
        MXRAGCache 获取cache实例的函数
        step1: 是否根据cache_name查看对应的实例是否已经构造，如果已经构造则返回该实例
        step2: 如果没有构造则调用_new_cache构造该实例

        Args:
            cache_name: 构造的缓存键值
        Return:
            缓存实例
        """
        _gptcache = self.cache_dict.get(cache_name, None)
        if not _gptcache:
            _gptcache = self._new_cache(cache_name)
        return _gptcache