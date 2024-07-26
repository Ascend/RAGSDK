# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
MXRAGCache 的similarity 适配器类
"""
from typing import Dict, Tuple, Any

from gptcache.similarity_evaluation import SimilarityEvaluation
from loguru import logger

from mx_rag.reranker.reranker import Reranker
from mx_rag.reranker.reranker_factory import RerankerFactory


class CacheSimilarity(SimilarityEvaluation):
    """
    功能描述:
        CacheSimilarity 为MXRAG适配gptcache similarity功能的适配器

    Attributes:
        _similarity_impl: (Reranker) 来自MXRAG的reranker实例
        _score_min: (float) 相似度最小值 默认值0
        _score_max: (float) 相似度最大值 默认值1
    """

    def __init__(self, similarity: Reranker, score_min: float = 0.0, score_max: float = 1.0):
        if similarity is None:
            raise ValueError("CacheSimilarity init failed reranker is None.")

        self._similarity_impl = similarity
        self._score_min = score_min
        self._score_max = score_max

    @staticmethod
    def create(**kwargs):
        """
        构造CacheSimilarity的静态方法

        Args:
            kwargs:(Dict[str, Any]) similarity配置参数
        Return:
            similarity 适配器实例
        """
        score_min = kwargs.pop("score_min", 0.0)
        score_max = kwargs.pop("score_max", 1.0)

        similarity = RerankerFactory.create_reranker(**kwargs)
        similarity = CacheSimilarity(similarity, score_min, score_max)
        return similarity

    def evaluation(
            self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        进行相似度匹配

        Args:
            src_dict:(Dict[str, Any]) 被比较的数据
            cache_dict:(Dict[str, Any]) 比较的数据
        Return:
            score 比较分数
        """
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]

            if src_question.lower() == cache_question.lower():
                return 1

            scores = self._similarity_impl.rerank(src_question, [cache_question], batch_size=1)
            return scores[0]
        except Exception as e:
            logger.error(f"CacheSimilarity evaluation fatal error. {e}")
            return 0

    def range(self) -> Tuple[float, float]:
        return self._score_min, self._score_max
