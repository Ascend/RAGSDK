# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import operator
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Union

import numpy as np
from loguru import logger


class SearchMode(Enum):
    DENSE = 0  # dense search
    SPARSE = 1  # sparse search
    HYBRID = 2  # hybrid search


class VectorStore(ABC):
    MAX_VEC_NUM = 100 * 1000 * 1000 * 1000
    MAX_SEARCH_BATCH = 1024 * 1024

    def __init__(self):
        self.score_scale = None

    @abstractmethod
    def delete(self, ids):
        pass

    @abstractmethod
    def search(self, embeddings, k):
        pass

    @abstractmethod
    def add(self, embeddings, ids):
        pass

    @abstractmethod
    def add_sparse(self, ids, sparse_embeddings):
        pass

    @abstractmethod
    def add_dense_and_sparse(self, ids, dense_embeddings, sparse_embeddings):
        pass

    @abstractmethod
    def get_all_ids(self):
        pass

    def search_with_threshold(self, embeddings: Union[np.ndarray, List[Dict[int, float]]],
                              k: int = 3, threshold: float = 0.1):
        """
        根据阈值进行查找 过滤调不满足的分数
        Args:
            embeddings: 词嵌入之后的查询
            k: top_k个结果
            threshold: 阈值

        Returns: 通过search过滤之后的分数

        """
        scores, indices = self.search(embeddings, k)[:2]

        logger.info(f"Filter is [>={threshold}]")

        filter_score = []
        filter_indices = []
        for i, score in enumerate(scores[0]):
            if score >= threshold:
                filter_score.append(scores[0][i])
                filter_indices.append(indices[0][i])

        return [filter_score], [filter_indices]

    def as_retriever(self, **kwargs):
        """
        矢量数据库转换为适量检索器
        Args:
            **kwargs:

        Returns: Retriever

        """
        from mx_rag.retrievers.retriever import Retriever

        return Retriever(vector_store=self, **kwargs)

    def save_local(self):
        pass

    def get_save_file(self):
        return ""

    def get_ntotal(self) -> int:
        return 0

    def _score_scale(self, scores: List[List[float]]) -> List[List[float]]:
        """
        分数量化
        Args:
            scores: 词嵌入得得分

        Returns: 量化之后的分数

        """
        if self.score_scale is not None:
            scores = [[self.score_scale(x) for x in row] for row in scores]
        return scores
