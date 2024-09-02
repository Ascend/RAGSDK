# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import operator
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


class SimilarityStrategy(Enum):
    FLAT_L2 = 0
    FLAT_IP = 1
    FLAT_COS = 2


class VectorStore(ABC):
    def __init__(self):
        self.score_comparator = operator.le

    @abstractmethod
    def delete(self, ids):
        pass

    @abstractmethod
    def search(self, embeddings, k):
        pass

    @abstractmethod
    def add(self, embeddings, ids):
        pass

    def search_with_threshold(self, embeddings: np.ndarray, k: int = 3, threshold: float = 0.1):
        """
        根据阈值进行查找 过滤调不满足的分数
        Args:
            embeddings: 词嵌入之后的查询
            k: top_k个结果
            threshold: 阈值

        Returns: 通过search过滤之后的分数

        """
        scores, indices = self.search(embeddings, k)

        if self.score_comparator == operator.le:
            logger.info(f"Filter is [<={threshold}]")
        else:
            logger.info(f"Filter is [>={threshold}]")

        for i, score in enumerate(scores[0]):
            if not self.score_comparator(score, threshold):
                scores[0].pop(i)
                indices[0].pop(i)

        return scores, indices

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
