# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Reranker(ABC):
    def __init__(self, k):
        self._k = k

    @abstractmethod
    def rerank(self,
               query: str,
               texts: List[str]):
        """ rank the texts and query"""

    def rerank_top_k(self,
                     objs: List,
                     scores: np.array) -> List:

        obj_scores = list(zip(objs, scores))
        obj_scores.sort(reverse=True, key=lambda ele: ele[1])

        res = [obj_score[0] for obj_score in obj_scores]
        return res[:self._k]
