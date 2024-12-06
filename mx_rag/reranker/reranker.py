# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from mx_rag.utils.common import (validate_params, validata_list_document, TEXT_MAX_LEN, STR_MAX_LEN)


class Reranker(ABC):
    def __init__(self, k: int):
        self._k = k

    @abstractmethod
    def rerank(self,
               query: str,
               texts: List[str]):
        """ rank the texts and query"""

    @validate_params(
        objs=dict(validator=lambda x: validata_list_document(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                  message="param must meets: Type is List[Document], list length range [1, 1000 * 1000], "
                           "pagecontent length range [1, 128 * 1024 * 1024]"),
        scores=dict(validator=lambda x: isinstance(x, np.ndarray) and x.dim == 1 and 1 <= len(x) <= TEXT_MAX_LEN,
                   message="np.array length range [1, 1000 * 1000]")
    )
    def rerank_top_k(self,
                     objs: List,
                     scores: np.ndarray) -> List:

        obj_scores = list(zip(objs, scores))
        obj_scores.sort(reverse=True, key=lambda ele: ele[1])

        res = [obj_score[0] for obj_score in obj_scores]
        return res[:self._k]
