# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from langchain_core.documents import Document

from mx_rag.utils.common import (validate_params, validata_list_document, TEXT_MAX_LEN, STR_MAX_LEN, validata_list_str)


class Reranker(ABC):
    def __init__(self, k: int):
        self._k = k

    @abstractmethod
    def rerank(self,
               query: str,
               texts: List[str]):
        """ rank the texts and query"""

    @validate_params(
        scores=dict(validator=lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and 1 <= len(x) <= TEXT_MAX_LEN,
                    message="np.array length range [1, 1000 * 1000]")
    )
    def rerank_top_k(self,
                     objs: List,
                     scores: np.ndarray) -> List:
        check_objs_flag = False
        if len(objs) > 0:
            if isinstance(objs[0], str) and validata_list_str(objs, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]):
                check_objs_flag = True
            if isinstance(objs[0], Document) and validata_list_document(objs, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]):
                check_objs_flag = True
        if not check_objs_flag:
            raise ValueError(f"param objs of function rerank_top_k must meets: Type is List[Document] or List[str], "
                             f"list length range [1, {TEXT_MAX_LEN}], "
                             f"str length range [1, {STR_MAX_LEN}]")

        obj_scores = list(zip(objs, scores))
        obj_scores.sort(reverse=True, key=lambda ele: ele[1])

        res = [obj_score[0] for obj_score in obj_scores]
        return res[:self._k]
