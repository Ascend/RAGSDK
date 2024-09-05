# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

import os
import operator
from typing import List

import ascendfaiss
import faiss
import numpy as np
from loguru import logger

from mx_rag.utils.file_check import FileCheck
from mx_rag.storage.vectorstore.vectorstore import VectorStore, SimilarityStrategy
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, MAX_TOP_K


class MindFAISSError(Exception):
    pass


class MindFAISS(VectorStore):
    SIMILARITY_STRATEGY_MAP = {
        SimilarityStrategy.FLAT_IP:
            {
                "index": ascendfaiss.AscendIndexFlat,
                "metric": faiss.METRIC_INNER_PRODUCT,
                "scale": lambda x: x if x <= 1.0 else 1.0
            },
        SimilarityStrategy.FLAT_L2:
            {
                "index": ascendfaiss.AscendIndexFlat,
                "metric": faiss.METRIC_L2,
                "scale": lambda x: (1.0 - x / 2.0) if x <= 2.0 else 0.0
            },
        # 归一化之后的COS距离 等于IP距离, 所以参数一致
        SimilarityStrategy.FLAT_COS:
            {
                "index": ascendfaiss.AscendIndexFlat,
                "metric": faiss.METRIC_INNER_PRODUCT,
                "scale": lambda x: x if x <= 1.0 else 1.0
            }
    }

    @validate_params(
        x_dim=dict(validator=lambda x: isinstance(x, int) and 0 < x <= MAX_VEC_DIM),
        similarity_strategy=dict(
            validator=lambda x: isinstance(x, SimilarityStrategy) and x in MindFAISS.SIMILARITY_STRATEGY_MAP),
        auto_save=dict(validator=lambda x: isinstance(x, bool))
    )
    def __init__(
            self,
            x_dim: int,
            similarity_strategy: SimilarityStrategy,
            devs: List[int],
            load_local_index: str,
            auto_save: bool = False
    ):
        super().__init__()
        self.devs = devs
        self.device = ascendfaiss.IntVector()
        if not isinstance(devs, list) or not devs:
            raise MindFAISSError("param devs need list type")
        if len(devs) != 1:
            raise MindFAISSError("currently only supports to set one device")
        for d in devs:
            self.device.push_back(d)
        self.auto_save = auto_save
        self.load_local_index = load_local_index
        if os.path.exists(self.load_local_index):
            try:
                FileCheck.check_input_path_valid(self.load_local_index, check_blacklist=True)
                cpu_index = faiss.read_index(self.load_local_index)
                self.index = ascendfaiss.index_cpu_to_ascend(self.device, cpu_index)
            except Exception as err:
                raise MindFAISSError(f"load index failed, {err}") from err
            return
        try:
            config = ascendfaiss.AscendIndexFlatConfig(self.device)
            similarity = self.SIMILARITY_STRATEGY_MAP.get(similarity_strategy, None)
            if similarity is None:
                raise MindFAISSError(f"index type {similarity_strategy} not support")

            ascend_index_creator = similarity.get("index")
            ascend_index_metrics = similarity.get("metric")
            self.index = ascend_index_creator(x_dim, ascend_index_metrics, config)
            self.score_scale = similarity.get("scale", None)
        except Exception as err:
            raise MindFAISSError(f"init index failed, {err}") from err

    @staticmethod
    def create(**kwargs):
        if "x_dim" not in kwargs or not isinstance(kwargs.get("x_dim"), int):
            raise KeyError("x_dim param error. ")

        if "similarity_strategy" not in kwargs or not isinstance(kwargs.get("similarity_strategy"), SimilarityStrategy):
            raise KeyError("similarity_strategy param error. ")

        if "devs" not in kwargs or not isinstance(kwargs.get("devs"), List):
            raise KeyError("devs param error. ")

        if "load_local_index" not in kwargs or not isinstance(kwargs.get("load_local_index"), str):
            raise KeyError("load_local_index param error. ")

        return MindFAISS(**kwargs)

    def save_local(self) -> None:
        FileCheck.check_input_path_valid(self.load_local_index, check_blacklist=True)
        try:
            if os.path.exists(self.load_local_index):
                logger.warning(f"the index path {self.load_local_index} has already exist, will be overwritten")
                os.remove(self.load_local_index)

            cpu_index = ascendfaiss.index_ascend_to_cpu(self.index)
            faiss.write_index(cpu_index, self.load_local_index)
            os.chmod(self.load_local_index, 0o600)
        except Exception as err:
            raise MindFAISSError(f"save index failed {err}") from err

    def get_save_file(self):
        return self.load_local_index

    @validate_params(ids=dict(validator=lambda x: all(isinstance(it, int) for it in x)))
    def delete(self, ids: List[int]):
        res = self.index.remove_ids(np.array(ids))
        if self.auto_save:
            self.save_local()
        return res

    @validate_params(k=dict(validator=lambda x: 0 < x <= MAX_TOP_K))
    def search(self, embeddings: np.ndarray, k: int = 3):
        scores, indices = self.index.search(embeddings, k)
        return self._score_scale(scores.tolist()), indices.tolist()

    @validate_params(ids=dict(validator=lambda x: all(isinstance(it, int) for it in x)))
    def add(self, embeddings: np.ndarray, ids: List[int]):
        try:
            self.index.add_with_ids(embeddings, np.array(ids))
        except Exception as err:
            raise MindFAISSError(f"add index failed, {err}") from err
        if self.auto_save:
            self.save_local()

    def get_ntotal(self) -> int:
        return self.index.ntotal

    def get_all_ids(self) -> List[int]:
        ids = []
        for dev in self.devs:
            dev_ids = ascendfaiss.FaissIdxVector()
            self.index.getIdxMap(dev, dev_ids)
            ids.extend([dev_ids.at(i) for i in range(dev_ids.size())])
        return ids
