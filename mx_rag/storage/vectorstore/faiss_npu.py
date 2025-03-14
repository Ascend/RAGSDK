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

from mx_rag.utils.file_check import FileCheck, FileCheckError
from mx_rag.storage.vectorstore.vectorstore import VectorStore
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, MAX_TOP_K, BOOL_TYPE_CHECK_TIP, \
    MAX_PATH_LENGTH, STR_LENGTH_CHECK_1024


class MindFAISSError(Exception):
    pass


class MindFAISS(VectorStore):
    SCALE_MAP = {
        "IP": lambda x: min(x, 1.0),
        "L2": lambda x: max(1.0 - x / 2.0, 0.0),
        "COSINE": lambda x: min(x, 1.0)
    }

    METRIC_MAP = {
        "IP": faiss.METRIC_INNER_PRODUCT,
        "L2":  faiss.METRIC_L2,
        # 归一化之后的COS距离 等于IP距离, 所以参数一致
        "COSINE": faiss.METRIC_INNER_PRODUCT,
    }

    INDEX_MAP = {
        "FLAT": ascendfaiss.AscendIndexFlat,
    }

    @validate_params(
        x_dim=dict(validator=lambda x: isinstance(x, int) and 0 < x <= MAX_VEC_DIM,
                   message="param must be int and value range (0, 1024 * 1024]"),
        load_local_index=dict(
            validator=lambda x: isinstance(x, str) and len(x) <= MAX_PATH_LENGTH, message=STR_LENGTH_CHECK_1024),
        index_type=dict(validator=lambda x: isinstance(x, str) and x in ("FLAT",),
                        message="param must be str and in [FLAT]"),
        metric_type=dict(validator=lambda x: isinstance(x, str) and x in ("IP", "L2", "COSINE"),
                         message="param must be str and in [IP, L2, COSINE]"),
        auto_save=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
    )
    def __init__(
            self,
            x_dim: int,
            devs: List[int],
            load_local_index: str,
            index_type="FLAT",
            metric_type="L2",
            auto_save: bool = True
    ):
        super().__init__()
        self.index_type = index_type
        self.metric_type = metric_type
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
        FileCheck.check_input_path_valid(self.load_local_index, check_blacklist=True)
        FileCheck.check_filename_valid(self.load_local_index)

        self.score_scale = self.SCALE_MAP.get(self.metric_type)

        if os.path.exists(self.load_local_index):
            logger.info(f"Loading index from local index file: '{self.load_local_index}'")
            try:
                cpu_index = faiss.read_index(self.load_local_index)
                self.index = ascendfaiss.index_cpu_to_ascend(self.device, cpu_index)

            except FileCheckError as fc_error:
                logger.error(f"Invalid local index file: {fc_error}")
                raise MindFAISSError(f"Failed to load index: {fc_error}") from fc_error
            except Exception as err:
                logger.error(f"Unexpected error during index loading: {err}")
                raise MindFAISSError(f"Failed to load index: {err}") from err
            return  # 成功加载本地索引
        try:
            ascend_index_creator = self.INDEX_MAP.get(self.index_type)
            ascend_index_metrics = self.METRIC_MAP.get(self.metric_type)
            config = ascendfaiss.AscendIndexFlatConfig(self.device)
            # 根据提供的策略创建索引
            self.index = ascend_index_creator(x_dim, ascend_index_metrics, config)

        except Exception as err:
            raise MindFAISSError(f"init index failed, {err}") from err

    @staticmethod
    def create(**kwargs):
        if "x_dim" not in kwargs or not isinstance(kwargs.get("x_dim"), int):
            raise KeyError("x_dim param error. ")

        if "devs" not in kwargs or not isinstance(kwargs.get("devs"), List):
            raise KeyError("devs param error. ")

        if "load_local_index" not in kwargs or not isinstance(kwargs.get("load_local_index"), str):
            raise KeyError("load_local_index param error. ")

        return MindFAISS(**kwargs)

    def add_sparse(self, ids, sparse_embeddings):
        raise NotImplementedError

    def add_dense_and_sparse(self, ids, dense_embeddings, sparse_embeddings):
        raise NotImplementedError

    def save_local(self) -> None:
        FileCheck.check_input_path_valid(self.load_local_index, check_blacklist=True)
        FileCheck.check_filename_valid(self.load_local_index)
        try:
            if os.path.exists(self.load_local_index):
                logger.warning(f"the index path '{self.load_local_index}' has already exist, will be overwritten")
                os.remove(self.load_local_index)

            cpu_index = ascendfaiss.index_ascend_to_cpu(self.index)
            faiss.write_index(cpu_index, self.load_local_index)
            os.chmod(self.load_local_index, 0o600)
        except OSError as os_error:
            logger.error(f"File system error during saving index to '{self.load_local_index}': {os_error}")
            raise MindFAISSError(f"File system error: {os_error}") from os_error
        except Exception as err:
            logger.error(f"Unexpected error during index saving: {err}")
            raise MindFAISSError(f"Failed to save index due to an unexpected error: {err}") from err

    def get_save_file(self):
        return self.load_local_index

    @validate_params(ids=dict(validator=lambda x: all(isinstance(it, int) for it in x),
                              message="param must be List[int]"))
    def delete(self, ids: List[int]):
        if len(ids) >= self.MAX_VEC_NUM:
            raise MindFAISSError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        res = self.index.remove_ids(np.array(ids))
        logger.debug(f"success remove {len(ids)} ids in MindFAISS.")
        if self.auto_save:
            self.save_local()
        return res

    @validate_params(
        k=dict(validator=lambda x: 0 < x <= MAX_TOP_K, message="param value range (0, 10000]"),
        embeddings=dict(validator=lambda x: isinstance(x, np.ndarray), message="embeddings must be np.ndarray type"))
    def search(self, embeddings: np.ndarray, k: int = 3):
        if len(embeddings.shape) != 2:
            raise MindFAISSError("shape of embedding must equal to 2")
        if embeddings.shape[0] >= self.MAX_SEARCH_BATCH:
            raise MindFAISSError(f"num of embeddings must less {self.MAX_SEARCH_BATCH}")
        scores, indices = self.index.search(embeddings, k)
        return self._score_scale(scores.tolist()), indices.tolist()

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]"),
        embeddings=dict(validator=lambda x: isinstance(x, np.ndarray), message="embeddings must be np.ndarray type"))
    def add(self, embeddings: np.ndarray, ids: List[int]):
        if len(embeddings.shape) != 2:
            raise MindFAISSError("shape of embedding must equal to 2")
        if embeddings.shape[0] != len(ids):
            raise MindFAISSError("Length of embeddings is not equal to number of ids")
        if len(ids) + self.index.ntotal >= self.MAX_VEC_NUM:
            raise MindFAISSError(f"total num of ids/embedding is reach to limit {self.MAX_VEC_NUM}")
        try:
            self.index.add_with_ids(embeddings, np.array(ids))
            logger.debug(f"success add {len(ids)} ids in MindFAISS.")
        except (AttributeError, AssertionError) as e:
            logger.error(f"Failed to add index due to an attribute or assertion error: {e}")
            raise MindFAISSError(f"Failed to add index: {e}") from e
        except Exception as err:
            logger.error(f"Unexpected error while adding index: {err}")
            raise MindFAISSError(f"Failed to add index: {err}") from err
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
