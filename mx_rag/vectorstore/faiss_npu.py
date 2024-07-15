# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

import os
from typing import List

import ascendfaiss
import faiss
import numpy as np
from loguru import logger

from mx_rag.utils import FileCheck
from mx_rag.vectorstore.vectorstore import VectorStore


class MindFAISSError(Exception):
    pass


class MindFAISS(VectorStore):
    INDEX_MAP = {
        "FLAT:L2": ascendfaiss.AscendIndexFlatL2,
    }

    def __init__(
            self,
            x_dim: int,
            index_type: str,
            devs: List[int],
            load_local_index: str = None,
            auto_save_path: str = None
    ):
        self.device = ascendfaiss.IntVector()
        if not isinstance(devs, list) or not devs:
            raise MindFAISSError("param devs need list type")
        if len(devs) != 1:
            raise MindFAISSError("currently only supports to set one device")
        for d in devs:
            self.device.push_back(d)
        self.auto_save_path = auto_save_path
        if load_local_index is not None:
            try:
                FileCheck.check_input_path_valid(load_local_index, check_blacklist=True)
                cpu_index = faiss.read_index(load_local_index)
                self.index = ascendfaiss.index_cpu_to_ascend(self.device, cpu_index)
            except Exception as err:
                raise MindFAISSError(f"load index failed, {err}") from err
            return
        try:
            config = ascendfaiss.AscendIndexFlatConfig(self.device)
            ascend_index = self.INDEX_MAP.get(index_type, None)
            if ascend_index is None:
                raise MindFAISSError(f"index type {ascend_index} not support")
            self.index = ascend_index(x_dim, config)
        except Exception as err:
            raise MindFAISSError(f"init index failed, {err}") from err

    @classmethod
    def load_local(cls, devs: List[int], index_path: str, auto_save_path: str = None) -> MindFAISS:
        return cls(0, "", devs, index_path, auto_save_path)

    def save_local(self, index_path: str) -> None:
        FileCheck.check_input_path_valid(index_path, check_blacklist=True)
        try:
            if os.path.exists(index_path):
                logger.warning(f"the index path {index_path} has already exist, will be overwritten")
                os.remove(index_path)

            cpu_index = ascendfaiss.index_ascend_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
            os.chmod(index_path, 0o600)
        except Exception as err:
            raise MindFAISSError(f"save index failed {err}") from err

    def delete(self, ids):
        res = self.index.remove_ids(np.array(ids))
        if self.auto_save_path is not None:
            self.save_local(self.auto_save_path)
        return res

    def search(self, embeddings: np.ndarray, k: int = 3):
        scores, indices = self.index.search(embeddings, k)
        return scores.tolist(), indices.tolist()

    def add(self, embeddings, ids):
        try:
            self.index.add_with_ids(embeddings, np.array(ids))
        except Exception as err:
            raise MindFAISSError(f"add index failed, {err}") from err
        if self.auto_save_path is not None:
            self.save_local(self.auto_save_path)
