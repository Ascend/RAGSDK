# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional

import ascendfaiss
import faiss
import numpy as np
from loguru import logger

from mx_rag.vectorstore.vectorstore import VectorStore


class MindFAISSError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)


class MindFAISS(VectorStore):
    DEVICES = None
    INDEX_MAP = {
        "FLAT:L2": ascendfaiss.AscendIndexFlatL2,
    }

    def __init__(
            self,
            x_dim: Optional[int],
            index_type: Optional[str],
            load_local_index: bool = False,
            auto_save_path: str = None,
            **kwargs
    ):

        if self.DEVICES is None:
            raise MindFAISSError("set devices first")
        self.auto_save_path = auto_save_path
        if load_local_index:
            return
        try:
            config = ascendfaiss.AscendIndexFlatConfig(self.DEVICES)
            ascend_index = self.INDEX_MAP.get(index_type, None)
            if ascend_index is None:
                raise MindFAISSError(f"index type {ascend_index} not support")
            self.index = ascend_index(x_dim, config)
        except Exception as err:
            raise MindFAISSError(f"init index failed, {err}") from err

    @classmethod
    def set_device(cls, dev: int):
        if cls.DEVICES is None:
            cls.DEVICES = ascendfaiss.IntVector()
        cls.DEVICES.push_back(dev)

    @classmethod
    def clear_device(cls):
        if cls.DEVICES is not None:
            cls.DEVICES.clear()

    @classmethod
    def load_local(cls, index_path: str, **kwargs) -> MindFAISS:
        if cls.DEVICES is None:
            raise MindFAISSError("set devices first")

        try:
            cpu_index = faiss.read_index(index_path)
            ascend_index = ascendfaiss.index_cpu_to_ascend(cls.DEVICES, cpu_index)
        except Exception as err:
            raise MindFAISSError(f"load index failed, {err}") from err
        index_obj = cls(None, None, load_local_index=True)
        index_obj.index = ascend_index
        return index_obj

    def save_local(self, index_path: str, **kwargs) -> None:
        if os.path.exists(index_path):
            logger.warning(f"the index path {index_path} has already exist, will be overwritten")
        try:
            cpu_index = ascendfaiss.index_ascend_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
            os.chmod(index_path, 0o600)
        except Exception as err:
            raise MindFAISSError(f"save index failed {err}") from err

    def delete(self, ids, *args, **kwargs):
        res = self.index.remove_ids(np.array(ids))
        if self.auto_save_path is not None:
            self.save_local(self.auto_save_path)
        return res

    def search(self, embeddings: np.ndarray, k: int = 3, *args, **kwargs):
        scores, indices = self.index.search(embeddings, k)
        return scores, indices.tolist()

    def add(self, embeddings, ids, *args, **kwargs):
        try:
            self.index.add_with_ids(embeddings, np.array(ids))
        except Exception as err:
            raise MindFAISSError(f"add index failed, {err}") from err
        if self.auto_save_path is not None:
            self.save_local(self.auto_save_path)
