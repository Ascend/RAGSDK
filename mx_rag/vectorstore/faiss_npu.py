# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations
import inspect
import os
from pathlib import Path
from typing import Optional, List, Any, Sized, Tuple, Callable, NoReturn

import ascendfaiss
import faiss
import numpy as np

from mx_rag.storage import Document, Docstore


class MindFAISSError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)


class MindFAISS:
    DEVICES = None
    INDEX_MAP = {
        "FLAT:L2": ascendfaiss.AscendIndexFlatL2,
    }

    def __init__(
            self,
            x_dim: Optional[int],
            index_type: Optional[str],
            document_store: Docstore,
            embed_func: Callable[[List[str]], np.ndarray],
            load_local_index: bool = False,
            **kwargs
    ):

        if self.DEVICES is None:
            raise MindFAISSError("set devices first")
        if not isinstance(document_store, Docstore):
            raise MindFAISSError("invalid document store type")
        self.document_store = document_store
        if not callable(embed_func):
            raise MindFAISSError("embed_func need callable")
        self.embed_func = embed_func

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

    @staticmethod
    def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
        if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
            raise MindFAISSError(f"{x_name} and {y_name} expected to be equal length")

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
    def load_local(cls, index_path: str, document_store: Docstore, embed_func: Callable, **kwargs) -> MindFAISS:
        if cls.DEVICES is None:
            raise MindFAISSError("set devices first")

        try:
            cpu_index = faiss.read_index(index_path)
            ascend_index = ascendfaiss.index_cpu_to_ascend(cls.DEVICES, cpu_index)
        except Exception as err:
            raise MindFAISSError(f"load index failed, {err}") from err
        index_obj = cls(None, None, document_store=document_store,
                        embed_func=embed_func, load_local_index=True)
        index_obj.index = ascend_index
        return index_obj

    def save_local(self, index_path: str, **kwargs) -> None:
        if os.path.exists(index_path):
            raise MindFAISSError(f"the index path {index_path} has already exist, please remove it first.")
        try:
            cpu_index = ascendfaiss.index_ascend_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
            os.chmod(index_path, 0o600)
        except Exception as err:
            raise MindFAISSError(f"save index failed {err}") from err

    def add_texts(
            self,
            doc_name: str,
            texts: List[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> NoReturn:
        embeddings = self.embed_func(texts)
        if not isinstance(embeddings, np.ndarray):
            raise MindFAISSError("The data type of embedding should be np.ndarray")
        self._add_index(doc_name, texts, embeddings, metadatas=metadatas, )

    def delete(self, doc_name: str, **kwargs: Any):
        try:
            ids = self.document_store.delete(doc_name)
            num_removed = self.index.remove_ids(np.array(ids))
        except Exception as err:
            raise MindFAISSError(f"delete index failed {err}") from err
        if len(ids) != num_removed:
            raise MindFAISSError(f"the number of documents does not match the number of vectors")

    def similarity_search(
            self,
            query: List[str],
            k: int = 4,
            **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        embeddings = self.embed_func(query)
        if not isinstance(embeddings, np.ndarray):
            raise MindFAISSError("The data type of embedding should be np.ndarray")
        return self.similarity_search_by_vector(embeddings, k, **kwargs)

    def similarity_search_by_vector(
            self,
            embeddings: np.ndarray,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if not isinstance(embeddings, np.ndarray):
            raise MindFAISSError("The data type of embedding should be np.ndarray")
        docs = []
        try:
            scores, indices = self.index.search(embeddings, k)
            for i, idx in enumerate(indices[0]):
                doc = self.document_store.search(idx.item())
                if doc is None:
                    continue
                docs.append((doc, scores[0][i]))
            return docs[:k]
        except Exception as err:
            MindFAISSError(f"index search failed: {err} ")
            return []

    def _add_index(self, doc_name, texts, embeddings, metadatas, **kwargs):
        metadatas = metadatas or ({} for _ in texts)
        self._len_check_if_sized(texts, metadatas, "texts", "metadatas")
        documents = [Document(page_content=t, metadata=m, document_name=doc_name) for t, m in zip(texts, metadatas)]
        self._len_check_if_sized(documents, embeddings, "documents", "embeddings")

        try:
            idxs = self.document_store.add(documents)
            self.index.add_with_ids(embeddings, np.array(idxs))
        except Exception as err:
            raise MindFAISSError(f"add index failed, {err}") from err
