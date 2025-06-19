# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Optional, List
import os
import faiss
import numpy as np

from mx_rag.storage.document_store.base_storage import StorageError
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.utils.file_check import FileCheck, check_disk_free_space


class FaissVectorStore:
    """
    A wrapper class for MindFAISS and FAISS vector index operations.
    """
    FREE_SPACE_LIMIT = 10 * 1024 * 1024 * 1024  # 10GB

    def __init__(self, dimension: int, index_path: str, **kwargs) -> None:
        """
        Initialize the FaissVectorStore.

        Args:
            dimension (int): The dimension of the vectors.
            index_path (str): Path to the FAISS index file.
            **kwargs: Additional keyword arguments for MindFAISS.
        """
        self.index_path = index_path
        self.index_type = kwargs.get("index_type", "FLAT")
        self.metric_type = kwargs.get("metric_type", "IP")
        self.auto_save = kwargs.get("auto_save", True)
        devs = kwargs.get("devs", [0])

        FileCheck.check_input_path_valid(self.index_path, check_blacklist=True)
        FileCheck.check_filename_valid(self.index_path)

        if self.index_type == "IndexHNSWFlat":
            m = kwargs.get("M", 16)
            ef_construction = kwargs.get("efConstruction", 100)
            ef_search = kwargs.get("efSearch", 50)
            self.index = (
                faiss.read_index(index_path)
                if os.path.exists(index_path)
                else faiss.IndexHNSWFlat(dimension, m)
            )
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            self.index = MindFAISS(
                x_dim=dimension,
                devs=devs,
                load_local_index=index_path,
                index_type=self.index_type,
                metric_type=self.metric_type,
                auto_save=self.auto_save,
            )

    @staticmethod
    def normalize_vectors_l2(vectors: np.ndarray) -> None:
        """
        Normalize vectors to unit length using L2 norm.

        Args:
            vectors (np.ndarray): The vectors to normalize.
        """
        faiss.normalize_L2(vectors)

    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """
        Add vectors to the index.

        Args:
            vectors (np.ndarray): The vectors to add.
            ids (List[int]): IDs corresponding to the vectors.
        """
        self.index.add(vectors) if self.index_type == "IndexHNSWFlat" else self.index.add(ids, vectors)

    def search(self, query_vectors: np.ndarray, top_k: int):
        """
        Search for the top_k most similar vectors.

        Args:
            query_vectors (np.ndarray): The query vectors.
            top_k (int): Number of top results to return.

        Returns:
            tuple: Distances and indices of the top_k results.
        """
        if self.index_type == "IndexHNSWFlat":
            return self.index.search(query_vectors, top_k)
        return self.index.search(query_vectors.tolist(), top_k, None)

    def ntotal(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            int: Number of vectors.
        """
        return self.index.ntotal if self.index_type == "IndexHNSWFlat" else self.index.get_ntotal()

    def clear(self) -> None:
        """
        Remove all vectors from the index.
        """
        self.index.reset() if self.index_type == "IndexHNSWFlat" else self.index.index.reset()

    def save(self) -> None:
        """
        Save the index to disk.
        """
        dirname = os.path.dirname(self.index_path)
        if check_disk_free_space(dirname if dirname else "./", self.FREE_SPACE_LIMIT):
            raise StorageError("Insufficient remaining space, please clear disk space")
        if self.index_type == "IndexHNSWFlat":
            faiss.write_index(self.index, self.index_path)
        else:
            self.index.save_local()
