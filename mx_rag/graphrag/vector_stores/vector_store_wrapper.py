#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Optional, List
import numpy as np

from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.vectorstore.vectorstore import VectorStore
from mx_rag.utils.common import validate_params


class VectorStoreWrapper:
    """
    A wrapper class for VectorStore.
    """
    @validate_params(
        vector_store=dict(
            validator=lambda x: isinstance(x, VectorStore),
            message="param must be an instance of VectorStore"
        )
    )
    def __init__(self, vector_store: VectorStore) -> None:
        """
        Initialize the VectorStoreWrapper.

        Args:
            vector_store (VectorStore): The vector store.
        """
        self.vector_store = vector_store

    @staticmethod
    def normalize_vectors_l2(vectors: np.ndarray) -> None:
        """
        Normalize vectors to unit length using L2 norm in place.

        Args:
            vectors (np.ndarray): The vectors to normalize.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors /= norms

    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """
        Add vectors to the index.

        Args:
            vectors (np.ndarray): The vectors to add.
            ids (List[int]): IDs corresponding to the vectors.
        """
        self.vector_store.add(ids, vectors)

    def search(self, query_vectors: np.ndarray, top_k: int):
        """
        Search for the top_k most similar vectors.

        Args:
            query_vectors (np.ndarray): The query vectors.
            top_k (int): Number of top results to return.

        Returns:
            tuple: Distances and indices of the top_k results.
        """
        # distances, indices
        return self.vector_store.search(query_vectors.tolist(), top_k)[:2]

    def ntotal(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            int: Number of vectors.
        """
        if isinstance(self.vector_store, MindFAISS):
            return self.vector_store.get_ntotal()
        return len(self.vector_store.get_all_ids())

    def clear(self) -> None:
        """
        Remove all vectors from the index.
        """
        self.vector_store.delete(self.vector_store.get_all_ids())

    def save(self) -> None:
        """
        Save the index to disk.
        """
        # only MindFAISS needs to call save_local
        if isinstance(self.vector_store, MindFAISS):
            self.vector_store.save_local()
