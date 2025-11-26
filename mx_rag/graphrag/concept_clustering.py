#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from typing import Dict, List, Tuple, Set
import numpy as np
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.vector_stores.vector_store_wrapper import VectorStoreWrapper
from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.utils.common import validate_params


class ConceptCluster:
    """
    Clusters concepts based on embedding similarity using a vector store and a graph wrapper.
    """

    def __init__(self, vector_store: VectorStoreWrapper, graph: GraphStore) -> None:
        self.vector_store = vector_store
        self.graph = graph

    @staticmethod
    def _build_edges(
            concept_names: List[str],
            distances: np.ndarray,
            indices: np.ndarray,
            threshold: float,
    ) -> List[Tuple[str, str]]:
        edges = []
        index = 0
        for distance, indice in zip(distances, indices):
            for j, neighbor_idx in enumerate(indice):
                if distance[j] > threshold and concept_names[index] != concept_names[neighbor_idx]:
                    edges.append((concept_names[index], concept_names[neighbor_idx]))
            index += 1
        return edges

    @validate_params(
        concept_embeddings=dict(
            validator=lambda x: isinstance(x, dict), message="param must be a dict"
        ),
        top_k=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= 100,
            message="param must be an integer, value range [1, 100]",
        ),
        threshold=dict(
            validator=lambda x: isinstance(x, (float, int)) and 0.0 <= x <= 1.0,
            message="param must be float or int and value range [0.0, 1.0]",
        ),
    )
    def find_clusters(
            self,
            concept_embeddings: Dict[str, np.ndarray],
            top_k: int = 5,
            threshold: float = 0.5,
            batch_size: int = 4,
    ) -> List[Set[str]]:
        if not concept_embeddings:
            logger.warning("No concept embeddings provided.")
            return []

        concept_names = list(concept_embeddings.keys())
        embeddings = np.array(list(concept_embeddings.values()), dtype=np.float32)

        for start_index in range(0, embeddings.shape[0], batch_size):
            ids = list(range(start_index, min(start_index + batch_size, embeddings.shape[0])))
            self.vector_store.add(embeddings[start_index:start_index + batch_size], ids)

        for start_index in tqdm(range(0, embeddings.shape[0], batch_size), desc="Building edges"):
            distances, indices = self.vector_store.search(embeddings[start_index:start_index + batch_size], top_k)
            edges = self._build_edges(concept_names, distances, indices, threshold)
            self.graph.add_edges_from(edges)

        clusters = list(self.graph.connected_components())
        logger.info(f"Total connected components (clusters) found: {len(clusters)}")
        return clusters
