# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from typing import Dict, List, Any, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.vector_stores.faiss_vector_store import FaissVectorStore
from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.utils.common import validate_params


class ConceptCluster:
    """
    Clusters concepts based on embedding similarity using a vector store and a graph wrapper.
    """
    def __init__(self, vector_store: FaissVectorStore, graph: GraphStore) -> None:
        self.vector_store = vector_store
        self.graph = graph

    @staticmethod
    def _extract_concepts_and_embeddings(
        concept_embeddings: Dict[str, np.ndarray]
    ) -> Tuple[List[str], np.ndarray]:
        concept_names = list(concept_embeddings.keys())
        embeddings = np.array(list(concept_embeddings.values()), dtype=np.float32)
        return concept_names, embeddings

    @staticmethod
    def _build_edges(
        concept_names: List[str],
        distances: np.ndarray,
        indices: np.ndarray,
        threshold: float
    ) -> List[Tuple[str, str]]:
        edges = []
        for i in tqdm(range(len(indices)), desc="Building edges"):
            for j, neighbor_idx in enumerate(indices[i]):
                if distances[i][j] < threshold and concept_names[i] != concept_names[neighbor_idx]:
                    edges.append((concept_names[i], concept_names[neighbor_idx]))
        return edges

    @validate_params(
        concept_embeddings=dict(
            validator=lambda x: isinstance(x, dict), message="param must be a dict"),
        top_k=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 100,
                   message="param must be an integer, value range [1, 100]"),
        threshold=dict(validator=lambda x: isinstance(x, float) and 0.0 <= x <= 1.0,
                       message="param must be a float number, value range [0.0, 1.0]")
    )
    def find_clusters(
        self,
        concept_embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
        threshold: float = 0.15
    ) -> List[List[str]]:
        if not concept_embeddings:
            logger.warning("No concept embeddings provided.")
            return []

        concept_names, embeddings = self._extract_concepts_and_embeddings(concept_embeddings)
        self._prepare_vector_store(embeddings)
        distances, indices = self._search_similarities(embeddings, top_k)
        edges = self._build_edges(concept_names, distances, indices, threshold)
        self._update_graph(edges)
        clusters = self._extract_clusters()
        logger.info(f"Total connected components (clusters) found: {len(clusters)}")
        return clusters

    def _prepare_vector_store(self, embeddings: np.ndarray) -> None:
        self.vector_store.normalize_vectors_l2(embeddings)
        self.vector_store.add(embeddings, list(range(embeddings.shape[0])))

    def _search_similarities(
        self, embeddings: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.vector_store.search(embeddings, top_k)

    def _update_graph(self, edges: List[Tuple[str, str]]) -> None:
        self.graph.add_edges_from(edges)

    def _extract_clusters(self) -> List[List[str]]:
        connected_components = list(self.graph.connected_components())
        return sorted(connected_components, key=len, reverse=True)
