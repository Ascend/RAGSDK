# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import random
from typing import List, Optional

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

from .tree_structures import Node

RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def _global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def _local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def _get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return int(optimal_clusters)


def _gmm_cluster(embeddings: np.ndarray, threshold: float):
    n_clusters = _get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=0)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def _perform_clustering(
        embeddings: np.ndarray, dim: int, threshold: float,
) -> List[np.ndarray]:
    reduced_embeddings_global = _global_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
    global_clusters, n_global_clusters = _gmm_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = _local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = _gmm_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def _clustering(
        nodes: List[Node],
        max_length_in_cluster: int = 3500,
        tokenizer=None,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
) -> List[List[Node]]:
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None.")
    embeddings = np.array([node.embeddings for node in nodes])
    clusters = _perform_clustering(
        embeddings, dim=reduction_dimension, threshold=threshold
    )
    node_clusters = []

    for label in np.unique(np.concatenate(clusters)):
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]

        cluster_nodes = [nodes[i] for i in indices]

        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        total_length = sum(
            [len(tokenizer.encode(node.text)) for node in cluster_nodes]
        )

        if total_length > max_length_in_cluster:
            node_clusters.extend(
                _clustering(cluster_nodes, max_length_in_cluster, tokenizer, reduction_dimension, threshold)
            )
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters
