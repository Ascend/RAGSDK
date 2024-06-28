from unittest import TestCase
import numpy as np

from mx_rag.retrievers.tree_retriever.cluster_alg import global_cluster_embeddings, local_cluster_embeddings, \
    get_optimal_clusters, perform_clustering, clustering, gmm_cluster
from mx_rag.retrievers.tree_retriever.tree_structures import Node


class Test(TestCase):
    def test_global_cluster_embeddings(self):
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        dim = 3
        metric = "cosine"
        result = global_cluster_embeddings(embeddings, dim, metric=metric)
        self.assertEqual((5, 3), result.shape)

    def test_local_cluster_embeddings(self):
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        dim = 3
        metric = "cosine"
        result = local_cluster_embeddings(embeddings, dim, num_neighbors=3, metric=metric)
        self.assertEqual((5, 3), result.shape)

    def test_get_optimal_clusters(self):
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        result = get_optimal_clusters(embeddings)
        self.assertEqual(1, result)

    def test_perform_clustering(self):
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        dim = 3
        result = perform_clustering(embeddings, dim, 3)
        self.assertEqual(5, len(result))

    def test_clustering_none_tokenizer(self):
        node_list= [Node("test1", 0, {0}, {}),
        Node("test2", 1, {0}, {})]
        self.assertRaises(ValueError, clustering, node_list)

    def test_gmm_cluster(self):
        labels = gmm_cluster(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]), 0.2)
        self.assertEqual(5, len(labels[0]))