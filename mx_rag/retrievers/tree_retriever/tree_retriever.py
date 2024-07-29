# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List, Callable

import numpy as np
from loguru import logger
from transformers import PreTrainedTokenizerBase

from .tree_structures import Node, Tree
from .utils import (_distances_from_embeddings, _get_embeddings,
                    _get_node_list, _get_text,
                    _indices_of_nearest_neighbors_from_distances,
                    _reverse_mapping)


class TreeRetrieverConfig:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            embed_func: Callable[[List[str]], np.ndarray],
            threshold: float = 0.5,
            top_k: int = 5,
            selection_mode: str = "top_k",
            collapse_tree: bool = True,
            max_tokens: int = 3500
    ):
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None.")
        self.tokenizer = tokenizer

        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if embed_func is None:
            raise ValueError("embed_func must be a callable function")
        self.embed_func = embed_func

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")
        self.collapse_tree = collapse_tree
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens


class TreeRetriever:
    def __init__(self, config: TreeRetrieverConfig, tree: Tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        self.tree = tree
        self.num_layers = tree.num_layers + 1
        self.start_layer = tree.num_layers
        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.tree_node_index_to_layer = _reverse_mapping(self.tree.layer_to_nodes)
        self.embed_func = config.embed_func
        self.collapse_tree = config.collapse_tree
        self.max_tokens = config.max_tokens

    def retrieve(
            self,
            query: str
    ) -> str:
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if self.collapse_tree:
            selected_nodes, context = self._retrieve_information_collapse_tree(query, self.top_k, self.max_tokens)
        else:
            layer_nodes = self.tree.layer_to_nodes[self.start_layer]
            selected_nodes, context = self._retrieve_information(layer_nodes, query, self.num_layers)
        selected_nodes_index = [node.index for node in selected_nodes]
        logger.debug(f"after retrieve, the selected nodes index is {selected_nodes_index}")
        return context

    def _create_embedding(self, text: str) -> List[float]:
        embeddings = self.embed_func([text]).flatten().tolist()
        logger.debug(f"the create_embedding embeddings dim is {len(embeddings)}")
        return embeddings

    def _retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> tuple:
        query_embedding = self._create_embedding(query)
        selected_nodes = []
        node_list = _get_node_list(self.tree.all_nodes)
        embeddings = _get_embeddings(node_list)
        distances = _distances_from_embeddings(query_embedding, embeddings)
        indices = _indices_of_nearest_neighbors_from_distances(distances)
        total_tokens = 0
        for idx in indices[:top_k]:
            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))
            if total_tokens + node_tokens > max_tokens:
                break
            selected_nodes.append(node)
            total_tokens += node_tokens
        context = _get_text(selected_nodes)
        return selected_nodes, context

    def _retrieve_information(
            self, current_nodes: List[Node], query: str, num_layers: int
    ) -> (List[Node], str):
        query_embedding = self._create_embedding(query)
        selected_nodes = []
        node_list = current_nodes
        for layer in range(num_layers):
            embeddings = _get_embeddings(node_list)
            distances = _distances_from_embeddings(query_embedding, embeddings)
            indices = _indices_of_nearest_neighbors_from_distances(distances)
            if self.selection_mode == "threshold":
                best_indices = [index for index in indices if distances[index] > self.threshold]
            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]
            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = _get_text(selected_nodes)
        return selected_nodes, context
