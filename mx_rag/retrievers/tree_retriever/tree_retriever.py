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
from ...utils.common import validate_params, MAX_TOP_K


class TreeRetrieverConfig:
    @validate_params(
        tokenizer=dict(validator=lambda x: isinstance(x, PreTrainedTokenizerBase)),
        embed_func=dict(validator=lambda x: isinstance(x, Callable)),
        threshold=dict(validator=lambda x: 0 <= x <= 1),
        top_k=dict(validator=lambda x: 1 <= x <= MAX_TOP_K),
        selection_mode=dict(validator=lambda x: x in ["top_k", "threshold"]),
        collapse_tree=dict(validator=lambda x: isinstance(x, bool)),
        max_tokens=dict(validator=lambda x: 50 <= x <= 10000)
    )
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            embed_func: Callable[[List[str]], List[List[float]]],
            threshold: float = 0.5,
            top_k: int = 5,
            selection_mode: str = "top_k",
            collapse_tree: bool = True,
            max_tokens: int = 3500
    ):
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.top_k = top_k
        self.selection_mode = selection_mode
        self.embed_func = embed_func
        self.collapse_tree = collapse_tree
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
        embeddings = self.embed_func([text])

        flat_list = []
        for row in embeddings:
            flat_list += row

        logger.debug(f"the create_embedding embeddings dim is {len(flat_list)}")
        return flat_list

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
