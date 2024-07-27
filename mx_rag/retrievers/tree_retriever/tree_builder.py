# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import copy
from typing import Dict, List, Optional, Set, Tuple, Callable

import numpy as np
from loguru import logger
from transformers import PreTrainedTokenizerBase

from .cluster_alg import _clustering
from .tree_structures import Node, Tree
from .utils import _get_node_list, _get_text
from ...chain import TreeText2TextChain


class TreeBuilderConfig:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase = None,
            max_tokens: int = 100,
            num_layers: int = 5,
            threshold: float = 0.1,
            reduction_dimension: int = 10,
            summarization_length: int = 100,
            summarization_model: TreeText2TextChain = None,
            max_length_in_cluster: int = 3500
    ):
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None.")
        self.tokenizer = tokenizer

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if not isinstance(reduction_dimension, int):
            raise ValueError("reduction_dimension must be an integer")
        self.reduction_dimension = reduction_dimension

        self.summarization_length = summarization_length

        if summarization_model is None:
            raise ValueError("summarization model must be defined")
        if not isinstance(summarization_model, TreeText2TextChain):
            raise ValueError(
                "summarization_model must be an instance of TreeText2TextChain"
            )
        self.summarization_model = summarization_model

        if not isinstance(max_length_in_cluster, int):
            raise ValueError("max_length_in_cluster must be an integer")
        self.max_length_in_cluster = max_length_in_cluster


class TreeBuilder:
    def __init__(self, config) -> None:
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.reduction_dimension = config.reduction_dimension
        self.threshold = config.threshold
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.max_length_in_cluster = config.max_length_in_cluster

    @staticmethod
    def create_node(
            index: int, text: str, embed_func: Callable[[List[str]], np.ndarray],
            children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        if children_indices is None:
            children_indices = set()
        # 使用传入的embed_func进行embedding, embed_func传入类型是List[str], 返回类型为np.ndarray
        embeddings = embed_func([text]).flatten()
        return index, Node(text, index, children_indices, embeddings)

    def build_from_text(self, embed_func: Callable[[List[str]], np.ndarray], chunks: List[str]) -> Tree:
        leaf_nodes = {}
        for index, text in enumerate(chunks):
            _, node = TreeBuilder.create_node(index, text, embed_func=embed_func)
            leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}
        logger.info(f"Layer 0, The number of leaf nodes is {len(leaf_nodes)}")
        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self._construct_tree(all_nodes, all_nodes, layer_to_nodes, embed_func)
        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)
        return tree

    def _summarize(self, context, max_tokens=100) -> Dict[str, str]:
        # 使用mxRAG调用llm的方式
        return self.summarization_model.summarize(context, max_tokens=max_tokens)

    def _process_cluster(self, cluster: List[Node], new_level_nodes: {}, next_node_index: int,
                         summarization_length: int, embed_func: Callable[[List[str]], np.ndarray]):
        node_texts = _get_text(cluster)
        summarized_result = self._summarize(
            context=node_texts,
            max_tokens=summarization_length,
        )
        summarized_text = summarized_result.get('result', '')
        _, new_parent_node = TreeBuilder.create_node(
            next_node_index, summarized_text, embed_func=embed_func, children_indices={node.index for node in cluster}
        )
        logger.info(f"Created node {next_node_index}")
        new_level_nodes[next_node_index] = new_parent_node

    def _construct_tree(
            self,
            current_level_nodes: Dict[int, Node],
            all_tree_nodes: Dict[int, Node],
            layer_to_nodes: Dict[int, List[Node]],
            embed_func: Callable[[List[str]], np.ndarray]
    ) -> Dict[int, Node]:
        next_node_index = len(all_tree_nodes)

        for layer in range(self.num_layers):
            new_level_nodes = {}
            node_list_current_layer = _get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                logger.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer + 1}"
                )
                self.num_layers = layer
                break

            clusters = _clustering(
                node_list_current_layer,
                reduction_dimension=self.reduction_dimension,
                tokenizer=self.tokenizer,
                threshold=self.threshold,
                max_length_in_cluster=self.max_length_in_cluster
            )

            for cluster in clusters:
                self._process_cluster(
                    cluster,
                    new_level_nodes,
                    next_node_index,
                    self.summarization_length,
                    embed_func
                )
                next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            logger.info(f"Layer {layer + 1}, The number of nodes is {len(current_level_nodes)}")
        logger.info(f"Create tree success, total layers {self.num_layers + 1}, "
                    f"total nodes {len(all_tree_nodes)}, the index start form 0")
        return current_level_nodes
