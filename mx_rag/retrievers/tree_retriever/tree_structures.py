# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Dict, List, Set

import numpy as np

from mx_rag.utils.common import validate_params, NODE_MAX_TEXT_LENGTH, TREE_MAX_NODES, TREE_MAX_LAYERS


class Node:
    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= NODE_MAX_TEXT_LENGTH),
        index=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= TREE_MAX_NODES),
        children=dict(validator=lambda x: isinstance(x, set) and 0 <= len(x) <= TREE_MAX_NODES),
    )
    def __init__(self, text: str, index: int, children: Set[int], embeddings: np.ndarray) -> None:
        if not (isinstance(embeddings, np.ndarray)):
            raise ValueError("embeddings can only be a numpy array")
        if embeddings.ndim != 1 or embeddings.dtype not in [np.float16, np.float32]:
            raise ValueError(f"embeddings must be a 1D array of floats: {type(embeddings)} {embeddings.dtype}")
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings


class Tree:
    @validate_params(
        all_nodes=dict(validator=lambda x: isinstance(x, dict) and len(x) <= TREE_MAX_NODES),
        root_nodes=dict(validator=lambda x: isinstance(x, dict) and len(x) <= TREE_MAX_NODES),
        leaf_nodes=dict(validator=lambda x: isinstance(x, dict) and len(x) <= TREE_MAX_NODES),
        num_layers=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= TREE_MAX_LAYERS),
        layer_to_nodes=dict(validator=lambda x: isinstance(x, dict) and len(x) <= TREE_MAX_NODES),
    )
    def __init__(
            self, all_nodes: Dict[int, Node], root_nodes: Dict[int, Node],
            leaf_nodes: Dict[int, Node], num_layers: int, layer_to_nodes: Dict[int, List[Node]]
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes


def _tree2dict(tree: Tree):
    return {
        "all_nodes": [{index: _node2str(node)} for index, node in tree.all_nodes.items()],
        "root_nodes": [{index: _node2str(node)} for index, node in tree.root_nodes.items()],
        "leaf_nodes": [{index: _node2str(node)} for index, node in tree.leaf_nodes.items()],
        "num_layers": tree.num_layers,
        "layer_to_nodes": [{index: _node_list2str(node_list)} for index, node_list in tree.layer_to_nodes.items()]
    }


def _node2str(node: Node):
    return {
        "text": node.text,
        "index": node.index,
        "children": list(node.children),
        "embeddings": [float(item) if isinstance(item, (np.float16, np.float32)) else item for item in node.embeddings]
    }


def _node_list2str(node_list: List[Node]):
    result = []
    for node in node_list:
        result.append(_node2str(node))
    return result
