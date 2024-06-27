# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Dict, List, Set

import numpy as np


class Node:
    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings


class Tree:
    def __init__(
            self, all_nodes: Dict[int, Node], root_nodes: Dict[int, Node],
            leaf_nodes: Dict[int, Node], num_layers: int, layer_to_nodes: Dict[int, List[Node]]
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes


def tree2dict(tree: Tree):
    return {
        "all_nodes": [{index: node2str(node)} for index, node in tree.all_nodes.items()],
        "root_nodes": [{index: node2str(node)} for index, node in tree.root_nodes.items()],
        "leaf_nodes": [{index: node2str(node)} for index, node in tree.leaf_nodes.items()],
        "num_layers": tree.num_layers,
        "layer_to_nodes": [{index: node_list2str(node_list)} for index, node_list in tree.layer_to_nodes.items()]
    }


def node2str(node: Node):
    return {
        "text": node.text,
        "index": node.index,
        "children": list(node.children),
        "embeddings": [float(item) if isinstance(item, (np.float16, np.float32)) else item for item in node.embeddings]
    }


def node_list2str(node_list: List[Node]):
    result = []
    for node in node_list:
        result.append(node2str(node))
    return result
