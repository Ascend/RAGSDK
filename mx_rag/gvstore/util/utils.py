# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import re
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

from mx_rag.utils.common import MAX_FILE_SIZE, TEXT_MAX_LEN
from mx_rag.utils.file_check import SecFileCheck

MAX_NAME_LENTH = 1024
LABEL_TYPE = ["text", "entity", "summary", "file"]


class KgOprMode(Enum):
    NEW = 1


@dataclass
class GraphUpdatedData:
    added_nodes: list = field(default_factory=list)
    added_edges: list = field(default_factory=list)
    deleted_nodes: list = field(default_factory=list)
    deleted_edges: list = field(default_factory=list)


def check_graph_name(graph_name: str):
    if not (isinstance(graph_name, str) and 0 < len(graph_name) <= MAX_NAME_LENTH):
        return False
    pattern = r'^[a-zA-Z0-9_]+$'
    return bool(re.match(pattern, graph_name))


def check_entity_validity(entity_name: str):
    illegal_strs = ['\\', '/', ';', '\n', '$$', 'AND', 'OR', 'WHERE', 'CREATE', 'DELETE', 'ALTER', 'INSERT', 'UPDATE',
                    '\"', '\'']
    for illegal_str in illegal_strs:
        if illegal_str.lower() in entity_name.lower():
            return False
    return True


def filter_graph_node(node_id, label, info: str):
    if not (str(node_id).isdigit() and label in LABEL_TYPE):
        return None, None, None
    # 不能以\结尾
    while info.endswith("\\"):
        info = info[:-1]
    info = info.replace('\n', '\\n').replace('"', ' ')
    return str(node_id), label, info


def filter_graph_edge(src, dst, name: str):
    # src和dst读取至文件则是字符串类型，更新时来自变量是int类型
    if not (str(src).isdigit() and str(dst).isdigit()):
        return None, None, None
    # 不能以\结尾
    while name.endswith("\\"):
        name = name[:-1]
    name = name.replace('\n', '\\n').replace('"', ' ')
    return str(src), str(dst), name


def safe_read_graphml(path):
    try:
        SecFileCheck(path, MAX_FILE_SIZE).check()
        graph = nx.read_graphml(path)
        for _, data in graph.nodes.data():
            if not (isinstance(data['id'], int) and data['label'] in LABEL_TYPE and len(data['info']) <= TEXT_MAX_LEN):
                raise Exception(f"the node'id must be int, label must in {LABEL_TYPE},"
                                f" and info length must less than {TEXT_MAX_LEN}.")
        for data in graph.edges.data():
            src, dst, label = data
            relation = label["name"]
            if not (src.isdigit() and dst.isdigit() and len(relation) <= MAX_NAME_LENTH):
                raise Exception(f"the edge's src and dst must be int,"
                                f" and relation length must less than {MAX_NAME_LENTH}")
        return graph
    except Exception as e:
        raise Exception("Failed to parse graphml file.") from e
