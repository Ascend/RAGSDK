# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os.path
from abc import ABC, abstractmethod

import networkx as nx
from networkx import DiGraph
from loguru import logger

from mx_rag.gvstore.graph_creator.lang import lang_dict, lang_zh
from mx_rag.gvstore.util.utils import GraphUpdatedData, safe_read_graphml
from mx_rag.utils.common import GRAPH_FILE_LIMIT, MAX_NODE_MUM, get_lang_param
from mx_rag.utils.file_check import SecFileCheck


class KGCoreError(Exception):
    pass


class GraphCore(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create_graph_index(self, **kwargs):
        pass

    def update_graph_index(self, updated_data: dict, **kwargs):
        pass

    @abstractmethod
    def search_indexes(self, query: str, k: int):
        pass

    @abstractmethod
    def get_sub_graph(self, ids: list, nodes: list, level: int):
        pass
    
    @abstractmethod
    def get_nodes(self, keywords):
        pass


class GraphNX(GraphCore):
    def __init__(self, graph_path: str, graph_name: str, **kwargs) -> None:
        self.vector_db = kwargs.get("vector_db", None)
        if self.vector_db is None:
            raise ValueError("input parameter value error: vector db for kg graph can't be none")
        SecFileCheck(graph_path, GRAPH_FILE_LIMIT).check()
        if "graph" in kwargs:
            if not isinstance(kwargs.get("graph"), DiGraph):
                raise KeyError("graph param error, it should be DiGraph")
            self.graph = kwargs.get("graph")
        else:
            self.graph = safe_read_graphml(graph_path)
        self.graph_path = graph_path
        self.graph_name = graph_name
        lang = get_lang_param(kwargs)
        self.lang_dict = lang_dict.get(lang, lang_zh)

    # 创建索引，索引数据包括：从文本中抽取出来的三元组实体信息，以及文本的原始文本块、文件层级信息等
    def create_graph_index(self, **kwargs):
        entity_filter_flag = kwargs.get("entity_filter_flag", False)
        if not entity_filter_flag:
            self.vector_db.create_index(self.graph)

    # 更新图谱数据
    def update_graph_index(self, updated_data: GraphUpdatedData, **kwargs):
        entity_filter_flag = kwargs.get("entity_filter_flag", False)
        if not entity_filter_flag:
            self.vector_db.update_index(updated_data)

    # 根据问题做向量相似性检索
    def search_indexes(self, query: str, k: int):
        # 此处partition_names为空，使用默认值，milvus是text，ascendfaiss是text和summary
        return self.vector_db.search_indexes(query, k, [])

    # 图多跳处理
    def get_sub_graph(self, ids, nodes, level):
        seeds, mem = set(ids), set(nodes)
        context_ids, relation_datas = set(), []
        while level > 0 and seeds:
            new_seeds = set()
            for seed_id in seeds:
                self._expand_edge(seed_id, context_ids, relation_datas, new_seeds, mem)
                if len(new_seeds) > MAX_NODE_MUM:
                    logger.warning(f"{level}-th hop: node count exceeds {MAX_NODE_MUM}, stopping traversal")
                    break
            if len(new_seeds) > MAX_NODE_MUM:
                break
            seeds, level = new_seeds, level - 1

        res = self._assemble_data(ids, list(context_ids), relation_datas)
        return res

    # 实体消歧时获取相实体节点
    def get_nodes(self, keywords):
        ids = []
        nodes = []
        data_list = self.vector_db.query_embedding(self.graph_name, keywords, partition_names=["entity"])
        for node_id, _, related_name, distance in data_list:
            if distance >= 0.95:
                ids.append(str(node_id))
                nodes.append(related_name)
        return ids, nodes

    # 组装图多跳后的结果，返回数据格式为：头节点 + 关系 + 尾节点
    def _assemble_data(self, ids: list, context_ids: list, relation_datas: list):
        id_list = []
        for src, tgt, _ in relation_datas:
            id_list.append(src)
            id_list.append(tgt)
        id_list.extend(context_ids)
        id_list.extend(ids)
        id_list = list(set(id_list))
        result_data = []
        ids, docs = self.vector_db.get_data(id_list)
        id_docs_dict = dict(zip(ids, docs))
        contents = [id_docs_dict[id] for id in context_ids if id in id_docs_dict]
        result_data.extend(contents)
        relations = [id_docs_dict[src] + ' ' + relation + ' ' + id_docs_dict[tgt]
                     for src, tgt, relation in relation_datas
                     if src in id_docs_dict and tgt in id_docs_dict]
        result_data.extend(relations)
        return list(set(result_data))

    # 根据embedding数据库相似性检索回来的topk做图的多跳遍历，基于广度优先算法
    def _expand_edge(self, seed_id: str, context_ids: set, relation_datas: list, new_seeds: set, mem: set):

        def _check_skip_label(u, v):
            return (self.graph.nodes[u]["label"] in ["file", "figure", "figure caption"]) \
                or (self.graph.nodes[v]["label"] in ["file", "figure", "figure caption"]) \
                or self.graph.nodes[v]["label"] != "text"

        def _filter_edge(u, v, data):
            nonlocal context_ids
            nonlocal relation_datas
            if "name" not in data or (data['name'] in [self.lang_dict["include_preceding_content"],
                                                       self.lang_dict["include_pic"],
                                                       self.lang_dict["include_table"],
                                                       self.lang_dict["include_following_content"],
                                                       self.lang_dict["summary"]]):
                return
            if data["name"] in [self.lang_dict["include_entity"]]:
                context_ids.add(u)
            elif data["name"] in [self.lang_dict["include_content"]]:
                if _check_skip_label(u, v):
                    return
                relation_datas.append((u, v, data['name']))
            else:
                return

        for u, v, data in self.graph.out_edges(seed_id, data=True):
            _filter_edge(u, v, data)
            if v not in mem:
                new_seeds.add(v)
                mem.add(v)

        for u, v, data in self.graph.in_edges(seed_id, data=True):
            _filter_edge(u, v, data)
            if u not in mem:
                new_seeds.add(u)
                mem.add(u)
