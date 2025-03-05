# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import time

import networkx as nx
from loguru import logger
from nebula3.gclient.net import Session

from mx_rag.gvstore.graph_creator.graph_core import GraphCore
from mx_rag.gvstore.util.utils import GraphUpdatedData
from mx_rag.utils.common import GRAPH_FILE_LIMIT
from mx_rag.utils.file_check import SecFileCheck


class NebulaGraphErr(Exception):
    pass


class NebulaGraph(GraphCore):
    def __init__(self, graph_path: str, graph_name: str, **kwargs):
        super().__init__()
        self.session = kwargs.get("session", None)
        if not isinstance(self.session, Session):
            raise ValueError("input parameter value error: nebula_session can't be none and must be type Session")
        SecFileCheck(graph_path, GRAPH_FILE_LIMIT).check()
        
        self.graph_name = graph_name
        self.graph_path = graph_path
        self.vector_db = kwargs.get("vector_db", None)
        if self.vector_db is None:
            raise ValueError("input parameter value error: vector db for kg graph can't be none")
        self.lang = kwargs.get("lang", "zh")

    @staticmethod
    def result_to_list(result):
        if not result.is_succeeded():
            raise NebulaGraphErr("Nebula excute result failed")
        re_info = []
        for row_num in range(result.row_size()):
            row = result.row_values(row_num)
            if len(row) < 6:
                continue
            if row[0].as_int() < 0:
                re_info.append(row[3].as_string())
            else:
                re_info.append(row[1].as_string())
        return re_info

    def create_graph_index(self, **kwargs):
        graph = nx.read_graphml(self.graph_path)
        entity_filter_flag = kwargs.get("entity_filter_flag", False)
        if not entity_filter_flag:
            self.vector_db.create_index(graph)
        self.init_space(self.graph_name)
        nodes = []
        edges = []
        for _, data in graph.nodes.data():
            if "id" in data and "info" in data and "label" in data:
                nodes.append(data)
        for data in graph.edges.data():
            edge = {}
            edge['src'], edge['dst'], label = data
            edge["name"] = label["name"]
            edges.append(edge)

        self.add_nodes_and_edges(nodes, edges)

    def update_graph_index(self, updated_data: GraphUpdatedData, **kwargs):
        # currently only process data added scenario
        entity_filter_flag = kwargs.get("entity_filter_flag", False)
        if not entity_filter_flag:
            self.vector_db.update_index(updated_data)
        added_nodes = updated_data.added_nodes
        added_edges = updated_data.added_edges
        self.use_space(self.graph_name)
        self.add_nodes_and_edges(added_nodes, added_edges)

    def search_indexes(self, query, k, **kwargs):
        return self.vector_db.search_indexes(query, k, **kwargs)

    def get_sub_graph(self, ids: list, nodes: list, level: int, **kwargs):
        if not nodes:
            return []
        self.use_space(self.graph_name)
        edge_result = []
        for node in ids:
            edges = self.get_khop_edges(node, level)
            edge_result.extend(edges)
        return list(set(edge_result))

    def get_nodes(self, keywords, **kwargs):
        self.use_space(self.graph_name)
        return self.search_nodes(keywords, "info")

    def execute(self, command):
        if self.session.ping_session():
            execute_res = self.session.execute(command)
        else:
            raise NebulaGraphErr("Nebula session timeout")
        if not execute_res.is_succeeded():
            raise NebulaGraphErr("Nebula execute failed")
        return execute_res

    def use_space(self, space_name):
        result = self.execute("USE {};".format(space_name))
        logger.info(f"Nebula use space result: {result.is_succeeded()}")

    def create_tag(self, props_info):
        return self.execute('CREATE TAG entity({})'.format(props_info))

    def create_edge(self, props_info):
        return self.execute('CREATE EDGE relation({})'.format(props_info))

    def create_index(self):
        return self.execute(
            'CREATE TAG INDEX IF NOT EXISTS entity_attri_index on entity(label(10), info(50), concept(10))')

    def rebuild_index(self):
        return self.execute('REBUILD TAG INDEX entity_attri_index')

    def init_space_simple(self):
        _ = self.create_tag('label string, info string, concept string')
        time.sleep(10)
        _ = self.create_edge('name string')

    def init_space(self, space_name, id_max_lens=25):
        create_space_res = self.execute(
            "CREATE SPACE IF NOT EXISTS {} (vid_type=FIXED_STRING({}));".format(space_name, str(id_max_lens)))
        logger.info(f"create_space_res: {create_space_res.is_succeeded()}")
        clear_space_res = self.clear_space(space_name)
        logger.info(f"clear_space_res: {clear_space_res.is_succeeded()}")
        self.session.execute_py(
            "USE {};"
            "CREATE TAG IF NOT EXISTS entity(label string, info string, concept string);"
            "CREATE EDGE IF NOT EXISTS relation(name string);".format(space_name)
        )
        create_index_res = self.create_index()
        logger.info(f"create_index_res: {create_index_res.is_succeeded()} ")
        time.sleep(20)

    def clear_space(self, space_name):
        self.use_space(self.graph_name)
        return self.execute("CLEAR SPACE IF EXISTS {};".format(space_name))

    def get_khop_edges(self, source, k):
        edge_property = "包含实体" if self.lang == "zh" else "include entity"
        result = self.execute(
            'GO 1 TO {} STEPS FROM "{}" OVER relation BIDIRECT '
            'WHERE properties(edge).name == "{}" '
            'and (properties($$).label == "text" or properties($^).label == "text")'
            'YIELD DISTINCT relation._type AS dir, properties($^).info AS src, properties(edge).name AS rel, '
            'properties($$).info AS dst, properties($^).label AS src_label,  properties($$).label AS dst_label'
            .format(k, source, edge_property))
        result_info = self.result_to_list(result)
        return list(set(result_info))

    def search_nodes(self, labels, entity_type):
        nodes_label = set()
        all_nodes_info = []
        all_nodes_id = []
        for label in labels:
            # 关键词检索node时不区分大小写
            result = self.execute('MATCH (v:entity) WHERE LOWER(v.entity.' + f'{entity_type}' + ') == LOWER('
                                  + f'\"{label}\"' + ') RETURN v')
            for row_num in range(result.row_size()):
                node = result.row_values(row_num)[0].as_node()
                tag = node.tags()[0]
                node_id = node.get_id().as_string()
                info = {k: v for k, v in node.properties(tag).items()}
                node_info = info['info'].as_string()
                nodes_label = nodes_label | set([info['label'].as_string()])
                all_nodes_id.append(node_id)
                all_nodes_info.append(node_info)
        return all_nodes_id, all_nodes_info

    def add_nodes_and_edges(self, nodes, edges):
        for node in nodes:
            try:
                if node['info'].endswith('\\'):
                    node['info'] = node['info'][:-1]
                self.execute('INSERT VERTEX entity(label, info) VALUES "{}":("{}", "{}");'
                             .format(node['id'], node['label'], node['info'].replace('\n', '\\n')
                                     .replace('"', ' ').replace('“', ' ').replace('”', ' ')))
            except Exception as e:
                logger.error(f"Nebula插入节点失败: {e}")

        for edge in edges:
            try:
                self.execute('INSERT EDGE relation(name) VALUES "{}"->"{}":("{}");'
                             .format(edge['src'], edge['dst'], edge['name']))
            except Exception as e:
                logger.error(f"Nebula插入边失败: {e}")
