# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re
from abc import ABC
from collections import defaultdict
import concurrent.futures
from concurrent.futures import as_completed
from loguru import logger
import networkx as nx
from networkx import DiGraph
from tqdm import tqdm

from mx_rag.gvstore.graph_creator.entity_filter import EntityFilter
from mx_rag.gvstore.graph_creator.lang import lang_dict, lang_zh
from mx_rag.gvstore.graph_creator.llm_extract import Extractor
from mx_rag.gvstore.util.utils import KgOprMode, GraphUpdatedData, safe_read_graphml
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params, get_lang_param

ENTITY = "entity"
TEXT = "text"
TABLE = "table"
FIGURE = "figure"
SUMMARY = "summary"
FILE = "file"


class GraphCreation:

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        graph=dict(validator=lambda x: x is None or isinstance(x, DiGraph),
                   message="graph param must be None or instance of DiGraph"),
        current_id=dict(validator=lambda x: x is None or isinstance(x, int),
                        message="current_id param must be None or int")
    )
    def __init__(self, llm: Text2TextLLM,
                 graph: DiGraph = None,
                 current_id: int = None,
                 entity_types=None,
                 **kwargs
                 ) -> None:
        if entity_types is None:
            entity_types = []
        param1_invalid = graph is None and current_id is not None
        param2_invalid = graph is not None and current_id is None
        if param1_invalid or param2_invalid:
            raise ValueError("input parameter value error: graph or current_id invalid")
        thread_num = kwargs.get("thread_num", 8)
        if not (isinstance(thread_num, int) and 0 < thread_num <= 20):
            raise KeyError("thread_num param error, it should be integer type, and range in [1, 20]")
        self.lang = get_lang_param(kwargs)
        self.llm = llm
        # 存储networkx图
        if graph is not None:
            self.graph = graph
        else:
            self.graph = DiGraph()
        # 建图时id为current_id+1
        if current_id is not None:
            self.current_id = current_id
        else:
            self.current_id = -1

        # 图信息存储
        self.node_id_map = {}

        self.file_contents = {}

        # info抽取类
        self.extractor = Extractor(llm=self.llm, entity_types=entity_types, **kwargs)
        # info抽取结果存储
        self.entity_id_map = {}
        self.extract_workers = concurrent.futures.ThreadPoolExecutor(max_workers=thread_num,
                                                                     thread_name_prefix="extract_workers")

    @staticmethod
    def _diff_graph(opr_mode: KgOprMode, old_graph: DiGraph, new_graph: DiGraph):
        update_data = GraphUpdatedData()
        if opr_mode == KgOprMode.NEW:
            new_nodes = set(new_graph.nodes()) - set(old_graph.nodes())
            incremental_nodes = [new_graph.nodes[node] for node in new_nodes]
            new_edges = set(new_graph.edges)
            old_edges = set(old_graph.edges)
            added_edges = [edge for edge in new_edges if edge not in old_edges]
            incremental_edges = []
            for edge in added_edges:
                data = {"src": edge[0], "dst": edge[1], "name": new_graph.edges[edge]["name"]}
                incremental_edges.append(data)
            update_data.added_nodes = []
            for node in incremental_nodes:
                if "id" in node and "info" in node and "label" in node:
                    update_data.added_nodes.append(node)
            update_data.added_edges = incremental_edges
        return update_data

    # 采用链式模式对txt、pdf、docx等文件通过大模型做图的三元组抽取并基于抽取的三元组信息构建图
    def graph_create(self, graphml_save_path: str, parsed_file_contents: dict, **kwargs):
        self.file_contents = parsed_file_contents
        logger.info("Graph Process start.")
        self._graph_process(graphml_save_path, False, **kwargs)

    def graph_update(self, graphml_save_path: str, parsed_file_contents: dict, opr_mode: KgOprMode, **kwargs):
        self.file_contents = parsed_file_contents
        old_graph = safe_read_graphml(graphml_save_path)
        self._graph_process(graphml_save_path, True, **kwargs)
        return self._diff_graph(opr_mode, old_graph, self.graph)

    def _graph_process(self, graphml_save_path, update_flag, **kwargs):
        kwargs.pop("lang")
        txt_graph_extract = TxtGraphExtract(self.file_contents, self.extractor, self.lang, llm=self.llm, **kwargs)
        logger.info("Graph extract start.")
        txt_graph_extract.handle_extract(self.graph, self.current_id, update_flag, self.extract_workers)
        try:
            nx.write_graphml(self.graph, graphml_save_path)
        except ValueError as e:
            if str(e) != "All strings must be XML compatible: Unicode or ASCII, no NULL bytes or control characters":
                raise e
            # filter invalid xml data and write graph again in case of above exception
            logger.error(str(e))
            self._filter_graph()
            nx.write_graphml(self.graph, graphml_save_path)
        logger.info("Graph write into networkx done.")

    def _filter_graph(self):
        def _filter_xml_string(text):
            return re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', text)

        for _, data in self.graph.nodes(data=True):
            if "info" in data:
                data["info"] = _filter_xml_string(data["info"])
            if "entity_type" in data:
                data["entity_type"] = _filter_xml_string(data["entity_type"])


def _update_entity_info(target_graph, entity, entity_id_map, current_id):
    if entity not in entity_id_map:
        entity_id = current_id + 1
        target_graph.add_node(entity_id, id=entity_id, label=ENTITY, info=str(entity))
        entity_id_map[entity] = entity_id
        # 更新current max id
        current_id += 1
    else:
        entity_id = entity_id_map[entity]
    return entity_id, current_id


class GraphExtract(ABC):

    def __init__(self, contexts: dict, extractor: Extractor, lang: str, **kwargs):
        self.extractor = extractor
        self.contexts = contexts
        self.lang_dict = lang_dict.get(lang, lang_zh)
        self.vector_db = kwargs.pop("vector_db", None)
        # entity_filter_flag控制是否消歧
        self.entity_filter_flag = kwargs.pop("entity_filter_flag", False)
        self.llm = kwargs.pop("llm", None)
        self.graph_name = kwargs.pop("graph_name", None)
        self.entity_filter = EntityFilter(graph_name=self.graph_name, llm=self.llm, vector_db=self.vector_db)

    def add_index(self, index_data: dict, partition_name: str):
        if not self.entity_filter_flag:
            return
        id_list = []
        data_list = []
        for key, value in index_data.items():
            id_list.append(key)
            data_list.append(value)
        self.vector_db.add_embedding(data_list, id_list, partition_name)

    def build_graph(self, target_graph: DiGraph, triplets_list: list,
                    entity_id_map: dict, current_id: int, chunk_id: int):
        """

        Args:
            target_graph: DiGraph对象，临时存储图关系
            triplets_list: 大模型抽取的三元组信息
            entity_id_map: 实体和ID映射，例如{'小红':1}, 初始建图为空,
            current_id: 建图当前的ID
            chunk_id:

        Returns:

        """
        if not triplets_list:
            return current_id
        entity_list = self.entity_filter.entity_disambiguation(triplets_list, self.entity_filter_flag)
        # 存储头、尾实体节点（entity）、连边（relation）
        for triplet in triplets_list:
            head_entity, relation, tail_entity = triplet[0], triplet[1], triplet[2],
            head_entity_id, current_id = _update_entity_info(target_graph, head_entity, entity_id_map, current_id)
            tail_entity_id, current_id = _update_entity_info(target_graph, tail_entity, entity_id_map, current_id)
            # head-tail连边、chunk-entity连边
            edge = target_graph.get_edge_data(head_entity_id, tail_entity_id)
            # 考虑到本次的head和tail实体都检索到了已有实体，则先获取原有关系，并拼接增加本次关系
            if edge is not None:
                relation = edge.get("name", "") + ", " + relation
            target_graph.add_edge(head_entity_id, tail_entity_id, name=str(relation))
            target_graph.add_edge(chunk_id, head_entity_id, name=self.lang_dict["include_entity"])
            target_graph.add_edge(chunk_id, tail_entity_id, name=self.lang_dict["include_entity"])
        # 如果是消歧场景，entity_list是本次新增且无相似的实体列表，需要加入向量数据库。
        if self.entity_filter_flag:
            # 节点数据写入向量数据库
            id_list = [entity_id_map[entity] for entity in entity_list if entity in entity_id_map]
            entity_list = [entity for entity in entity_list if entity in entity_id_map]
            self.entity_filter.add_embedding(entity_list, id_list, partition_name="entity")
        return current_id

    def handle_extract(self, graph: DiGraph, current_id: int, update_flag: bool,
                       threadpool: concurrent.futures.ThreadPoolExecutor):
        pass

    def parallel_extract(self, info: str, chunk_id: int):
        """

        Args:
            info: 切分的chunk字符串
            chunk_id: 对应的id

        Returns:

        """
        # chunk信息抽取
        summary, entity_schema_map, triplets_list, relations_list = self.extractor.extract(
            input_text_str=info)
        return summary, triplets_list, chunk_id

    def _update_graph_after_tasks(self, all_task, graph, entity_id_map, current_id, chunk_index_data):

        for future in tqdm(as_completed(all_task), total=len(all_task), desc="generating entities"):
            try:
                summary, triplets_list, chunk_id = future.result()
            except Exception as e:
                logger.error(f"Extract failed with exception:{e}")
                continue
            current_id = self.build_graph(graph, triplets_list, entity_id_map, current_id, chunk_id)
            # summary节点
            if summary != '':
                summary_id = current_id + 1
                graph.add_node(summary_id, id=summary_id, label=SUMMARY, info=str(summary))
                # chunk-summary连边
                graph.add_edge(chunk_id, summary_id, name=self.lang_dict[SUMMARY])
                # 更新current max id
                current_id = current_id + 1
                chunk_index_data[summary_id] = str(summary)
        return current_id, chunk_index_data


class TxtGraphExtract(GraphExtract):
    def handle_extract(self, graph, current_id, update_flag, threadpool):
        entity_id_map = {}
        # 如果是更新图，先从graph读取实体和id映射关系
        if update_flag:
            for _, data in graph.nodes.data():
                entity_id_map[data["info"]] = data["id"]
        txt_files_contents = self.contexts
        node_id_map = defaultdict(lambda: None)
        chunk_index_data = {}
        for file in txt_files_contents.keys():
            logger.info(f"processing file {file}...")
            chunks = txt_files_contents[file]
            # 创建txt文件节点
            file_id = current_id + 1
            graph.add_node(file_id, id=file_id, label=FILE, info=str(file))
            chunk_index_data[file_id] = str(file)
            node_id_map[file] = file_id
            current_id = current_id + 1
            all_task = []
            # 遍历txt文件下chunk
            for i, index in enumerate(chunks.keys()):
                chunk = chunks[index]
                chunk_old_id = chunk["id"]
                # chunk节点
                chunk_id = current_id + 1
                info = "\n".join(chunk["info"])
                graph.add_node(chunk_id, id=chunk_id, label=TEXT, info=info)
                chunk_index_data[chunk_id] = info
                node_id_map[chunk_old_id] = chunk_id
                current_id = current_id + 1
                # file-chunk连边
                graph.add_edge(file_id, chunk_id, name=self.lang_dict["include_content"])
                # chunk上下文连边
                if i > 0:
                    graph.add_edge(chunk_id, node_id_map[chunk_old_id - 1],
                                   name=self.lang_dict["include_preceding_content"])
                    graph.add_edge(node_id_map[chunk_old_id - 1], chunk_id,
                                   name=self.lang_dict["include_following_content"])

                # chunk信息抽取
                all_task.append(threadpool.submit(self.parallel_extract, info, chunk_id))
            current_id, chunk_index_data = self._update_graph_after_tasks(all_task, graph, entity_id_map,
                                                                          current_id, chunk_index_data)
            # add chunk data into vector db
            self.add_index(chunk_index_data, partition_name="text")
