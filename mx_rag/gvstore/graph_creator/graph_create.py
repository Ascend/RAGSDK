# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import concurrent.futures
from abc import ABC
from collections import defaultdict
from concurrent.futures import as_completed

import networkx as nx
from loguru import logger
from networkx import DiGraph

from mx_rag.gvstore.graph_creator.lang import lang_dict, lang_zh
from mx_rag.gvstore.graph_creator.llm_extract import Extractor
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params, validata_list_str

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
                        message="current_id param must be None or int"),
        entity_types=dict(validator=lambda x: x is None or validata_list_str(x, [1, 1000], [1, 1000]),
                          message="entity_types param must be None or list[str]")
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
        if "thread_num" in kwargs and not isinstance(kwargs.get("thread_num"), int):
            raise KeyError("thread_num param error, it should be integer type")
        thread_num = kwargs.get("thread_num", 8)
        if "lang" in kwargs:
            if not isinstance(kwargs.get("lang"), str):
                raise KeyError("lang param error, it should be str type")
            if kwargs.get("lang") not in ["zh", "en"]:
                raise ValueError(f"lang param error, value must be in [zh, en]")
        self.lang = kwargs.get("lang", "zh")
        self.model = llm
        # 存储networkx图
        if graph is not None:
            self.graph = graph
        else:
            self.graph = DiGraph()

        if current_id is not None:
            self.current_id = current_id
        else:
            self.current_id = -1

        # 图信息存储
        self.node_id_map = {}

        # info抽取类
        self.extractor = Extractor(llm=self.model, entity_types=entity_types, **kwargs)
        # info抽取结果存储
        self.entity_id_map = {}
        self.extract_workers = concurrent.futures.ThreadPoolExecutor(max_workers=thread_num,
                                                                     thread_name_prefix="extract_workers")

    # 采用链式模式对txt、pdf、docx等文件通过大模型做图的三元组抽取并基于抽取的三元组信息构建图
    def graph_create(self, graphml_save_path: str, parsed_file_contents: dict):
        file_contents = parsed_file_contents
        entity_map = {}
        txt_graph_extract = TxtGraphExtract(file_contents, self.extractor, self.lang)
        txt_graph_extract.handle_extract(self.graph, self.current_id, entity_map, self.extract_workers)
        nx.write_graphml(self.graph, graphml_save_path)


class GraphExtract(ABC):

    def __init__(self, contexts: dict, extractor: Extractor, lang: str):
        self.extractor = extractor
        self.contexts = contexts
        self.lang_dict = lang_dict.get(lang, lang_zh)

    def build_graph(self, target_graph: DiGraph, triplets_list: list,
                    entity_id_map: dict, current_id: int, chunk_id: int):
        # 存储头、尾实体节点（entity）、连边（relation）
        for triplet in triplets_list:
            head_entity, relation, tail_entity = triplet[0], triplet[1], triplet[2],
            if head_entity not in entity_id_map:
                head_entity_id = current_id + 1
                target_graph.add_node(head_entity_id, id=head_entity_id, label=ENTITY, info=str(head_entity))
                entity_id_map[head_entity] = head_entity_id
                # 更新current max id
                current_id += 1
            else:
                head_entity_id = entity_id_map[head_entity]

            if tail_entity not in entity_id_map:
                tail_entity_id = current_id + 1
                target_graph.add_node(tail_entity_id, id=tail_entity_id, label=ENTITY, info=str(tail_entity))
                entity_id_map[tail_entity] = tail_entity_id
                # 更新current max id
                current_id += 1
            else:
                tail_entity_id = entity_id_map[tail_entity]
            # head-tail连边、chunk-entity连边
            edge = target_graph.get_edge_data(head_entity_id, tail_entity_id)
            if edge is not None:
                relation = edge.get("name", "") + ", " + relation
            target_graph.add_edge(head_entity_id, tail_entity_id, name=str(relation))
            target_graph.add_edge(chunk_id, head_entity_id, name=self.lang_dict["include_entity"])
            target_graph.add_edge(chunk_id, tail_entity_id, name=self.lang_dict["include_entity"])
        return current_id

    def handle_extract(self, graph: DiGraph, current_id: int, entity_id_map: dict,
                       threadpool: concurrent.futures.ThreadPoolExecutor):
        pass

    def parallel_extract(self, info: str, chunk_id: int):
        # chunk信息抽取
        summary, entity_schema_map, triplets_list, relations_list = self.extractor.extract(
            input_text_str=info)
        return summary, triplets_list, chunk_id

    def _update_graph_after_tasks(self, all_task, graph, entity_id_map, current_id):
        for future in as_completed(all_task):
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
        return current_id


class TxtGraphExtract(GraphExtract):
    def handle_extract(self, graph, current_id, entity_id_map, threadpool):
        txt_files_contents = self.contexts
        node_id_map = defaultdict(lambda: None)
        for file in txt_files_contents.keys():
            logger.info(f"processing file {file}...")
            chunks = txt_files_contents[file]
            # 创建txt文件节点
            file_id = current_id + 1
            graph.add_node(file_id, id=file_id, label=FILE, info=str(file))
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
            current_id = self._update_graph_after_tasks(all_task, graph, entity_id_map, current_id)
