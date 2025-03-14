# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from loguru import logger
from mx_rag.gvstore.graph_creator.vdb.vector_db import VectorDBBase
from mx_rag.gvstore.prompt.prompt_template import PROMPTS
from mx_rag.gvstore.util.utils import MAX_NAME_LENTH
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params


class EntityFilter:
    """
    创建知识图谱：实体消岐处理
    """

    @validate_params(
        graph_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_NAME_LENTH,
                        message=f"param must be a str and its length meets (0, {MAX_NAME_LENTH}]"),
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        vector_db=dict(validator=lambda x: isinstance(x, VectorDBBase),
                       message="param must be instance of subclass of VectorDBBase"),
    )
    def __init__(self, graph_name: str, llm: Text2TextLLM, vector_db) -> None:
        self.graph_name = graph_name
        self.llm = llm
        self.vector_db = vector_db
        self.prompt = PROMPTS["SAME_ENTITY_CHECK"]

    @staticmethod
    def extract_head_tail(triplet_list):
        result = []
        for triplet in triplet_list:
            if triplet:
                result.append(triplet[0])
                result.append(triplet[-1])
        result = list(set(result))
        return result

    def entity_disambiguation(self, triplet_list: list[list], entity_filter_flag):
        """
        1. 如果不消歧，直接返回本次大模型抽取的三元组
        2. 如果消歧，先检索已有的近似的实体，triplet_list中能检索到近似实体，则替换为已有的实体；检索不到的近似实体，通过entity_list返回。
        Args:
            triplet_list: 本次大模型抽取的三元组
            entity_filter_flag: 是否消歧

        Returns:
        """
        collection_empty = self.vector_db.row_count() == 0
        entity_list = self.extract_head_tail(triplet_list)
        # 如果不消歧，直接返回实体
        if not entity_filter_flag:
            return entity_list

        def merge_triplets(triplets: list):
            nonlocal related_entities_map
            for triplet in triplets:
                if not triplet:
                    continue
                # check whether head and tail entity already in embedding db
                if triplet[0] in related_entities_map:
                    triplet[0] = related_entities_map[triplet[0]]
                if triplet[-1] in related_entities_map:
                    triplet[-1] = related_entities_map[triplet[-1]]

        if collection_empty:
            return entity_list
        else:
            query_result = self.query_embedding(self.graph_name, triplet_list, partition_names=["entity"])
            related_entities_map = self.filter_entities(query_result)
            filtered_entities = entity_list
            if related_entities_map:
                logger.debug(f"Identify and eliminate ambiguous nodes {len(related_entities_map)}")
                filtered_entities = [entity for entity in entity_list if entity not in related_entities_map]
                # 将triplet_list中判断为已有实体的，替换为已有实体
                merge_triplets(triplet_list)
            return filtered_entities

    def add_embedding(self, entity_list: list, id_list: list, partition_name: str):
        if not entity_list:
            return
        self.vector_db.add_embedding(entity_list, id_list, partition_name)

    def query_embedding(self, collection_name, triplet_list: list, **kwargs):
        entities = self.extract_head_tail(triplet_list)
        return self.vector_db.query_embedding(collection_name, entities, **kwargs)

    def filter_entities(self, data_list: list) -> dict:
        related_entities_map = {}
        for _, name, related_name, distance in data_list:
            if distance >= 0.9:
                related_entities_map[name] = related_name
        if related_entities_map:
            for key in list(related_entities_map.keys()):
                related_value = related_entities_map.get(key, None)
                if related_value == key:
                    continue
                same = self.llm_check_same_entity(key, related_entities_map.get(key))
                if not same:
                    related_entities_map.pop(key)
        return related_entities_map

    def llm_check_same_entity(self, entity1: str, entity2: str):
        prompt = self.prompt.format(entity1=entity1, entity2=entity2)
        answer = self.llm.chat(prompt, llm_config=self.llm.llm_config)
        result = True if answer == "是" else False
        return result
