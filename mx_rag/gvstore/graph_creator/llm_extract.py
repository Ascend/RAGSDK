# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import ast
import json
import re

from loguru import logger

from mx_rag.gvstore.prompt.prompt_template import PROMPTS
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params


class Extractor:
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM")
    )
    def __init__(self, llm: Text2TextLLM,
                 extractor_method: str = "triplets_and_summary",
                 entity_types: list[str] = None,
                 **kwargs) -> None:
        self.model = llm
        if "lang" in kwargs:
            if not isinstance(kwargs.get("lang"), str):
                raise KeyError("lang param error, it should be str type")
            if kwargs.get("lang") not in ["zh", "en"]:
                raise ValueError(f"lang param error, value must be in [zh, en]")
        self.lang = kwargs.get("lang", "zh")
        if entity_types is None:
            self.entity_types = []
        else:
            self.entity_types = entity_types
        self.default_prompt = PROMPTS["DEFAULT_TOTAL_INFO_TMPL"]
        if extractor_method == "triplets_and_summary":
            if self.entity_types:
                self.default_prompt = PROMPTS["DEFAULT_TOTAL_INFO_TMPL_WITH_ENTITY_TYPES"] \
                    if self.lang == "zh" else PROMPTS["DEFAULT_TOTAL_INFO_TMPL_WITH_ENTITY_TYPES_EN"]
            else:
                self.default_prompt = PROMPTS["DEFAULT_TOTAL_INFO_TMPL"] \
                    if self.lang == "zh" else PROMPTS["DEFAULT_TOTAL_INFO_TMPL_EN"]

    # 调用大模型进行三元组抽取
    def model_process(self, input_text_str: str) -> dict:
        input_text_str = input_text_str.replace("\"", "\'")
        if self.entity_types:
            input_entity_types = "，".join(self.entity_types)
            prompt = self.default_prompt.format(text=input_text_str, entity_types=input_entity_types)
        else:
            prompt = self.default_prompt.format(text=input_text_str)
        response = self.model.chat(prompt, llm_config=self.model.llm_config)
        result = response.replace('\n', '')
        try:
            pattern_text = re.findall('{.*}', result)
            if pattern_text and pattern_text[0]:
                return ast.literal_eval(pattern_text[0])
        except Exception as e:
            logger.error(f"解析LLM抽取triplets结果失败: {e}")
            if str(e).startswith("malformed node or string on line"):
                dict_result = json.loads(result)
                return dict_result
        return {}

    # 整合大模型返回的三元组信息，结构化续创图所需数据
    def extract_triplet_and_relation_list(self, triplet: list, entity_schema: dict, entity_schema_map: dict,
                                          triplets_list: list, relations_list: list):
        if len(triplet) < 3:
            logger.warning(f"Triplet from LLM is not correct: {triplet}")
            return

        def _extract_entity_schema(entity: str):
            nonlocal entity_schema_map
            if entity in entity_schema:
                concept = entity_schema[entity]
            else:
                # 用户预定义entity types schema，没有提取到schema的节点直接跳过
                if self.entity_types:
                    return
                else:
                    concept = entity

            if entity not in entity_schema_map:
                entity_schema_map[entity] = [concept]
            else:
                entity_schema_map[entity].append(concept)
                entity_schema_map[entity] = list(set(entity_schema_map[entity]))

        head_entity, relation, tail_entity = triplet[0], triplet[1], triplet[2]
        if not head_entity or not relation or not tail_entity:
            logger.warning(f"Triplet from LLM invalid: {triplet}")
            return
        # extract head entity schema
        _extract_entity_schema(head_entity)
        # extract tail entity schema
        _extract_entity_schema(tail_entity)
        triplets_list.append(triplet)
        relations_list.append(relation)

    # 调用大模型进行三元组抽取
    def extract(self, input_text_str: str):
        summary = ""
        entity_schema_map = {}
        triplets_list = []
        relations_list = []

        try:
            text_dict = self.model_process(input_text_str)
        except Exception as e:
            logger.error(f"LLM抽取triplets失败: {e}")
            return [summary, entity_schema_map, triplets_list, relations_list]

        if len(input_text_str) >= 50:
            if 'Summary' in text_dict:
                summary = text_dict['Summary']

        if ('Triplets' in text_dict) and ('Entity' in text_dict):
            text_triplets = text_dict['Triplets']
            entity_schema = text_dict['Entity']
            for triplet in text_triplets:
                # 提取当前三元组
                try:
                    self.extract_triplet_and_relation_list(triplet, entity_schema, entity_schema_map,
                                                           triplets_list, relations_list)
                except Exception as e:
                    logger.error(f"Parse triplets failed: {e}")
                    pass

        return [summary, entity_schema_map, triplets_list, relations_list]
