# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from json_repair import repair_json
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.prompts.extract_graph import (
    CHAT_TEMPLATE,
    PASSAGE_START_CN,
    PASSAGE_START_EN,
    TRIPLE_INSTRUCTIONS_CN,
    TRIPLE_INSTRUCTIONS_EN,
)
from mx_rag.graphrag.prompts.repair_json import JSON_REPAIR_PROMPT
from mx_rag.graphrag.utils.json_util import (
    extract_json_like_substring,
    fix_entity_event_json_string,
    fix_entity_relation_json_string,
    fix_event_relation_json_string,
    normalize_json_string,
)
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import Lang
from mx_rag.storage.document_store import MxDocument


def _parse_and_repair_json(
    llm: Text2TextLLM,
    text: str,
    answer_start_token: str,
    repair_function: Optional[Callable[[str], str]] = None,
    remove_space: bool = False,
    handle_single_quote: bool = False,
    llm_repair_prompt_template: str = JSON_REPAIR_PROMPT,
) -> List[dict]:
    """
    Efficiently parse and repair a JSON-like string, escalating from local fixes to LLM repair.
    """
    json_text = extract_json_like_substring(text, answer_start_token).strip()
    json_text = normalize_json_string(json_text, remove_space, handle_single_quote)

    def try_parse(s: str) -> Optional[List[dict]]:
        try:
            return json.loads(s)
        except Exception:
            return None

    # Try direct parse
    result = try_parse(json_text)
    if result is not None:
        return result

    # Try repair_json + repair_function if provided
    if repair_function:
        try:
            repaired = repair_json(json_text)
            repaired = normalize_json_string(repaired, remove_space, handle_single_quote)
            repaired = repair_function(repaired)
            result = try_parse(repaired)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("Repair function failed: %s", e)

        # Try only repair_function
        try:
            repaired = repair_function(json_text)
            result = try_parse(repaired)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("Repair function (direct) failed: %s", e)

    # Try LLM repair
    try:
        query = llm_repair_prompt_template.format(q=json_text)
        llm_output = llm.chat(query)
        result = try_parse(llm_output)
        if result is not None:
            logger.info("Successfully fixed by LLM!")
            return result
        else:
            logger.warning("LLM output could not be parsed: %s", llm_output)
    except Exception as e:
        logger.warning("LLM repair failed: %s", e)

    logger.warning("All repair attempts failed. Discarding: %s", json_text)
    return []


def generate_relations_cn(
    llm: Text2TextLLM,
    pad_token: str,
    texts: List[str],
    answer_start_token: str,
    repair_function: Callable[[str], str],
) -> List[List[dict]]:
    """
    Generalized function to generate a list of relations from the model output (Chinese).
    """
    processed_texts = [text.replace(pad_token, "") for text in texts]
    relations = []
    for text in processed_texts:
        relations.append(
            _parse_and_repair_json(llm, text, answer_start_token, repair_function, True, True)
        )
    return relations


def generate_relations_en(
    llm: Text2TextLLM, texts: List[str], answer_start_token: str
) -> List[List[dict]]:
    """
    Generates a list of entity relation dictionaries from the model output (English).
    """
    return [
        _parse_and_repair_json(llm, text, answer_start_token, repair_json)
        for text in texts
    ]


class LLMRelationExtractor:
    """
    LLMRelationExtractor is a class designed to extract relations and entities from text using a language model (LLM).
    It supports multiple configurations for different extraction tasks and languages.
    Attributes:
        llm (Text2TextLLM): The language model used for text-to-text generation.
        pad_token (str): The token used for padding in the LLM.
        language (Lang): The language setting, defaulting to Chinese (Lang.CH).
        triple_instructions (dict): Instructions for extracting triples based on the language.
        chat_template (dict): Template configurations for the LLM model.
        prompt_start (str): The starting prompt for the LLM.
        prompt_end (str): The ending prompt for the LLM.
        model_start (str): The prefix indicating the model's response.
        system_start (str): The prefix indicating the system's instructions.
        configs (dict): Configurations for different extraction tasks (entity_relation, event_entity, event_relation).
    """
    def __init__(self, llm: Text2TextLLM, pad_token: str, language: Lang = Lang.CH, max_workers=None):
        logger.info("Initializing RelationExtractorLLM...")
        self.llm = llm
        self.pad_token = pad_token
        self.language = language
        self.max_workers = max_workers
        self.triple_instructions = TRIPLE_INSTRUCTIONS_CN if language == Lang.CH else TRIPLE_INSTRUCTIONS_EN

        self.chat_template = CHAT_TEMPLATE.get(self.llm.model_name, {})

        self.prompt_start = self.chat_template.get("prompt_start", "")
        self.prompt_end = self.chat_template.get("prompt_end", "")
        self.model_start = self.chat_template.get("model_start", "")
        self.system_start = self.chat_template.get("system_start", "")
        passage_start = PASSAGE_START_CN if self.language == Lang.CH else PASSAGE_START_EN

        self.configs = {
            "entity_relation": self._build_config("entity_relation", passage_start),
            "event_entity": self._build_config("event_entity", passage_start),
            "event_relation": self._build_config("event_relation", passage_start),
        }

    def query(self, docs: List[MxDocument]) -> List[dict]:
        outputs = {key: [] for key in self.configs}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for key, config in self.configs.items():
                logger.info(f"Processing texts for key: {key}")
                texts = [doc.page_content for doc in docs]
                outputs[key] = list(
                    tqdm(
                        executor.map(lambda text: self._generate_stage_output(text, config), texts),
                        total=len(texts),
                        desc=f"Processing {key}",
                    )
                )
                logger.info(f"Finished processing texts for key: {key}")

        logger.info("Processing entity relations...")
        entity_relations = self._process_relations(outputs["entity_relation"], fix_entity_relation_json_string)

        logger.info("Processing event-entity relations...")
        event_entity_relations = self._process_relations(outputs["event_entity"], fix_entity_event_json_string)

        logger.info("Processing event relations...")
        event_relations = self._process_relations(outputs["event_relation"], fix_event_relation_json_string)

        return [
            {
                "raw_text": doc.page_content,
                "file_id": doc.metadata["source"],
                "entity_relations": entity_relations[i],
                "event_entity_relations": event_entity_relations[i],
                "event_relations": event_relations[i],
                "output_stage_one": outputs["entity_relation"][i],
                "output_stage_two": outputs["event_entity"][i],
                "output_stage_three": outputs["event_relation"][i],
            }
            for i, doc in enumerate(docs)
        ]
    
    def _build_config(self, key: str, passage_start: str) -> dict:
        return {
            "prefix": f"{self.system_start}{self.prompt_start}{self.triple_instructions.get(key)}{passage_start}",
            "suffix": f"{self.prompt_end}{self.model_start}",
        }

    def _generate_stage_output(self, text: str, config: dict) -> str:
        query = f"{config['prefix']}{text}{config['suffix']}"
        output = self.model_start + self.llm.chat(query)
        return output

    def _process_relations(self, outputs: List[str], repair_function: Callable) -> List[List[dict]]:
        if self.language == Lang.CH:
            relations = generate_relations_cn(self.llm, self.pad_token, outputs, self.model_start, repair_function)
        else:
            relations = generate_relations_en(self.llm, outputs, self.model_start)
        return relations