# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import time
import random
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from tqdm import tqdm

from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import Lang
from mx_rag.graphrag.prompts.extract_graph import (
    CHAT_TEMPLATE,
    ENTITY_PROMPT_CN,
    ENTITY_PROMPT_EN,
    EVENT_PROMPT_CN,
    EVENT_PROMPT_EN,
    RELATION_PROMPT_CN,
    RELATION_PROMPT_EN,
)
from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.graphrag.graphs.opengauss_graph import OpenGaussGraph


def extract_event_nodes(graph) -> List[Any]:
    """
    Extract all event nodes from the graph.
    """
    return graph.get_nodes_by_attribute(key="type", value="event")


def extract_entity_nodes(graph) -> List[Any]:
    """
    Extract all entity nodes from the graph.
    """
    return graph.get_nodes_by_attribute(key="type", value="entity")


def extract_relation_edges(graph) -> List[Any]:
    """
    Extract all relation edges from the graph.
    """
    seen = set()
    relations = []
    for relation in graph.get_edge_attribute_values(key="relation"):
        if relation not in seen:
            seen.add(relation)
            relations.append(relation)
    return relations


class GraphConceptualizer:
    """
    Conceptualizes events, entities, and relations in a graph using an LLM.
    """

    def __init__(
        self,
        llm: Text2TextLLM,
        graph: GraphStore,
        sample_num: Optional[int] = None,
        lang: Lang = Lang.CH,
        chat_template: Optional[Dict[str, str]] = None,
        err_instructions: Optional[Dict[str, str]] = None,
        seed: int = 4096,
    ) -> None:
        random.seed(seed)
        self.llm = llm
        self.graph = graph
        self.sample_num = sample_num

        self.prompt_map = {
            "event_prompt": EVENT_PROMPT_CN if lang == Lang.CH else EVENT_PROMPT_EN,
            "entity_prompt": ENTITY_PROMPT_CN if lang == Lang.CH else ENTITY_PROMPT_EN,
            "relation_prompt": RELATION_PROMPT_CN if lang == Lang.CH else RELATION_PROMPT_EN,
        }
        if err_instructions:
            self.prompt_map.update(err_instructions)

        self.events = extract_event_nodes(self.graph)
        self.entities = extract_entity_nodes(self.graph)
        self.relations = extract_relation_edges(self.graph)

        if sample_num:
            self.events = random.sample(self.events, min(sample_num, len(self.events)))
            self.entities = random.sample(self.entities, min(sample_num, len(self.entities)))
            self.relations = random.sample(self.relations, min(sample_num, len(self.relations)))

        self.chat_template = CHAT_TEMPLATE.get(self.llm.model_name, {})
        if chat_template:
            self.chat_template.update(chat_template)

    def conceptualize(self) -> List[Dict[str, Any]]:
        """
        Conceptualize events, entities, and relations in the graph in parallel.

        Returns:
            List of conceptualized nodes and relations.
        """
        result = []

        def run_parallel(items, func, desc):
            outputs = []
            with ThreadPoolExecutor() as executor:
                future_to_item = {executor.submit(func, item): item for item in items}
                for future in tqdm(as_completed(future_to_item), total=len(items), desc=desc):
                    outputs.append(future.result())
            return outputs

        result.extend(run_parallel(self.events, self._conceptualize_event, "Conceptualizing events"))
        result.extend(run_parallel(self.entities, self._conceptualize_entity, "Conceptualizing entities"))
        result.extend(run_parallel(self.relations, self._conceptualize_relation, "Conceptualizing relations"))

        return result

    def _build_query(self, prompt: str) -> str:
        """
        Build a query string for the LLM.

        Args:
            prompt: The prompt to use.

        Returns:
            The formatted query string.
        """
        return (
            self.chat_template.get("system_start", "")
            + self.chat_template.get("prompt_start", "")
            + prompt
            + self.chat_template.get("prompt_end", "")
            + self.chat_template.get("model_start", "")
        )

    def _conceptualize_event(self, event: str) -> Dict[str, Any]:
        """
        Conceptualize a single event node.

        Args:
            event: The event node.

        Returns:
            Dict with conceptualized event.
        """
        prompt = self.prompt_map["event_prompt"].replace("[EVENT]", event)
        query = self._build_query(prompt)
        answer = self.llm.chat(query)
        return {
            "node": event,
            "conceptualized_node": answer,
            "node_type": "event",
        }

    def _conceptualize_entity(self, entity: str) -> Dict[str, Any]:
        """
        Conceptualize a single entity node.

        Args:
            entity: The entity node.

        Returns:
            Dict with conceptualized entity.
        """
        entity_name = entity.split(":::")[0] if ":::" in entity else entity
        prompt = self.prompt_map["entity_prompt"].replace("[ENTITY]", entity_name)

        if isinstance(self.graph, OpenGaussGraph):
            # Multi-thread case: each thread gets its connection
            local_graph = OpenGaussGraph(self.graph.graph_name, self.graph.conf)
        else:
            local_graph = self.graph
        t0 = time.time()
        entity_predecessors = list(local_graph.predecessors(entity))
        t1 = time.time()
        entity_successors = list(local_graph.successors(entity))
        t2 = time.time()
        predecessors_time_ms = (t1 - t0) * 1000
        successors_time_ms = (t2 - t1) * 1000
        total_time_ms = (t2 - t0) * 1000
        
        # Only log if operations are slow (> 200ms)
        if total_time_ms > 200:
            logger.debug(
                f"Graph query performance for entity '{entity_name}': "
                f"predecessors={predecessors_time_ms:.2f}ms ({len(entity_predecessors)} nodes), "
                f"successors={successors_time_ms:.2f}ms ({len(entity_successors)} nodes), "
                f"total={total_time_ms:.2f}ms"
            )

        context = ""
        if entity_predecessors:
            neighbors = random.sample(entity_predecessors, min(1, len(entity_predecessors)))
            context += ", ".join(
                f"{neighbor} {local_graph.get_edge_attributes(neighbor, entity, 'relation')}" for neighbor in neighbors
            )
        if entity_successors:
            neighbors = random.sample(entity_successors, min(1, len(entity_successors)))
            if context:
                context += ", "
            context += ", ".join(
                f"{local_graph.get_edge_attributes(entity, neighbor, 'relation')} {neighbor}" for neighbor in neighbors
            )

        prompt = prompt.replace("[CONTEXT]", context)
        query = self._build_query(prompt)
        answer = self.llm.chat(query)
        return {
            "node": entity,
            "conceptualized_node": answer,
            "node_type": "entity",
        }

    def _conceptualize_relation(self, relation: str) -> Dict[str, Any]:
        """
        Conceptualize a single relation.

        Args:
            relation: The relation.

        Returns:
            Dict with conceptualized relation.
        """
        prompt = self.prompt_map["relation_prompt"].replace("[RELATION]", relation)
        query = self._build_query(prompt)
        answer = self.llm.chat(query)
        return {
            "node": relation,
            "conceptualized_node": answer,
            "node_type": "relation",
        }
