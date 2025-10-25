#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

import networkx as nx
from loguru import logger
from tqdm import tqdm

from mx_rag.utils.common import Lang, validate_params
from mx_rag.graphrag.graph_merger import (
    extract_event_entity_triples,
    get_language_keys,
    RAW_TEXT_KEY,
    FILE_ID_KEY,
)
from mx_rag.graphrag.graphs.graph_store import GraphStore


NAMED_ENTITY_PROMPT_EN = '''
Here I will give you a short piece of text, and you need to tell me if is a named entity. a real-world object, such as a person, location, organization, product, etc., that can be denoted with a proper name.
For example, "New York City" is a named entity, but "city" is not a named entity.
Here are the requirements:
1. If it is a named entity, please type "yes", otherwise, type "no".
2. If the NODE is any nominal noun, like guitar, table, doctor, or cup of coffee, as long as its the same meaning, you shoud say "no".
3. If you are not sure, say "no".
4. Do not reply anything other than "yes" or "no".

TEXT: [TEXT]
Your answer:
'''

NAMED_ENTITY_PROMPT_CN = '''
在这里，我将给你一段简短的文本，你需要告诉我它是否是一个命名实体。命名实体通常指特定的、可用专有名称表示的对象，如人名、地点名、组织名、产品名等。与专有名词相对的是通用名词，指代一类产品，没有特定的指向。
例如，“昆明市”是一个命名实体，但“城市”不是命名实体；“三星手机”是一个命名实体，但“手机”不是一个命名实体；“中信建投证券股份有限公司”是一个命名实体，但“公司”不是一个命名实体；“昆船智能技术股份有限公司保荐机构”是一个命名实体，但“机构”不是一个命名实体。
以下是要求：
1. 如果它是命名实体，请键入“yes”，否则键入“no”。
2. 如果节点是任何名词，如吉他、桌子、医生或咖啡杯，只要意思相同，你应该说“no”。
3. 如果你不确定，回答“no”。
4. 除了“yes”或“no”，不要回复其他内容。

文本：[TEXT]
你的答案：
'''

ENTITY_PROMPT_EN = '''
Here I will give you two named entities and their context, you need to tell me if they have the same meaning by conducting inference according to the context.
Here are the requirements:
1. Given the context, if the named entities do not have the same meaning, please say "no", otherwise, say "yes".
2. Do not reply anything other than "yes" or "no".

NODE 1: [NODE1]
CONTEXT 1: [CONTEXT1]
NODE 2: [NODE2]
CONTEXT 2: [CONTEXT2]
Your answer:
'''

ENTITY_PROMPT_CN = '''
在这里，我将给你两个命名实体及其上下文，你需要根据上下文推断它们是否具有相同的含义。
例如，同一个人名可能指代不同的人：在“网球运动员李娜曾是世界排名第二的选手”和“李娜是中国优秀的跳水运动员”中，“李娜”并不具有相同含义。
以下是要求：
1. 根据上下文，如果命名实体没有相同的含义，请说“no”，否则说“yes”。
2. 除了“yes”或“no”，不要回复其他内容。

节点 1: [NODE1]
上下文 1: [CONTEXT1]
节点 2: [NODE2]
上下文 2: [CONTEXT2]
你的答案：
'''


@dataclass
class DisambiguationConfig:
    """Configuration for disambiguation process."""
    ner_batch_size: int = 10
    disambiguation_batch_size: int = 10
    max_token_length: int = 4096
    max_workers: Optional[int] = None


class EntityContextManager:
    """Manages entity context extraction and storage."""
    
    def __init__(self):
        self.entity_context_data: Dict[str, Dict[str, str]] = {}
    
    def extract_context_for_entity(self, triple: Tuple[str, str, str], raw_text: str, file_id: str) -> None:
        """Extract and assign context sentences for head and tail entities of a triple."""
        def assign_context(entity: str) -> None:
            sentences = re.split(r"。|\.", raw_text)
            triple_sentences = [s for s in sentences if entity in s or entity.lower() in s]
            selected_sentence = triple_sentences[0] if triple_sentences else random.choice(sentences)
            
            if entity not in self.entity_context_data:
                self.entity_context_data[entity] = {}
            self.entity_context_data[entity][str(file_id)] = selected_sentence

        assign_context(triple[0])  # head
        assign_context(triple[2])  # tail
    
    def get_context(self, entity: str, file_id: str) -> Optional[str]:
        """Get context for a specific entity and file."""
        return self.entity_context_data.get(entity, {}).get(file_id)
    

class NamedEntityRecognizer:
    """Handles named entity recognition using LLM."""
    
    def __init__(self, llm, model_name: str, max_workers: Optional[int] = None):
        self.llm = llm
        self.model_name = model_name
        if max_workers is not None:
            max_workers = min(max_workers, (os.cpu_count() or 1) + 4)
        self.max_workers = max_workers
    
    @validate_params(entity_nodes=dict(validator=lambda x: len(x) < 50000, message="Too many entities"))
    def identify_named_entities(self, entity_nodes: List[str], lang: Lang) -> List[str]:
        """Identify named entities from entity nodes using parallelized LLM prompts."""
        prompt_template = NAMED_ENTITY_PROMPT_CN if lang == Lang.CH else NAMED_ENTITY_PROMPT_EN
        
        prompts = [
            prompt_template.replace("[TEXT]", node)
            for node in entity_nodes
        ]
        
        results = self._process_parallel_llm_calls(
            prompts, entity_nodes, "Identifying named entities"
        )
        
        return [
            entity_nodes[idx] for idx, result in enumerate(results)
            if result and result.lower() == "yes"
        ]
    
    def _process_parallel_llm_calls(self, prompts: List[str], items: List[str], desc: str) -> List[str]:
        """Process LLM calls in parallel and return results."""
        results = [""] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.llm.chat, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(items), desc=desc):
                idx = future_to_idx[future]
                try:
                    output = future.result()
                    answer = next(
                        (line.strip() for line in reversed(output.splitlines()) if line.strip()), 
                        ""
                    )
                    results[idx] = answer
                except TimeoutError as e:
                    logger.error(f"LLM call timed out for item {items[idx]}: {e}")
                except ConnectionError as e:
                    logger.error(f"LLM call connection error for item {items[idx]}: {e}")
                except Exception as e:
                    logger.error(f"LLM call failed for item {items[idx]}: {e}")
        
        return results
    

class EntityDisambiguator:
    """Handles entity disambiguation using context comparison."""
    
    def __init__(self, llm, model_name: str, graph: GraphStore, max_workers: Optional[int] = None):
        if not isinstance(graph, GraphStore):
            raise ValueError("graph must be an instance of GraphStore.")
        self.llm = llm
        self.model_name = model_name
        self.graph = graph
        if max_workers is not None:
            max_workers = min(max_workers, (os.cpu_count() or 1) + 4)
        self.max_workers = max_workers
    
    @validate_params(named_entity_nodes=dict(validator=lambda x: len(x) < 50000, message="Too many entities"))
    def disambiguate_entities(self, named_entity_nodes: List[str], 
                              context_manager: EntityContextManager, lang: Lang) -> None:
        """Disambiguate named entity nodes by context using LLM-based inference."""
        for node in tqdm(named_entity_nodes, desc="Disambiguating entities"):
            self._process_single_entity(node, context_manager, lang)
    
    def _process_single_entity(self, node: str, context_manager: EntityContextManager, lang: Lang) -> None:
        """Process disambiguation for a single entity."""
        file_ids_string = self.graph.get_node_attributes(node, "file_id")
        file_ids_unique = list(set(file_ids_string.split(","))) if file_ids_string else []
        
        if len(file_ids_unique) <= 1:
            return
        
        context_graph = self._build_context_graph(node, file_ids_unique, context_manager, lang)
        connected_components = list(nx.connected_components(context_graph))
        
        if len(connected_components) > 1:
            logger.info(f"Disambiguating {node}...")
            self._split_node_by_components(node, connected_components)
    
    def _build_context_graph(self, node: str, file_ids: List[str], 
                             context_manager: EntityContextManager, lang: Lang) -> nx.Graph:
        """Build a context graph for entity disambiguation."""
        context_graph = nx.Graph()
        for file_id in file_ids:
            context_graph.add_node(file_id)
        
        # Collect node pairs for comparison
        node_pairs = []
        file_id_pairs = []
        
        for i, file_id1 in enumerate(file_ids):
            for j in range(i + 1, len(file_ids)):
                file_id2 = file_ids[j]
                context1 = context_manager.get_context(node, file_id1)
                context2 = context_manager.get_context(node, file_id2)
                
                if context1 and context2:
                    node_pairs.append((node, context1, node, context2))
                    file_id_pairs.append((file_id1, file_id2))
        
        if not node_pairs:
            return context_graph
        
        # Check if pairs refer to same entity
        similarity_results = self._check_entity_similarity(node_pairs, lang)
        
        # Add edges for similar entities
        for (file_id1, file_id2), is_same in zip(file_id_pairs, similarity_results):
            if is_same:
                context_graph.add_edge(file_id1, file_id2)
        
        return context_graph
    
    def _check_entity_similarity(self, node_pairs: List[Tuple[str, str, str, str]], lang: Lang) -> List[bool]:
        """Check if entity pairs refer to the same meaning using LLM."""
        prompt_template = ENTITY_PROMPT_CN if lang == Lang.CH else ENTITY_PROMPT_EN
        
        prompts = [
            prompt_template.replace("[NODE1]", pair[0]).replace("[CONTEXT1]", pair[1]).replace("[NODE2]", pair[2]).replace("[CONTEXT2]", pair[3])
            for pair in node_pairs
        ]
        
        results = self._process_parallel_similarity_checks(prompts)
        return [result.lower() == "yes" for result in results]
    
    def _process_parallel_similarity_checks(self, prompts: List[str]) -> List[str]:
        """Process similarity checks in parallel."""
        results = [""] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.llm.chat, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc="Checking entity similarity"):
                idx = future_to_idx[future]
                try:
                    output = future.result()
                    answer = next(
                        (line.strip() for line in reversed(output.splitlines()) if line.strip()), 
                        ""
                    )
                    results[idx] = answer
                except TimeoutError as e:
                    logger.error(f"LLM call failed for similarity check {idx}: {e}")
                except ConnectionError as e:
                    logger.error(f"LLM call failed for similarity check {idx}: {e}")
                except Exception as e:
                    logger.error(f"LLM call failed for similarity check {idx}: {e}")
        
        return results
    
    def _split_node_by_components(self, node: str, connected_components: List[Set[str]]) -> None:
        """Split a node into multiple nodes based on connected components."""
        new_edges = []
        
        for component_id, component in enumerate(connected_components):
            component_list = list(component)
            new_node = f"{node}:::{component_id + 1}"
            
            # Create new node with filtered file IDs
            self._create_split_node(node, new_node, component_list)
            
            # Handle edges for the new node
            new_edges.extend(self._create_edges_for_split_node(node, new_node, component_list))
        
        # Apply changes to graph
        self.graph.add_edges_from([edge for edge in new_edges if edge[0] is not None])
        self.graph.remove_node(node)
    
    def _create_split_node(self, original_node: str, new_node: str, component_list: List[str]) -> None:
        """Create a new node for a split component."""
        original_file_ids = self.graph.get_node_attributes(original_node, "file_id")
        new_file_ids = [fid for fid in original_file_ids.split(",") if fid in component_list]
        
        if not self.graph.has_node(new_node):
            self.graph.add_node(new_node)
        
        self.graph.update_node_attribute(new_node, "type", "named_entity")
        self.graph.update_node_attribute(new_node, "file_id", ",".join(new_file_ids))
    
    def _create_edges_for_split_node(self, original_node: str, new_node: str, 
                                     component_list: List[str]) -> List[Tuple[str, str, Dict]]:
        """Create edges for a split node."""
        new_edges = []
        
        # Outgoing edges
        for tail_node in list(self.graph.successors(original_node)):
            edge = self._build_new_edge(original_node, tail_node, new_node, component_list, outgoing=True)
            if edge[0] is not None:
                new_edges.append(edge)
        
        # Incoming edges
        for head_node in list(self.graph.predecessors(original_node)):
            edge = self._build_new_edge(head_node, original_node, new_node, component_list, outgoing=False)
            if edge[0] is not None:
                new_edges.append(edge)
        
        return new_edges
    
    def _build_new_edge(self, head: str, tail: str, new_node: str, component_list: List[str], 
                        outgoing: bool = True) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Build a new edge for a split node, filtering file_ids by component."""
        try:
            src, dst = (new_node, tail) if outgoing else (head, new_node)
            orig_src, orig_dst = head, tail
            
            # Filter relation file IDs
            relation_file_ids = self.graph.get_edge_attributes(orig_src, orig_dst, "file_id")
            new_relation_file_ids = [fid for fid in relation_file_ids.split(",") if fid in component_list]
            if not new_relation_file_ids:
                return None, None, None
            
            # Filter node file IDs
            node_to_check = tail if outgoing else head
            node_file_ids = self.graph.get_node_attributes(node_to_check, "file_id")
            new_node_file_ids = [fid for fid in node_file_ids.split(",") if fid in component_list]
            if not new_node_file_ids:
                return None, None, None
            
            return src, dst, {
                "relation": self.graph.get_edge_attributes(orig_src, orig_dst, "relation"),
                "file_id": ",".join(new_relation_file_ids)
            }
        except KeyError as e:
            logger.error(f"Key error: Attribute not found - {e}")
            return None, None, None
        except ValueError as e:
            logger.error(f"Value error: Invalid value encountered - {e}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error building edge: {e}")
            return None, None, None


class Disambiguation:
    """Handles named entity recognition and disambiguation in a knowledge graph using LLM-based prompts."""
    
    def __init__(self, graph: GraphStore, llm, graph_save_path: str, config: Optional[DisambiguationConfig] = None):
        """Initialize the Disambiguation class."""
        if not isinstance(graph, GraphStore):
            raise ValueError("graph must be an instance of GraphStore.")
        self.llm = llm
        self.model_name = llm.model_name
        self.graph = graph
        self.graph_save_path = graph_save_path
        self.config = config or DisambiguationConfig()
        
        # Initialize components
        self.context_manager = EntityContextManager()
        self.ner = NamedEntityRecognizer(llm, self.model_name, self.config.max_workers)
        self.disambiguator = EntityDisambiguator(llm, self.model_name, graph, self.config.max_workers)

    @staticmethod
    @validate_params(relations=dict(validator=lambda x: len(x) < 50000, message="Too many relations"))
    def extract_triples(relations: List, keys: Dict, triple_type: str) -> List[Tuple[str, str, str]]:
        """Extract triples from relation data."""
        triples = []
        try:
            for relation in relations:
                if triple_type == "event" and isinstance(relation, list) and relation:
                    relation = relation[0]
                if not isinstance(relation, dict):
                    logger.warning(f"Wrong relation: {relation}")
                    continue
                
                if triple_type == "entity":
                    head = relation.get(keys["head_entity"])
                    rel = relation.get(keys["relation"])
                    tail = relation.get(keys["tail_entity"])
                else:
                    head = relation.get(keys["head_event"])
                    rel = relation.get(keys["relation"])
                    tail = relation.get(keys["tail_event"])
                
                triples.append((head, rel, tail))
        except KeyError as e:
            logger.error(f"KeyError extracting triples {e}")
        except ValueError as e:
            logger.error(f"ValueError extracting triples {e}")
        except Exception as e:
            logger.error(f"Error extracting triples: {e}")
        
        return triples

    @validate_params(relationships=dict(validator=lambda x: len(x) < 50000, message="Too many relationships"))
    def run(self, relationships: List[Dict], lang: Lang = Lang.CH) -> None:
        """Run the disambiguation process on the provided relationships."""
        self._log_graph_statistics("Initial")
        
        # Extract context and process relationships
        self._extract_contexts(relationships, lang)
        
        # Get entity nodes and identify named entities
        entity_nodes = self.graph.get_nodes_by_attribute("type", "entity")
        logger.info(f"Number of entity nodes: {len(entity_nodes)}")
        
        named_entity_nodes = self.ner.identify_named_entities(entity_nodes, lang)
        logger.info(f"Number of named entity nodes: {len(named_entity_nodes)}")
        
        # Update node types
        for node in named_entity_nodes:
            self.graph.update_node_attribute(node, "type", "named_entity")
        
        # Disambiguate named entities
        logger.info("Disambiguating named entity nodes...")
        self.disambiguator.disambiguate_entities(named_entity_nodes, self.context_manager, lang)
        
        # Save results
        logger.info("Saving graph after disambiguation...")
        self._log_graph_statistics("Final")
        self.graph.save(self.graph_save_path)
    
    def _extract_contexts(self, relationships: List[Dict], lang: Lang) -> None:
        """Extract contexts from relationships."""
        keys = get_language_keys(lang)
        
        for relationship in relationships:
            file_id = relationship[FILE_ID_KEY]
            raw_text = relationship[RAW_TEXT_KEY]
            
            # Extract all triples
            entity_triples = self.extract_triples(relationship["entity_relations"], keys, "entity")
            event_triples = self.extract_triples(relationship["event_relations"], keys, "event")
            event_entity_triples = extract_event_entity_triples(relationship["event_entity_relations"], keys)
            
            # Extract context for all triples
            all_triples = entity_triples + event_triples + event_entity_triples
            for triple in all_triples:
                self.context_manager.extract_context_for_entity(triple, raw_text, file_id)
    
    def _log_graph_statistics(self, stage: str) -> None:
        """Log graph statistics."""
        logger.info(
            f"{stage} graph statistics - Nodes: {self.graph.number_of_nodes()}, "
            f"Edges: {self.graph.number_of_edges()}, "
            f"Density: {self.graph.density():.6f}"
        )
