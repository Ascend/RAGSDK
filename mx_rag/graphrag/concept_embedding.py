# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import List, Dict, Callable, Any, Set

ConceptData = List[Dict[str, Any]]
EmbeddingFunc = Callable[[List[str]], List[Any]]


class ConceptEmbedding:
    """
    Handles embeddings of conceptualized nodes using a provided embedding function.
    The embedding function should accept a list of concept strings and return a list of embeddings.
    """

    def __init__(self, embed_func: EmbeddingFunc) -> None:
        self._validate_callable(embed_func)
        self._embed_func = embed_func

    @staticmethod
    def _parse_conceptualized_nodes(concept_data: ConceptData) -> Set[str]:
        concepts = set()
        for item in concept_data:
            conceptualized_nodes = item.get("conceptualized_node", "")
            for node in str(conceptualized_nodes).split(","):
                node = node.strip()
                if node:
                    concepts.add(node)
        return concepts

    @staticmethod
    def _validate_callable(func: Callable) -> None:
        if not callable(func):
            raise ValueError("embed_func must be callable")

    def extract_concepts(self, concept_data: ConceptData) -> List[str]:
        return sorted(self._parse_conceptualized_nodes(concept_data))

    def embed(self, concept_data: ConceptData, batch_size=1) -> Dict[str, Any]:
        concepts = self.extract_concepts(concept_data)
        if not concepts:
            return {}
        embeddings = self._embed_func(concepts, batch_size=batch_size)
        if len(embeddings) != len(concepts):
            raise ValueError("Embedding function returned mismatched number of embeddings.")
        return dict(zip(concepts, embeddings))