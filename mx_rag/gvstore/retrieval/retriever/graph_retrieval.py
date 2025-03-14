# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import concurrent
from concurrent.futures import as_completed
from typing import List
from typing import Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from loguru import logger

from mx_rag.gvstore.graph_creator.graph_core import GraphNX
from mx_rag.gvstore.graph_creator.nebula_graph import NebulaGraph
from mx_rag.gvstore.retrieval.preprocess.keywords_extract import KeywordsExtract
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params, TEXT_MAX_LEN, MAX_TOP_K


class GraphRetriever(BaseRetriever):
    graph_name: str
    graph: Union[GraphNX, NebulaGraph]
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    k_hop: int = Field(default=2, ge=1, le=5)
    llm: Text2TextLLM

    # 图及索引导入

    class Config:
        arbitrary_types_allowed = True

    def graph_search(self, question, k, **kwargs):
        ids, docs = self.search_nodes(question, **kwargs)
        return self.graph.get_sub_graph(ids, docs, k)

    def search_nodes(self, question, **kwargs):
        # search entities by keywords
        ids = []
        docs = []
        keyword_extract_flag = kwargs.get("keyword_extract", True)
        if keyword_extract_flag:
            keyword_extract = KeywordsExtract(self.llm)
            extracted_keywords = keyword_extract.extract_keywords(question)
            if extracted_keywords and extracted_keywords[0] != question:
                ids, docs = self.graph.get_nodes(extracted_keywords, **kwargs)
        return ids, docs

    # 根据问题召回相关原始文本块、图的实体节点，以及通过图算法召回相关多跳节点
    def retrieval(self, keywords: list, k: int, **kwargs):
        query_workers = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                              thread_name_prefix="query_workers")
        all_tasks = [query_workers.submit(self.graph_search, keywords[0], self.k_hop, **kwargs)]
        results = []
        for keyword in keywords:
            scores, ids, docs = self.graph.search_indexes(keyword, k)
            results.extend(docs)
        for future in as_completed(all_tasks):
            try:
                graph_context = future.result()
                results.extend(graph_context)
            except Exception as e:
                logger.error(f"document retrieval: graph search failed: {e}")
                continue
        return list(set(results))

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List:
        retrieval_contexts = self.retrieval(keywords=[query], k=self.top_k, khop=self.k_hop)
        return retrieval_contexts
