# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Union, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1 import Field

from mx_rag.utils.common import validate_params, TEXT_MAX_LEN, MAX_TOP_K
from mx_rag.gvstore.graph_creator.graph_core import GraphNX


class GraphRetrieval(BaseRetriever):
    graph_name: str
    graph: GraphNX
    k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    khop: int = Field(default=2, ge=1, le=5)

    class Config:
        arbitrary_types_allowed = True

    # 根据问题召回相关原始文本块、图的实体节点，以及通过图算法召回相关多跳节点
    def retrieval(self, keywords: list, k: int, khop: int):
        nodes = []
        db_ids = []
        docs = []
        for keyword in keywords:
            ids, nodes = self.graph.search_indexes(keyword, k)
            db_ids.extend(ids)
            docs.extend(nodes)
        return self.graph.get_sub_graph(db_ids, nodes, khop)

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List:
        keywords = [query] if isinstance(query, str) else query
        retrieval_contexts = self.retrieval(keywords=keywords, k=self.k, khop=self.khop)
        return retrieval_contexts
