#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os

import numpy as np
import pytest
from pymilvus import MilvusClient

from mx_rag.graphrag.graph_rag_model import GraphRAGModel
from mx_rag.graphrag.graphrag_pipeline import GraphRAGPipeline
from mx_rag.graphrag.graphs.networkx_graph import NetworkxGraph
from mx_rag.graphrag.vector_stores.vector_store_wrapper import VectorStoreWrapper
from mx_rag.llm import LLMParameterConfig
from mx_rag.storage.document_store import MilvusDocstore
from mx_rag.storage.vectorstore import MilvusDB


pytestmark = pytest.mark.skipif(
    os.getenv("RAGSDK_ENABLE_MILVUS_LITE_ST") != "1",
    reason="Milvus Lite smoke depends on local Milvus server features; set RAGSDK_ENABLE_MILVUS_LITE_ST=1 to run it.",
)

TEXT_NODE = "混合检索可以通过节点文本命中知识图谱上下文"
VECTOR_NODE = "向量召回节点"
IRRELEVANT_NODE = "无关节点"


class LocalMockLLM:
    def chat(self, query, sys_messages=None, role="user", llm_config=None):
        if query.startswith("Extract all named entities"):
            return VECTOR_NODE
        return "本地Mock回答"


def deterministic_embed(texts, batch_size=4):
    embeddings = []
    for text in texts:
        vector = np.zeros(4, dtype=np.float32)
        if VECTOR_NODE in text or "向量问题" in text:
            vector[0] = 1.0
        elif "知识图谱上下文" in text or "混合检索" in text:
            vector[1] = 1.0
        else:
            vector[3] = 1.0
        embeddings.append(vector.tolist())
    return embeddings


def build_graph():
    graph = NetworkxGraph()
    graph.add_node(VECTOR_NODE, type="raw_text")
    graph.add_node(TEXT_NODE, type="raw_text")
    graph.add_node(IRRELEVANT_NODE, type="raw_text")
    graph.add_edge(VECTOR_NODE, TEXT_NODE, relation="text_conclude")
    return graph


def build_local_milvus_model(tmp_path, document_store=None, retrieval_mode="vector"):
    client = MilvusClient(str(tmp_path / "graph_text_retrieval_milvus.db"))
    milvus_db = MilvusDB.create(
        client=client,
        x_dim=4,
        collection_name="graph_text_retrieval_local",
        metric_type="COSINE",
    )
    assert milvus_db is not None

    return GraphRAGModel(
        llm=LocalMockLLM(),
        llm_config=LLMParameterConfig(),
        embed_func=deterministic_embed,
        graph_store=build_graph(),
        vector_store=VectorStoreWrapper(milvus_db),
        use_text=True,
        retrieval_top_k=3,
        subgraph_depth=1,
        retrieval_mode=retrieval_mode,
        document_store=document_store,
        batch_size=2,
    )


def build_local_milvus_docstore(tmp_path):
    client = MilvusClient(str(tmp_path / "graph_text_retrieval_docstore.db"))
    docstore = MilvusDocstore(client, collection_name="graph_text_retrieval_docstore")
    pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)
    pipeline.document_store = docstore
    pipeline.graph_name = "graph"
    pipeline.graph_node_document_id = 0
    pipeline._store_graph_nodes([VECTOR_NODE, TEXT_NODE, IRRELEVANT_NODE])
    return docstore


def test_local_milvus_lite_retrieval_and_context_generation(tmp_path):
    model = build_local_milvus_model(tmp_path)

    assert model.vector_store.ntotal() == 3

    vector_result = model.retrieve("向量问题", top_k=2, retrieval_mode="vector")
    assert vector_result[0] == VECTOR_NODE


def test_local_milvus_docstore_bm25_retrieval_and_context_generation(tmp_path):
    docstore = build_local_milvus_docstore(tmp_path)
    bm25_docs = docstore.full_text_search("知识图谱 上下文", top_k=2)
    assert bm25_docs
    assert bm25_docs[0].page_content == TEXT_NODE

    model = build_local_milvus_model(tmp_path, document_store=docstore, retrieval_mode="hybrid")

    text_result = model.retrieve("知识图谱 上下文", top_k=2, retrieval_mode="text")
    assert text_result == [TEXT_NODE]

    hybrid_result = model.retrieve("向量问题 知识图谱上下文", top_k=3, retrieval_mode="hybrid")
    assert VECTOR_NODE in hybrid_result
    assert TEXT_NODE in hybrid_result

    contexts = model.generate(["这个问题需要哪些上下文？"], retrieve_only=True)
    assert contexts == [[TEXT_NODE]]
