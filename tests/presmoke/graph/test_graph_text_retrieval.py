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

import unittest
from unittest.mock import Mock, patch

from mx_rag.graphrag.graph_rag_model import GraphRAGModel
from mx_rag.storage.document_store import MilvusDocstore, MxDocument


class TestGraphTextRetrievalPresmoke(unittest.TestCase):
    def setUp(self):
        self.mock_embed_func = Mock(return_value=[[0.1, 0.2, 0.3]])
        self.mock_vector_store = Mock()
        self.mock_vector_store.search.return_value = (None, [[0]])
        self.mock_graph_store = Mock()

        with patch.object(GraphRAGModel, '_initialize_databases'):
            self.model = GraphRAGModel(
                llm=Mock(),
                llm_config=Mock(),
                embed_func=self.mock_embed_func,
                graph_store=self.mock_graph_store,
                vector_store=self.mock_vector_store,
                use_text=True,
                retrieval_top_k=3,
                retrieval_mode="hybrid",
            )

        self.model.node_names = [
            "向量召回节点",
            "混合检索可以通过节点文本命中知识图谱上下文",
            "无关节点",
        ]
        self.model.text_nodes = list(self.model.node_names)
        self.document_store = Mock(spec=MilvusDocstore)
        self.document_store.full_text_search.return_value = [
            MxDocument(
                page_content="混合检索可以通过节点文本命中知识图谱上下文",
                metadata={"node_name": "混合检索可以通过节点文本命中知识图谱上下文"},
                document_name="graph.graph_nodes",
            )
        ]
        self.model.document_store = self.document_store

    def test_text_mode_retrieves_node_without_vector_call(self):
        result = self.model.retrieve("知识图谱 上下文", top_k=2, retrieval_mode="text")

        self.assertEqual(result[0], "混合检索可以通过节点文本命中知识图谱上下文")
        self.mock_embed_func.assert_not_called()
        self.mock_vector_store.search.assert_not_called()
        self.document_store.full_text_search.assert_called_once()

    def test_hybrid_mode_keeps_vector_result_and_adds_text_result(self):
        result = self.model.retrieve("知识图谱 上下文", top_k=3, retrieval_mode="hybrid")

        self.assertEqual(result[0], "向量召回节点")
        self.assertIn("混合检索可以通过节点文本命中知识图谱上下文", result)
        self.assertLessEqual(len(result), 3)

    def test_hybrid_mode_returns_text_result_when_vector_retrieval_fails(self):
        self.mock_embed_func.side_effect = RuntimeError("embedding service unavailable")

        result = self.model.retrieve("知识图谱 上下文", top_k=2, retrieval_mode="hybrid")

        self.assertEqual(result, ["混合检索可以通过节点文本命中知识图谱上下文"])
        self.document_store.full_text_search.assert_called_once()

    def test_hybrid_mode_returns_vector_result_when_text_retrieval_fails(self):
        self.document_store.full_text_search.side_effect = RuntimeError("BM25 service unavailable")

        result = self.model.retrieve("知识图谱 上下文", top_k=2, retrieval_mode="hybrid")

        self.assertEqual(result, ["向量召回节点"])
        self.mock_vector_store.search.assert_called_once()


if __name__ == '__main__':
    unittest.main()
