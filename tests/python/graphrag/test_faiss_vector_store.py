# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import unittest
from unittest.mock import patch, MagicMock

from mx_rag.graphrag.vector_stores import FaissVectorStore


class TestFaissVectorStoreInit(unittest.TestCase):
    @patch("os.path.exists")
    @patch("faiss.read_index")
    @patch("faiss.IndexHNSWFlat")
    def test_init_indexhnswflat_with_existing_index(self, mock_hnswflat, mock_read_index, mock_exists):
        mock_exists.return_value = True
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index

        store = FaissVectorStore(128, "dummy_path", index_type="IndexHNSWFlat", M=32, efConstruction=200, efSearch=80)

        mock_read_index.assert_called_once_with("dummy_path")
        mock_hnswflat.assert_not_called()
        self.assertIs(store.index, mock_index)
        self.assertEqual(store.index_path, "dummy_path")
        self.assertEqual(store.index_type, "IndexHNSWFlat")
        self.assertEqual(store.metric_type, "IP")
        self.assertTrue(store.auto_save)
        self.assertEqual(store.index.hnsw.efConstruction, 200)
        self.assertEqual(store.index.hnsw.efSearch, 80)

    @patch("os.path.exists")
    @patch("faiss.read_index")
    @patch("faiss.IndexHNSWFlat")
    def test_init_indexhnswflat_without_existing_index(self, mock_hnswflat, mock_read_index, mock_exists):
        mock_exists.return_value = False
        mock_index = MagicMock()
        mock_hnswflat.return_value = mock_index

        store = FaissVectorStore(64, "dummy_path2", index_type="IndexHNSWFlat", M=8, efConstruction=50, efSearch=20)

        mock_read_index.assert_not_called()
        mock_hnswflat.assert_called_once_with(64, 8)
        self.assertIs(store.index, mock_index)
        self.assertEqual(store.index.hnsw.efConstruction, 50)
        self.assertEqual(store.index.hnsw.efSearch, 20)

    @patch("mx_rag.graphrag.vector_stores.faiss_vector_store.MindFAISS")
    def test_init_mindfaiss_flat(self, mock_mindfaiss):
        mock_index = MagicMock()
        mock_mindfaiss.return_value = mock_index

        store = FaissVectorStore(
            256, "dummy_path3", index_type="FLAT", metric_type="L2", auto_save=False, devs=[1, 2]
        )

        mock_mindfaiss.assert_called_once_with(
            x_dim=256,
            devs=[1, 2],
            load_local_index="dummy_path3",
            index_type="FLAT",
            metric_type="L2",
            auto_save=False,
        )
        self.assertIs(store.index, mock_index)
        self.assertEqual(store.index_type, "FLAT")
        self.assertEqual(store.metric_type, "L2")
        self.assertTrue(not store.auto_save)
        self.assertEqual(store.index_path, "dummy_path3")

    @patch("mx_rag.graphrag.vector_stores.faiss_vector_store.MindFAISS")
    def test_init_default_kwargs(self, mock_mindfaiss):
        mock_index = MagicMock()
        mock_mindfaiss.return_value = mock_index

        store = FaissVectorStore(32, "dummy_path4")

        mock_mindfaiss.assert_called_once_with(
            x_dim=32,
            devs=[0],
            load_local_index="dummy_path4",
            index_type="FLAT",
            metric_type="IP",
            auto_save=True,
        )
        self.assertIs(store.index, mock_index)
        self.assertEqual(store.index_type, "FLAT")
        self.assertEqual(store.metric_type, "IP")
        self.assertTrue(store.auto_save)
        self.assertEqual(store.index_path, "dummy_path4")
