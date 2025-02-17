# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from sqlalchemy import URL
from sqlalchemy.engine import Engine

from mx_rag.storage.vectorstore import SearchMode
from mx_rag.storage.vectorstore.opengauss import OpenGaussDB, vector_model_factory


class TestVectorModelFactory(unittest.TestCase):

    def test_dense_model(self):
        table_name = "dense_table"
        model = vector_model_factory(table_name, SearchMode.DENSE, dim=128)
        self.assertEqual(model.__tablename__, table_name)
        self.assertTrue(hasattr(model, 'vector'))

    def test_sparse_model(self):
        table_name = "sparse_table"
        model = vector_model_factory(table_name, SearchMode.SPARSE, sparse_dim=128)
        self.assertEqual(model.__tablename__, table_name)
        self.assertTrue(hasattr(model, 'sparse_vector'))

    def test_hybrid_model(self):
        table_name = "hybrid_table"
        model = vector_model_factory(table_name, SearchMode.HYBRID, dim=128, sparse_dim=128)
        self.assertEqual(model.__tablename__, table_name)
        self.assertTrue(hasattr(model, 'vector'))
        self.assertTrue(hasattr(model, 'sparse_vector'))


class TestOpenGaussDB(unittest.TestCase):
    @patch('sqlalchemy.create_engine')
    def setUp(self, mock_create_engine):
        # Mock the engine and connection
        self.mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = self.mock_engine  # Make create_engine return the mock_engine

        self.url = URL.create(drivername="opengauss+psycopg2", database="test_db")

    @patch('mx_rag.storage.vectorstore.OpenGaussDB.create_collection')
    def test_create_success(self, mock_create_collection):
        mock_create_collection.return_value = None  # Mock the create_collection method

        db_instance = OpenGaussDB.create(url=self.url)

        self.assertIsNotNone(db_instance)
        self.assertEqual(db_instance.url, self.url)

    @patch('mx_rag.storage.vectorstore.OpenGaussDB.create_collection')
    def test_create_failure(self, mock_create_collection):
        mock_create_collection.side_effect = Exception("Creation failed")

        db_instance = OpenGaussDB.create(url=self.url)

        self.assertIsNone(db_instance)

    @patch('mx_rag.storage.vectorstore.OpenGaussDB._transaction')
    def test_create_collection_with_error(self, mock_transaction):
        mock_transaction.side_effect = Exception("Error during collection creation")
        db_instance = OpenGaussDB.create(url=self.url)

        with self.assertRaises(Exception):
            db_instance.create_collection(dense_dim=128, sparse_dim=128)

    @patch('mx_rag.storage.vectorstore.OpenGaussDB.create_collection')
    @patch.object(OpenGaussDB, '_internal_add')
    def test_add_dense(self, mock_internal_add, mock_create_collection):
        mock_create_collection.return_value = None

        embeddings = np.random.rand(10, 128)
        ids = list(range(10))

        db_instance = OpenGaussDB.create(url=self.url)
        db_instance.add(embeddings, ids)

        mock_internal_add.assert_called_once_with(ids, embeddings)

    @patch('mx_rag.storage.vectorstore.OpenGaussDB.create_collection')
    @patch.object(OpenGaussDB, '_internal_add')
    def test_add_sparse(self, mock_internal_add, mock_create_collection):
        mock_create_collection.return_value = None

        sparse_embeddings = [{i: np.random.rand()} for i in range(10)]
        ids = list(range(10))

        db_instance = OpenGaussDB.create(url=self.url, search_mode=SearchMode.SPARSE)
        db_instance.add_sparse(ids, sparse_embeddings)

        mock_internal_add.assert_called_once_with(ids, sparse=sparse_embeddings)

    @patch('mx_rag.storage.vectorstore.OpenGaussDB.create_collection')
    @patch.object(OpenGaussDB, '_internal_add')
    def test_add_dense_and_sparse(self, mock_internal_add, mock_create_collection):
        mock_create_collection.return_value = None

        embeddings = np.random.rand(10, 128)
        sparse_embeddings = [{i: np.random.rand()} for i in range(10)]
        ids = list(range(10))

        db_instance = OpenGaussDB.create(url=self.url, search_mode=SearchMode.HYBRID, dense_dim=128)
        db_instance.add_dense_and_sparse(ids, embeddings, sparse_embeddings)

        mock_internal_add.assert_called_once_with(ids, embeddings, sparse_embeddings)

    @patch('mx_rag.storage.vectorstore.OpenGaussDB.create_collection')
    @patch('mx_rag.storage.vectorstore.OpenGaussDB._parallel_search')
    def test_search(self, mock_parallel_search, mock_create_collection):
        mock_create_collection.return_value = None

        db_instance = OpenGaussDB.create(url=self.url)

        embeddings = np.random.rand(10, 128).tolist()
        k = 3

        mock_parallel_search.return_value = ([0.9] * k, [list(range(k))] * 10)  # Mocked return

        scores, ids = db_instance.search(embeddings, k)

        self.assertEqual(len(scores), 3)
        self.assertEqual(len(ids), 3)


if __name__ == "__main__":
    unittest.main()
