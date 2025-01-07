# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pymilvus import MilvusClient

from mx_rag.storage.vectorstore.milvus import MilvusError
from mx_rag.storage.vectorstore.vectorstore import SimilarityStrategy, VectorStore
from mx_rag.storage.vectorstore import MilvusDB


class TestMlvusClient(unittest.TestCase):
    def test_faiss(self):
        with patch("pymilvus.MilvusClient") as MilvusClient:
            embeddings = np.random.random((3, 1024))
            query = embeddings[0]
            my_milvus = MagicMock()
            my_milvus.set_collection_name("test_ccc")
            my_milvus.client.has_collection = MagicMock(return_value=False)
            my_milvus.create_collection(768, "FLAT", "IP")
            my_milvus.client.has_collection = MagicMock(return_value=True)
            my_milvus.add(embeddings, [0, 1, 2])
            res = my_milvus.search(query)
            my_milvus.delete([0, 1, 2])
            my_milvus.drop_collection()

    def setUp(self):
        self.client = MagicMock(spec=MilvusClient)
        self.params = {
            'client': self.client,
            'x_dim': 1024,
            'similarity_strategy': SimilarityStrategy.FLAT_L2,
            'collection_name': "test_collection"
        }


    def create_milvus_db(self):
        milvus_db = MilvusDB.create(**self.params)
        return milvus_db

    def test_create_no_client(self):
        del self.params['client']
        milvus_db = MilvusDB.create(**self.params)
        self.assertIsNone(milvus_db)

    def test_create_no_x_dim(self):
        del self.params['x_dim']
        milvus_db = MilvusDB.create(**self.params)
        self.assertIsNone(milvus_db)

    def test_create_no_similarity_strategy(self):
        del self.params['similarity_strategy']
        milvus_db = MilvusDB.create(**self.params)
        self.assertIsNone(milvus_db)

    def test_create_wrong_params(self):
        milvus_db = MilvusDB.create(**self.params, params="params")
        self.assertIsNone(milvus_db)

    def test_create_success(self):
        milvus_db = MilvusDB.create(**self.params)
        self.assertIsInstance(milvus_db, MilvusDB)
        self.assertEqual(milvus_db._collection_name, self.params['collection_name'])

    def test_add_data(self):
        vecs = np.random.randn(3, 1024)
        self.create_milvus_db().add(vecs, [0, 1, 2])
        self.create_milvus_db().client.insert.assert_called_once()
        self.create_milvus_db().client.refresh_load.assert_called_once()

        with self.assertRaises(MilvusError):
            vecs = np.random.randn(3, 2, 1024)
            self.create_milvus_db().add(vecs, [0, 1, 2])

        with self.assertRaises(MilvusError):
            vecs = np.random.randn(2, 1024)
            self.create_milvus_db().add(vecs, [0, 1, 2])

        vecs = np.random.randn(3, 1024)
        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db().add(vecs, [0, 1, 2])

        with patch.object(VectorStore, 'MAX_VEC_NUM', 1):
            with self.assertRaises(MilvusError):
                self.create_milvus_db().add(vecs, [0, 1, 2])

    def test_delete_data(self):
        vecs = np.random.randn(3, 1024)
        self.create_milvus_db().add(vecs, [0, 1, 2])
        with patch.object(VectorStore, 'MAX_VEC_NUM', 1):
            with self.assertRaises(MilvusError):
                self.create_milvus_db().delete([1, 2])

        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db().delete([0])

        with self.assertRaises(ValueError):
            self.create_milvus_db().delete(['0'])

        self.create_milvus_db().client.delete.return_value = {'delete_count': len(vecs)}
        self.create_milvus_db().client.has_collection.return_value = True
        result = self.create_milvus_db().delete([0])
        self.assertEqual(result, len(vecs))

    def test_get_all_ids(self):
        ids = self.create_milvus_db().get_all_ids()
        self.assertEqual(ids, [])

    def test_drop_collection(self):
        self.create_milvus_db().drop_collection()
        self.create_milvus_db().client.drop_collection.assert_called_once_with(self.create_milvus_db()._collection_name)

        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db().drop_collection()

    def test_search(self):
        with patch('mx_rag.storage.vectorstore.vectorstore.VectorStore._score_scale') as mock_score_scale:
            mock_score_scale.return_value = [1, 2, 3]
            self.client.search.return_value = [
                [{'distance': 0.1, 'id': 1}, {'distance': 0.2, 'id': 2}, {'distance': 0.3, 'id': 3}],
                [{'distance': 0.4, 'id': 4}, {'distance': 0.5, 'id': 5}, {'distance': 0.6, 'id': 6}]
            ]
            embedding = np.array([[1, 2, 3], [4, 5, 6]])
            scores, ids = self.create_milvus_db().search(embedding, 3)
            self.assertEqual(scores, [1, 2, 3])
            self.assertEqual(ids, [[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(MilvusError):
            embedding = np.random.randn(3, 2, 1024)
            self.create_milvus_db().search(embedding, 3)

        with patch.object(VectorStore, 'MAX_SEARCH_BATCH', 1):
            with self.assertRaises(MilvusError):
                self.create_milvus_db().search(np.array([[1, 2], [4, 5]]))

        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db().search(np.array([[1, 2]]))


