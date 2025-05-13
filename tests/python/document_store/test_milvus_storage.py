# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest
from unittest.mock import MagicMock, patch
from pymilvus import MilvusClient

from mx_rag.storage.document_store import MilvusDocstore, MxDocument


class TestMilvusDocStore(unittest.TestCase):

    def setUp(self):
        # Mock the MilvusDB and MxDocument classes
        self.mock_client = MagicMock(spec=MilvusClient)

        # Create instance of MilvusDocStore
        self.store = MilvusDocstore(self.mock_client, enable_bm25=False)
        self.store_bm25 = MilvusDocstore(self.mock_client, collection_name="doc_store_bm25")
        self.mock_client.has_collection.return_value = False
        self.mock_document = MagicMock(MxDocument)
        self.mock_document.metadata = {"chunk_id": 1, "document_id": 123, "document_name": "Test Document"}
        self.mock_document.page_content = "Some content here."
        self.mock_document.document_name = "Test Document"

    def test_initialization(self):
        # Test if the instance is initialized correctly
        self.assertEqual(self.store.client, self.mock_client)
        self.assertEqual(self.store.collection_name, "doc_store")

        self.assertEqual(self.store_bm25.client, self.mock_client)
        self.assertEqual(self.store_bm25.collection_name, "doc_store_bm25")

    def test_add_documents(self):
        documents = [self.mock_document]

        # Mock insert and refresh_load
        self.mock_client.insert.return_value = {"insert_count": 1, "ids": [1]}
        self.mock_client.refresh_load.return_value = None

        # Call the add method
        result = self.store.add(documents, 123)

        # Assertions
        self.assertEqual(result, [1])  # Check if the returned chunk_id is correct
        self.mock_client.query.return_value = [{"document_id": 123}]
        ids = self.store.get_all_document_id()
        self.assertEqual(ids, [123])
        self.mock_client.query.return_value = [{"id": 123}]
        ids = self.store.get_all_chunk_id()
        self.assertEqual(ids, [123])
        self.mock_client.insert.assert_called_once()  # Ensure insert was called
        self.mock_client.refresh_load.assert_called_once()  # Ensure refresh_load was called

    def test_add_documents_bm25(self):
        documents = [self.mock_document]

        # Mock insert and refresh_load
        self.mock_client.insert.return_value = {"insert_count": 1, "ids": [1]}
        self.mock_client.refresh_load.return_value = None

        # Call the add method
        result1 = self.store_bm25.add(documents, 123)
        # Assertions
        self.assertEqual(result1, [1])
        self.mock_client.insert.assert_called_once()  # Ensure insert was called
        self.mock_client.refresh_load.assert_called_once()  # Ensure refresh_load was called

    def test_add_invalid_documents(self):
        # Prepare invalid documents
        documents = ["Invalid document"]

        # Expect a ValueError due to invalid documents
        with self.assertRaises(ValueError):
            self.store.add(documents, 123)
            self.store_bm25.add(documents, 123)

    def test_search_found(self):
        self.mock_client.get.return_value = [
            {"id": 1, "page_content": "Some content here.",
             "document_name": "Test Document", "metadata": self.mock_document.metadata}]

        # Call search method
        result = self.store.search(1)
        result1 = self.store_bm25.search(1)

        # Assertions
        self.assertIsInstance(result, MxDocument)  # Ensure result is an MxDocument instance
        self.assertIsInstance(result1, MxDocument)
        self.assertEqual(result.page_content, "Some content here.")  # Check content is correct
        self.assertEqual(result1.page_content, "Some content here.")

    def test_search_not_found(self):
        # Mock client.get to return empty result
        self.mock_client.get.return_value = []

        # Call search method
        result = self.store.search(999)  # Search for a non-existent document
        result1 = self.store_bm25.search(999)
        # Assertions
        self.assertIsNone(result)  # Ensure result is None for non-existent document
        self.assertIsNone(result1)

    def test_delete(self):
        # Mock client.query to return sample IDs
        self.mock_client.query.return_value = [{"id": 1}]
        self.mock_client.delete.return_value = {"delete_count": 1}

        # Call delete method
        self.store.delete(123)

        # Assertions
        self.mock_client.delete.assert_called_once_with(self.store.collection_name, [1])
        self.mock_client.refresh_load.assert_called_once()
        self.mock_client.query.assert_called_once()  # Ensure query was called

    def test_delete_bm25(self):
        # Mock client.query to return sample IDs
        self.mock_client.query.return_value = [{"id": 1}]
        self.mock_client.delete.return_value = {"delete_count": 1}

        # Call delete method
        self.store_bm25.delete(123)

        # Assertions
        self.mock_client.delete.assert_called_once_with(self.store_bm25.collection_name, [1])
        self.mock_client.refresh_load.assert_called_once()
        self.mock_client.query.assert_called_once()  # Ensure query was called

    def test_delete_no_results(self):
        # Mock client.query to return empty result
        self.mock_client.query.return_value = []
        self.mock_client.delete.return_value = {}

        # Call delete method (nothing to delete)
        res = self.store.delete(999)
        res1 = self.store_bm25.delete(999)

        # Assertions
        self.mock_client.delete.assert_not_called()  # Ensure delete was not called
        self.assertEqual(res, [])
        self.assertEqual(res1, [])

    def test_full_text_search(self):
        self.mock_client.has_collection.return_value = False
        self.store._create_collection()
        self.mock_client.search.return_value = [[
            {
                "distance": 0.1,
                "entity":{
                    "metadata": {
                        "score": 0.1
                    },
                    "page_content": "Some content here.",
                    "document_name": "test.doc"
                }
            }
        ]]
        result = self.store_bm25.full_text_search("here")
        self.assertIsInstance(result[0], MxDocument)  # Ensure result is an MxDocument instance
        self.assertEqual(result[0].page_content, "Some content here.")  # Check content is correct

if __name__ == '__main__':
    unittest.main()
