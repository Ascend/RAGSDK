# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import unittest
from unittest.mock import patch, MagicMock

from sqlalchemy import URL, Engine

from mx_rag.storage.document_store.base_storage import MxDocument
from mx_rag.storage.document_store.helper_storage import _DocStoreHelper
from mx_rag.storage.document_store.opengauss_storage import OpenGaussDocstore
from mx_rag.utils.common import MAX_CHUNKS_NUM


class TestOpenGaussDocstore(unittest.TestCase):
    @patch("mx_rag.storage.document_store.opengauss_storage._DocStoreHelper")
    @patch('sqlalchemy.create_engine')
    def setUp(self, mock_create_engine, MockDocStoreHelper):
        # Mock the engine and connection
        self.mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = self.mock_engine  # Make create_engine return the mock_engine
        self.mock_helper = MagicMock(spec=_DocStoreHelper)  # mock HelperDocStore
        MockDocStoreHelper.return_value = self.mock_helper
        self.docstore = OpenGaussDocstore(self.mock_engine)
        self.test_documents = [
            MxDocument(page_content="content1", document_name="test1.docx", metadata={"key": "value1"}),
            MxDocument(page_content="content2", document_name="test1.docx", metadata={"key": "value2"}),
        ]

    def test_add_documents(self):
        doc_id = 1
        expected_ids = [1, 2]
        self.mock_helper.add.return_value = expected_ids

        returned_ids = self.docstore.add(self.test_documents, doc_id)

        self.assertEqual(returned_ids, expected_ids)
        self.mock_helper.add.assert_called_once_with(self.test_documents, doc_id)

    def test_add_documents_invalid_input(self):
        with self.assertRaises(ValueError):
            self.docstore.add([1, 2, 3], 1)  # Invalid document type

        with self.assertRaises(ValueError):
            self.docstore.add([], 1)  # Empty list

        with self.assertRaises(ValueError):
            self.docstore.add([MxDocument(page_content="test")] * (MAX_CHUNKS_NUM + 1), 1)  # Too many documents

    def test_delete_documents(self):
        doc_id = 1
        expected_ids = [1, 2]
        self.mock_helper.delete.return_value = expected_ids

        returned_ids = self.docstore.delete(doc_id)

        self.assertEqual(returned_ids, expected_ids)
        self.mock_helper.delete.assert_called_once_with(doc_id)

    def test_search_document(self):
        chunk_id = 1
        expected_doc = MxDocument(page_content="test", document_name="test1.docx", metadata={})
        self.mock_helper.search.return_value = expected_doc

        returned_doc = self.docstore.search(chunk_id)

        self.assertEqual(returned_doc, expected_doc)
        self.mock_helper.search.assert_called_once_with(chunk_id)

    def test_search_document_invalid_input(self):
        with self.assertRaises(ValueError):
            self.docstore.search(-1)

    def test_get_all_index_id(self):
        expected_ids = [1, 2, 3]
        self.mock_helper.get_all_index_id.return_value = expected_ids

        returned_ids = self.docstore.get_all_index_id()

        self.assertEqual(returned_ids, expected_ids)
        self.mock_helper.get_all_index_id.assert_called_once()

    def test_init_invalid_params(self):
        with self.assertRaises(ValueError):  # Invalid URL - string, not URL object
            OpenGaussDocstore("not a engine")

        with self.assertRaises(ValueError):
            OpenGaussDocstore(
                self.mock_engine, encrypt_fn=123
            )  # Invalid encrypt_fn

        with self.assertRaises(ValueError):
            OpenGaussDocstore(
                self.mock_engine, decrypt_fn=123
            )  # Invalid decrypt_fn


if __name__ == "__main__":
    unittest.main()
