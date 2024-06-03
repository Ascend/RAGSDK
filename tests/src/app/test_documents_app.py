# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock


class TestDocumentApp(unittest.TestCase):
    def test_document_app(self):
        with patch("mx_rag.vectorstore.MindFAISS") as MindFAISS:
            from mx_rag.app.documents_app import DocumentApp
            total = np.random.random((3, 1024))
            query = np.array([total[0]])
            def embed_func(texts):
                if len(texts) > 1:
                    return total
                return query

            MindFAISS.set_device = MagicMock()
            current_dir = os.path.dirname(os.path.realpath(__file__))
            doc_app = DocumentApp("./sql.db", dev=0, embed_func=embed_func, )
            top_path = os.path.dirname(os.path.dirname(current_dir))
            doc_app.index_faiss.document_store.check_document_exist = MagicMock(return_value=False)
            doc_app.upload_file(os.path.join(top_path, "data/demo.docx"))
            doc_app.index_faiss.document_store.check_document_exist = MagicMock(return_value=True)
            doc_app.upload_file(os.path.join(top_path, "data/demo.docx"), force=True)
            doc_app.save_index("faiss.index")
            doc_app.index_faiss.document_store.check_document_exist = MagicMock(return_value=True)
            doc_app.delete_files(["demo.docx"])
            doc_app.index_faiss.document_store.check_document_exist.reset_mock()
            doc_app.index_faiss.document_store.check_document_exist = MagicMock(return_value=False)
            doc_app.upload_dir(os.path.join(top_path, "data"))


