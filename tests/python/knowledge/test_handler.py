# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from paddle.base import libpaddle
import pathlib
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from mx_rag.knowledge import KnowledgeStore, KnowledgeDB, upload_files, FilesLoadInfo, delete_files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.knowledge import upload_dir
from mx_rag.document.loader import DocxLoader, PdfLoader, ExcelLoader
from mx_rag.document import LoaderMng
from mx_rag.knowledge.handler import FileHandlerError
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.utils.file_check import FileCheckError

SQL_PATH = "./sql.db"


def embed_func(texts):
    embeddings = np.concatenate([np.random.random((1, 1024))])

    return [embeddings] * len(texts)


class TestHandler(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    white_paths = os.path.realpath(os.path.join(current_dir, "../../data/"))
    test_file = os.path.realpath(os.path.join(current_dir, "../../data/test.pdf"))
    test_folder = os.path.realpath(os.path.join(current_dir, "../../data/files/"))

    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    def create_knowledge_db(self, knowledge_name="test001"):
        loader_mng = LoaderMng()
        loader_mng.register_loader(DocxLoader, [".docx"])
        loader_mng.register_loader(PdfLoader, [".pdf"])

        loader_mng.register_splitter(RecursiveCharacterTextSplitter,
                                     [".docx", ".pdf"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        vector_store = MagicMock(spec=MindFAISS)
        vector_store.add = MagicMock(return_value=None)
        chunk_store = SQLiteDocstore(db_path=SQL_PATH)
        knowledge_store = KnowledgeStore(db_path=SQL_PATH)
        return KnowledgeDB(knowledge_store=knowledge_store, chunk_store=chunk_store, vector_store=vector_store,
                           knowledge_name=knowledge_name, white_paths=[self.white_paths])

    def test_upload_files_with_invalid_knowledge(self):
        with self.assertRaises(ValueError):
            upload_files(knowledge=None, files=[self.test_file], loader_mng=LoaderMng(),
                         embed_func=embed_func, force=True)

    def test_upload_files_with_invalid_file_paths(self):
        knowledge_db = self.create_knowledge_db()
        with self.assertRaises(FileCheckError):
            upload_files(knowledge=knowledge_db, files=['/test/test.docx' * 100], loader_mng=LoaderMng(),
                         embed_func=embed_func, force=True)

    def test_upload_files_with_too_many_files(self):
        knowledge_db = self.create_knowledge_db()
        knowledge_db.max_file_count = 1
        with self.assertRaises(FileHandlerError):
            upload_files(knowledge=knowledge_db, files=[self.test_file, self.test_file], loader_mng=LoaderMng(),
                         embed_func=embed_func, force=True)

    def test_upload_files_with_invalid_loader(self):
        knowledge_db = self.create_knowledge_db()
        with self.assertRaises(ValueError):
            upload_files(knowledge=knowledge_db, files=[self.test_file], loader_mng=None,
                         embed_func=embed_func, force=True)

    def test_upload_files_with_invalid_embed_func(self):
        knowledge_db = self.create_knowledge_db()
        with self.assertRaises(ValueError):
            upload_files(knowledge=knowledge_db, files=[self.test_file], loader_mng=LoaderMng(),
                         embed_func=None, force=True)

    def test_upload_files_with_add_file_failure(self):
        knowledge_db = self.create_knowledge_db()
        with patch('mx_rag.knowledge.KnowledgeDB.add_file') as mock_add_file:
            mock_add_file.side_effect = Exception('Add file failed')
            result = upload_files(knowledge=knowledge_db, files=[self.test_file], loader_mng=LoaderMng(),
                                  embed_func=embed_func, force=True)
            self.assertEqual(result, self.test_file)

    def test_upload_files_success(self):
        knowledge_db = self.create_knowledge_db()
        result = upload_files(knowledge=knowledge_db, files=[self.test_file], loader_mng=LoaderMng(),
                              embed_func=embed_func, force=True)
        self.assertEqual(result, [])

    def test_upload_dir(self):
        knowledge_db = self.create_knowledge_db()
        params = FilesLoadInfo(knowledge=knowledge_db, dir_path=self.test_folder, loader_mng=LoaderMng(),
                               embed_func=embed_func, force=True, load_image=False)
        upload_dir(params=params)

    def test_delete_files(self):
        knowledge_db = self.create_knowledge_db()
        delete_files(knowledge_db, ['test.pdf'])
