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
from mx_rag.document.loader import DocxLoader, PdfLoader, ImageLoader
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
    not_white_paths = os.path.realpath(os.path.join(current_dir, "../../python/knowledge/test_handler.py"))
    test_file = os.path.realpath(os.path.join(current_dir, "../../data/test.pdf"))
    test_png = os.path.realpath(os.path.join(current_dir, "../../data/test.png"))
    test_folder = os.path.realpath(os.path.join(current_dir, "../../data/files/"))
    loader_mng = LoaderMng()
    loader_mng.register_loader(DocxLoader, [".docx"])
    loader_mng.register_loader(PdfLoader, [".pdf"])
    loader_mng.register_loader(ImageLoader, [".png"])

    loader_mng.register_splitter(RecursiveCharacterTextSplitter,
                                 [".docx", ".pdf"],
                                 {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})

    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

        # 初始化参数
        self.knowledge_db = self.create_knowledge_db()
        self.common_params = {
            'knowledge': self.knowledge_db,
            'loader_mng': self.loader_mng,
            'embed_func': embed_func,
            'force': True
        }

    def create_knowledge_db(self, knowledge_name="test001"):
        vector_store = MagicMock(spec=MindFAISS)
        vector_store.add = MagicMock(return_value=None)
        chunk_store = SQLiteDocstore(db_path=SQL_PATH)
        knowledge_store = KnowledgeStore(db_path=SQL_PATH)
        return KnowledgeDB(knowledge_store=knowledge_store, chunk_store=chunk_store, vector_store=vector_store,
                           knowledge_name=knowledge_name, white_paths=[self.white_paths])

    def test_upload_with_invalid_knowledge(self):
        self.common_params['knowledge'] = None
        with self.assertRaises(ValueError):
            upload_files(**self.common_params, files=[self.test_file])
        with self.assertRaises(ValueError):
            params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=False)
            upload_dir(params=params)
        with self.assertRaises(ValueError):
            delete_files(None, ['test.pdf'])

    def test_upload_with_invalid_file_paths(self):
        with self.assertRaises(FileCheckError):
            upload_files(**self.common_params, files=['/test/test.docx' * 100])
        with self.assertRaises(FileCheckError):
            upload_files(**self.common_params, files=[self.not_white_paths])
        with self.assertRaises(ValueError):
            params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder * 100, load_image=False)
            upload_dir(params=params)
        with self.assertRaises(ValueError):
            delete_files(self.knowledge_db, ['test.pdf' * 200])
        with self.assertRaises(FileHandlerError):
            delete_files(self.knowledge_db, ['test123.pdf', 123])
        with self.assertRaises(FileHandlerError):
            delete_files(self.knowledge_db, 'test123.pdf')

    def test_with_too_many_files(self):
        knowledge_db = self.create_knowledge_db()
        knowledge_db.max_file_count = 1
        self.common_params['knowledge'] = knowledge_db
        with self.assertRaises(FileHandlerError):
            upload_files(**self.common_params, files=[self.test_file, self.test_file])
        with self.assertRaises(FileHandlerError):
            params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=False)
            upload_dir(params=params)
        with self.assertRaises(FileHandlerError):
            delete_files(knowledge_db, ['file1', 'file2'])

    def test_with_no_files(self):
        res1 = upload_files(**self.common_params, files=[])
        self.assertEqual(res1, [])
        res2 = delete_files(self.knowledge_db, [])
        self.assertEqual(res2, None)

    def test_upload_with_invalid_loader(self):
        self.common_params['loader_mng'] = None
        with self.assertRaises(ValueError):
            upload_files(**self.common_params, files=[self.test_file])
        with self.assertRaises(ValueError):
            params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=False)
            upload_dir(params=params)

    def test_upload_with_invalid_embed_func(self):
        self.common_params['embed_func'] = None
        with self.assertRaises(ValueError):
            upload_files(**self.common_params, files=[self.test_file])
        with self.assertRaises(ValueError):
            params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=False)
            upload_dir(params=params)

    def test_upload_with_add_file_failure(self):
        with patch('mx_rag.knowledge.KnowledgeDB.add_file') as mock_add_file:
            mock_add_file.side_effect = Exception('Add file failed')
            result = upload_files(**self.common_params, files=[self.test_file])
            self.assertEqual(result, [self.test_file])

            params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=False)
            result = upload_dir(params=params)
            self.assertEqual(len(result), 3)

    def test_upload_image(self):
        res1 = upload_files(**self.common_params, files=[self.test_png])
        self.assertEqual(res1, [])
        params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=True)
        res2 = upload_dir(params=params)
        self.assertEqual(len(res2), 2)

    def test_upload_success(self):
        res1 = upload_files(**self.common_params, files=[self.test_file])
        self.assertEqual(res1, [])
        params = FilesLoadInfo(**self.common_params, dir_path=self.test_folder, load_image=False)
        res2 = upload_dir(params=params)
        self.assertEqual(len(res2), 1)

    def test_upload_with_not_force(self):
        upload_files(**self.common_params, files=[self.test_file])
        self.common_params['force'] = False
        with self.assertRaises(FileHandlerError):
            upload_files(**self.common_params, files=[self.test_file])

    def test_delete_files_success(self):
        upload_files(**self.common_params, files=[self.test_file])
        delete_files(self.knowledge_db, ['test.pdf'])
        res = self.knowledge_db.check_document_exist('test.pdf')
        self.assertEqual(res, False)
