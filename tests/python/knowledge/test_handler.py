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
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS

SQL_PATH = "./sql.db"


class TestHandler(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    white_paths = os.path.realpath(os.path.join(current_dir, "../../data/"))
    test_file = os.path.realpath(os.path.join(current_dir, "../../data/test.pdf"))
    test_folder = os.path.realpath(os.path.join(current_dir, "../../data/files/"))

    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    def test_Handler(self):
        loader_mng = LoaderMng()
        loader_mng.register_loader(DocxLoader, [".docx"])
        loader_mng.register_loader(PdfLoader, [".pdf"])

        loader_mng.register_splitter(RecursiveCharacterTextSplitter,
                                     [".docx", ".pdf"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        # 初始化向量数据库
        vector_store = MagicMock(spec=MindFAISS)
        vector_store.add = MagicMock(return_value=None)
        # 初始化文档chunk 关系数据库

        chunk_store = SQLiteDocstore(db_path=SQL_PATH)
        # 初始化知识管理关系数据库
        knowledge_store = KnowledgeStore(db_path=SQL_PATH)
        # 初始化知识管理
        knowledge_db = KnowledgeDB(knowledge_store=knowledge_store, chunk_store=chunk_store, vector_store=vector_store,
                                   knowledge_name="test001", white_paths=[self.white_paths])

        def embed_func(texts):
            embeddings = np.concatenate([np.random.random((1, 1024))])

            return [embeddings] * len(texts)

        upload_files(knowledge=knowledge_db, files=[self.test_file], loader_mng=loader_mng,
                     embed_func=embed_func, force=True)

        params = FilesLoadInfo(knowledge=knowledge_db, dir_path=self.test_folder, loader_mng=loader_mng,
                               embed_func=embed_func, force=True, load_image=False)
        upload_dir(params=params)
        delete_files(knowledge_db, ['test.pdf'])
