# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import inspect
import os
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import numpy as np
from loguru import logger

from mx_rag.document.loader import DocxLoader, ExcelLoader, PdfLoader
from mx_rag.document.splitter import CharTextSplitter, TextSplitterBase
from mx_rag.storage import SQLiteDocstore
from mx_rag.utils import FileCheck
from mx_rag.vectorstore import MindFAISS


class DocumentAppError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)


class DocumentApp:
    SUPPORT_IMAGE_TYPE = (".jpg", ".png")
    SUPPORT_DOCUMENT_TYPE = (".docx", ".xlsx", ".xls", ".csv", ".pdf")
    SUPPORT_LOADER = (DocxLoader, ExcelLoader, ExcelLoader, ExcelLoader, PdfLoader)
    LOADER_MAP = dict(zip(SUPPORT_DOCUMENT_TYPE, SUPPORT_LOADER))

    MAX_LOOP_LIMIT = 1000

    def __init__(
            self,
            db_path: str,
            dev: int,
            local_index_path: str,
            x_dim: int = 1024,
            index_type: str = "FLAT:L2",
            splitter: TextSplitterBase = CharTextSplitter()
    ):
        self.local_index_path = local_index_path
        FileCheck.check_input_path_valid(db_path)
        sql_db = SQLiteDocstore(db_path)
        if MindFAISS.DEVICES is None:
            MindFAISS.set_device(dev)
        if os.path.exists(self.local_index_path):
            FileCheck.check_input_path_valid(self.local_index_path)
            self.index_faiss = MindFAISS.load_local(self.local_index_path, sql_db)
        else:
            self.index_faiss = MindFAISS(x_dim, index_type, sql_db)

        self.splitter = splitter

    @staticmethod
    def parse_image(filepath:Path) -> Tuple[List[str], List[Dict[str, str]]]:
        return [filepath.as_posix()], [{"path": filepath.as_posix()}]

    def parse_document(self, filepath: Path) -> Tuple[List[str], List[Dict[str, str]]]:
        loader_cls = self.LOADER_MAP.get(filepath.suffix)
        if loader_cls is None:
            raise DocumentAppError(f"file type {filepath.suffix} is not support")
        metadatas = []
        texts = []
        for doc in loader_cls(filepath.as_posix()).load():
            split_texts = self.splitter.split_text(doc.page_content)
            metadatas.extend(doc.metadata for _ in split_texts)
            texts.extend(split_texts)
        return texts, metadatas


    def upload_files(self, files: List[str], embed_func: Callable[[List[str]], np.ndarray], force: bool = False):
        """上传单个文档，不支持的文件类型会抛出异常，如果文档重复，可选择强制覆盖"""
        if len(files) > self.MAX_LOOP_LIMIT:
            logger.error(f'files list length must less than {self.MAX_LOOP_LIMIT}, upload files failed')
            return

        for file in files:
            FileCheck.check_path_is_exist_and_valid(file)
            file_obj = Path(file)
            if not file_obj.is_file():
                raise DocumentAppError(f"{file} is not a normal file")
            if self.index_faiss.document_store.check_document_exist(file_obj.name):
                if not force:
                    raise DocumentAppError(f"file path {file_obj.name} is already exist")
                else:
                    self.delete_files([file_obj.name])

            if file_obj.suffix in self.SUPPORT_IMAGE_TYPE:
                texts, metadatas = self.parse_image(file_obj)
            else:
                texts, metadatas = self.parse_document(file_obj)

            self.index_faiss.add_texts(file_obj.name, texts, embed_func, metadatas)

            self.save_index(self.local_index_path)

    def upload_dir(self, dir_path, embed_func: Callable[[List[str]], np.ndarray], force=False, load_image=False):
        """
        只遍历当前目录下的文件，不递归查找子目录文件，目录中不支持的文件类型会跳过，如果文档重复，可选择强制覆盖，超过最大文件数量则退出
        load_image为True时导入支持的类型图片, False时支持导入支持的文档
        """
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            raise DocumentAppError(f"dir path {dir_path} is invalid")
        count = 0

        support_file_type = self.SUPPORT_DOCUMENT_TYPE
        if load_image:
            support_file_type = self.SUPPORT_IMAGE_TYPE

        for file in Path(dir_path).glob("*"):
            if count > self.MAX_LOOP_LIMIT:
                logger.warning("the number of files reaches the maximum limit")
                break
            if file.is_file() and file.suffix in support_file_type:
                self.upload_files([file.as_posix()], embed_func, force=force)
            count += 1

    def delete_files(self, file_names: List[str]):
        """删除上传的文档，需传入待删除的文档名称"""
        if not isinstance(file_names, list) or not file_names:
            raise DocumentAppError(f"files param {file_names} is invalid")

        count = 0
        for filename in file_names:
            if count > self.MAX_LOOP_LIMIT:
                logger.warning("the number of files reaches the maximum limit")
                break
            if not isinstance(filename, str):
                raise DocumentAppError(f"file path {filename} is invalid")
            if not self.index_faiss.document_store.check_document_exist(filename):
                raise DocumentAppError(f"file path {filename} is not exist")

            self.index_faiss.delete(filename)
            count += 1

    def save_index(self, index_path):
        FileCheck.check_input_path_valid(index_path)
        self.index_faiss.save_local(index_path)
