# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import inspect
from pathlib import Path
from typing import List, Dict, Tuple

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
    SUPPORT_TYPE = (".docx", ".xlsx", ".xls", ".csv", ".pdf")
    SUPPORT_LOADER = (DocxLoader, ExcelLoader, ExcelLoader, ExcelLoader, PdfLoader)
    LOADER_MAP = dict(zip(SUPPORT_TYPE, SUPPORT_LOADER))

    MAX_LOOP_LIMIT = 1000

    def __init__(
            self,
            db_path: str,
            dev: int,
            embed_func,
            x_dim: int = 1024,
            index_type: str = "FLAT:L2",
            auto_save_path: str = None,
            load_local_index: bool = False,
            local_index_path: str = None,
            splitter: TextSplitterBase = CharTextSplitter()
    ):
        FileCheck.check_input_path_valid(db_path)
        sql_db = SQLiteDocstore(db_path)
        MindFAISS.set_device(dev)
        if load_local_index:
            FileCheck.check_path_is_exist_and_valid(local_index_path)
            self.index_faiss = MindFAISS.load_local(local_index_path, sql_db, embed_func)
        else:
            self.index_faiss = MindFAISS(x_dim, index_type, sql_db, embed_func)
        self.auto_save_path = auto_save_path
        self.splitter = splitter

    def upload_file(self, filepath: str, force: bool = False):
        """上传单个文档，不支持的文件类型会抛出异常，如果文档重复，可选择强制覆盖"""
        FileCheck.check_path_is_exist_and_valid(filepath)
        file_obj = Path(filepath)
        if not file_obj.is_file():
            raise DocumentAppError(f"{filepath} is not a normal file")
        if self.index_faiss.document_store.check_document_exist(file_obj.name):
            if not force:
                raise DocumentAppError(f"file path {file_obj.name} is already exist")
            else:
                self.delete_files([file_obj.name])

        texts, metadatas = self.parse_document(file_obj)
        self.index_faiss.add_texts(file_obj.name, texts, metadatas)
        if self.auto_save_path is not None:
            self.save_index(self.auto_save_path)

    def upload_dir(self, dir_path, force=False):
        """只遍历当前目录下的文件，不递归查找子目录文件，目录中不支持的文件类型会跳过，如果文档重复，可选择强制覆盖，超过最大文件数量则退出"""
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            raise DocumentAppError(f"dir path {dir_path} is invalid")
        count = 0
        for file in Path(dir_path).glob("*"):
            if count > self.MAX_LOOP_LIMIT:
                logger.warning("the number of files reaches the maximum limit")
                break
            if file.is_file() and file.suffix in self.SUPPORT_TYPE:
                self.upload_file(file.as_posix(), force=force)
            count += 1

    def delete_files(self, files: List[str]):
        """删除上传的文档，需传入待删除的文档名称"""
        if not isinstance(files, list) or not files:
            raise DocumentAppError(f"files param {files} is invalid")

        count = 0
        for filename in files:
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
