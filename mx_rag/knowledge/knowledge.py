# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Any, NoReturn

import numpy as np
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

from mx_rag.document import parse_file
from mx_rag.knowledge.base_knowledge import KnowledgeBase, KnowledgeError
from mx_rag.storage import Docstore, Document
from mx_rag.utils import FileCheck
from mx_rag.vectorstore import VectorStore

Base = declarative_base()


class KnowledgeMgrModel(Base):
    __tablename__ = "knowledgeMgr_table"

    id = Column(Integer, primary_key=True)
    knowledge_name = Column(String, comment="知识库名称", unique=True)


class KnowledgeModel(Base):
    __tablename__ = "knowledge_table"

    id = Column(Integer, primary_key=True)
    knowledge_name = Column(String, comment="知识库名称")
    document_name = Column(String, comment="文档名称", unique=True)


class Knowledge(KnowledgeBase):
    SUPPORT_IMAGE_TYPE = (".jpg", ".png")
    SUPPORT_DOC_TYPE = (".docx", ".xlsx", ".xls", ".csv", ".pdf")

    def __init__(
            self,
            db_path: str,
            document_store: Docstore,
            vector_store: VectorStore,
            knowledge_name: str,
            white_paths: List[str],
            max_loop_limit: int = 1000,
            parse_func: Callable[[str], Tuple[List[str], List[Dict[str, str]]]] = parse_file
    ):
        super().__init__(white_paths)
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

        self._vector_store = vector_store
        self._document_store = document_store
        self._max_loop_limit = max_loop_limit
        self.knowledge_name = knowledge_name
        self._parse_func = parse_func

    def register_parse_func(self, parse_func):
        self._parse_func = parse_func

    def upload_files(self, files: List[str], embed_func: Callable[[List[str]], np.ndarray],
                     force: bool = False, *args, **kwargs):
        """上传单个文档，不支持的文件类型会抛出异常，如果文档重复，可选择强制覆盖"""
        if len(files) > self._max_loop_limit:
            raise KnowledgeError(f'files list length must less than {self._max_loop_limit}, upload files failed')

        for file in files:
            FileCheck.check_path_is_exist_and_valid(file)
            file_obj = Path(file)
            for p in self.white_paths:
                if not file_obj.absolute().is_relative_to(p):
                    raise KnowledgeError(f"{file_obj.as_posix()} is not in whitelist path")
            if not file_obj.is_file():
                raise KnowledgeError(f"{file} is not a normal file")

            if self._check_document_exist(file_obj.name):
                if not force:
                    raise KnowledgeError(f"file path {file_obj.name} is already exist")
                else:
                    self._delete(file_obj.name)

            texts, metadatas = self._parse_func(file_obj.as_posix())
            try:
                self._add_texts(file_obj.name, texts, embed_func, metadatas)
            except Exception as err:
                # 当添加文档失败时，删除已添加的部分文档做回滚，捕获异常是为了正常回滚
                try:
                    self._delete(file_obj.name)
                except Exception as e:
                    logger.warning(f"exception encountered while rollback, {e}")
                raise KnowledgeError(f"add {file_obj.name} failed, {err}") from err

    def upload_dir(self, dir_path, embed_func: Callable[[List[str]], np.ndarray],
                   force=False, load_image=False, *args, **kwargs):
        """
        只遍历当前目录下的文件，不递归查找子目录文件，目录中不支持的文件类型会跳过，如果文档重复，可选择强制覆盖，超过最大文件数量则退出
        load_image为True时导入支持的类型图片, False时支持导入支持的文档
        """
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.is_dir():
            raise KnowledgeError(f"dir path {dir_path} is invalid")
        count = 0

        support_file_type = self.SUPPORT_DOC_TYPE
        if load_image:
            support_file_type = self.SUPPORT_IMAGE_TYPE

        for file in Path(dir_path).glob("*"):
            if count > self._max_loop_limit:
                logger.warning("the number of files reaches the maximum limit")
                break
            if file.is_file() and file.suffix in support_file_type:
                self.upload_files([file.as_posix()], embed_func, force=force)
            count += 1

    def delete_files(self, file_names: List[str], *args, **kwargs):
        """删除上传的文档，需传入待删除的文档名称"""
        if not isinstance(file_names, list) or not file_names:
            raise KnowledgeError(f"files param {file_names} is invalid")

        count = 0
        for filename in file_names:
            if not isinstance(filename, str):
                raise KnowledgeError(f"file path {filename} is invalid")
            if count > self._max_loop_limit:
                logger.warning("the number of files reaches the maximum limit")
                break
            if not self._check_document_exist(filename):
                continue
            self._delete(filename)
            count += 1

    def get_all_documents(self, *args, **kwargs):
        """获取当前已上传的所有文档"""
        with self.session() as session:
            ret = []
            for doc in session.query(
                    KnowledgeModel.document_name
            ).filter_by(knowledge_name=self.knowledge_name).distinct().all():
                ret.append(doc[0])
            return ret

    def _add_texts(
            self,
            doc_name: str,
            texts: List[str],
            embed_func: Callable[[List[str]], np.ndarray],
            metadatas: Optional[List[dict]] = None,
            *args,
            **kwargs,
    ) -> NoReturn:
        embeddings = embed_func(texts)
        if not isinstance(embeddings, np.ndarray):
            raise KnowledgeError("The data type of embedding should be np.ndarray")
        metadatas = metadatas or [{} for _ in texts]
        if len(texts) != len(metadatas) != len(embeddings):
            raise KnowledgeError("texts, metadatas, embeddings expected to be equal length")
        documents = [Document(page_content=t, metadata=m, document_name=doc_name) for t, m in zip(texts, metadatas)]
        with self.session() as session:
            try:
                session.add(KnowledgeModel(knowledge_name=self.knowledge_name, document_name=doc_name))
                idxs = self._document_store.add(documents, session)
                self._vector_store.add(embeddings, np.array(idxs))
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add chunk failed, {err}") from err

    def _delete(self, doc_name: str, **kwargs: Any):
        with self.session() as session:
            try:
                session.query(KnowledgeModel).filter_by(
                    knowledge_name=self.knowledge_name, document_name=doc_name).delete()
                ids = self._document_store.delete(doc_name, session)
                num_removed = self._vector_store.delete(ids)
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

            if len(ids) != num_removed:
                logger.warning("the number of documents does not match the number of vectors")

    def _check_document_exist(self, doc_name: str) -> bool:
        with self.session() as session:
            chunk = session.query(KnowledgeModel).filter_by(
                knowledge_name=self.knowledge_name, document_name=doc_name).first()
            return True if chunk is not None else False


class KnowledgeBase:

    def __init__(self, db_path: str):
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    def register(self, knowledge: Knowledge):
        if knowledge.knowledge_name in self.get_all():
            raise KnowledgeError(f"knowledge {knowledge.knowledge_name} has been registered")

        with self.session() as session:
            try:
                session.add(KnowledgeMgrModel(knowledge_name=knowledge.knowledge_name))
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add knowledge failed, {err}") from err

    def delete(self, knowledge: Knowledge):
        if knowledge.knowledge_name not in self.get_all():
            raise KnowledgeError(f"knowledge {knowledge.knowledge_name} is not be registered")
        if knowledge.get_all_documents():
            raise KnowledgeError(f"please clear knowledge, before delete")
        with self.session() as session:
            try:
                session.query(KnowledgeMgrModel).filter_by(knowledge_name=knowledge.knowledge_name).delete()
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

    def get_all(self):
        with self.session() as session:
            names = []
            for k in session.query(KnowledgeMgrModel).all():
                names.append(k.knowledge_name)
            return names
