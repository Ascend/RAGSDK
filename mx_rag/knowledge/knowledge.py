# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
from typing import List, Callable, Optional, NoReturn

import numpy as np
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

from mx_rag.knowledge.base_knowledge import KnowledgeBase, KnowledgeError
from mx_rag.storage.document_store.base_storage import Docstore, MxDocument
from mx_rag.storage.vectorstore import VectorStore
from mx_rag.utils.common import validate_params, INT_32_MAX, FILE_COUNT_MAX, \
    check_db_file_limit, validata_list_str, TEXT_MAX_LEN, STR_TYPE_CHECK_TIP_1024
from mx_rag.utils.file_check import FileCheck, check_disk_free_space

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


class KnowledgeStore:
    FREE_SPACE_LIMIT = 200 * 1024 * 1024

    @validate_params(
        db_path=dict(validator=lambda x: 0 < len(x) <= 1024 and isinstance(x, str), message=STR_TYPE_CHECK_TIP_1024)
    )
    def __init__(self, db_path: str):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def add(self, knowledge_name: str, doc_name: str):
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            logger.error("Insufficient remaining space. Please clear disk space.")
            raise KnowledgeError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        with self.session() as session:
            try:
                session.add(KnowledgeModel(knowledge_name=knowledge_name, document_name=doc_name))
                session.commit()
                logger.debug(f"success add (knowledge_name={knowledge_name}, "
                             f"doc_name={doc_name}) in knowledge_table.")
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(
                    f"Database error while adding knowledge: '{knowledge_name}', document: '{doc_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to add knowledge: '{knowledge_name}', document: '{doc_name}' "
                    "due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add chunk failed, {err}") from err

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def delete(self, knowledge_name: str, doc_name: str):
        with self.session() as session:
            try:
                doc_to_delete = session.query(KnowledgeModel).filter_by(
                    knowledge_name=knowledge_name, document_name=doc_name).first()
                if not doc_to_delete:
                    logger.debug(f"{doc_name} does not exist in {knowledge_name}, no need delete.")
                else:
                    session.delete(doc_to_delete)
                    session.commit()
                    logger.debug(f"success delete (knowledge_name={knowledge_name}, "
                                 f"doc_name={doc_name}) in knowledge_table.")
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(
                    f"Database error while deleting knowledge: '{knowledge_name}', document: '{doc_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to delete knowledge: '{knowledge_name}', document: '{doc_name}' "
                    "due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def get_all(self, knowledge_name: str):
        with self.session() as session:
            ret = []
            for doc in session.query(
                    KnowledgeModel.document_name
            ).filter_by(knowledge_name=knowledge_name).distinct().all():
                ret.append(doc[0])
            return ret

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def check_document_exist(self, knowledge_name: str, doc_name: str) -> bool:
        with self.session() as session:
            chunk = session.query(KnowledgeModel).filter_by(
                knowledge_name=knowledge_name, document_name=doc_name).first()
            return True if chunk is not None else False


def _check_metadatas(metadatas: List[dict] = None) -> bool:
    if metadatas is None:
        return True
    if not isinstance(metadatas, list) or not (0 < len(metadatas) <= INT_32_MAX):
        return False
    for item in metadatas:
        return validate_dict(item, max_str_length=1024*1024, max_list_length=4096,
                             max_dict_length=1024, max_check_depth=3)


class KnowledgeDB(KnowledgeBase):
    @validate_params(
        knowledge_store=dict(validator=lambda x: isinstance(x, KnowledgeStore),
                             message="param must be instance of KnowledgeStore"),
        chunk_store=dict(validator=lambda x: isinstance(x, Docstore),
                         message="param must be instance of Docstore"),
        vector_store=dict(validator=lambda x: isinstance(x, VectorStore),
                          message="param must be instance of VectorStore"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        max_file_count=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= FILE_COUNT_MAX,
                            message=f"param value range must be [1, {FILE_COUNT_MAX}]")
    )
    def __init__(
            self,
            knowledge_store: KnowledgeStore,
            chunk_store: Docstore,
            vector_store: VectorStore,
            knowledge_name: str,
            white_paths: List[str],
            max_file_count: int = 1000,
    ):
        super().__init__(white_paths)
        self._knowledge_store = knowledge_store
        self._vector_store = vector_store
        self._document_store = chunk_store
        self.max_file_count = max_file_count
        self.knowledge_name = knowledge_name
        self._check_store_accordance()

    def get_all_documents(self):
        """获取当前已上传的所有文档"""
        return self._knowledge_store.get_all(self.knowledge_name)

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        texts=dict(validator=lambda x: validata_list_str(x, [1, INT_32_MAX], [1, TEXT_MAX_LEN]),
                   message="param must meets: Type is List[str], "
                           f"list length range [1, {INT_32_MAX}], str length range [1, {TEXT_MAX_LEN}]"),
        metadatas=dict(validator=lambda x: _check_metadatas(x),
                       message='param must meets: Type is List[dict] or None,'
                               f' list length range [1, {INT_32_MAX}], other check please see the log')
    )
    def add_file(
            self,
            doc_name: str,
            texts: List[str],
            embed_func: Callable[[List[str]], List[List[float]]],
            metadatas: Optional[List[dict]] = None
    ) -> NoReturn:
        embeddings = embed_func(texts)
        if not isinstance(embeddings, List):
            raise KnowledgeError("The data type of embedding should be np.ndarray")
        metadatas = metadatas or [{} for _ in texts]
        if not len(texts) == len(metadatas) == len(embeddings):
            raise KnowledgeError("texts, metadatas, embeddings expected to be equal length")
        documents = [MxDocument(page_content=t, metadata=m, document_name=doc_name) for t, m in zip(texts, metadatas)]
        self._knowledge_store.add(self.knowledge_name, doc_name)
        ids = self._document_store.add(documents)
        self._vector_store.add(np.array(embeddings), ids)

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def delete_file(self, doc_name: str):
        self._knowledge_store.delete(self.knowledge_name, doc_name)
        ids = self._document_store.delete(doc_name)
        num_removed = self._vector_store.delete(ids)
        if len(ids) != num_removed:
            logger.warning("the number of documents does not match the number of vectors")

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def check_document_exist(self, doc_name: str) -> bool:
        return self._knowledge_store.check_document_exist(self.knowledge_name, doc_name)

    def _check_store_accordance(self) -> None:
        doc_chunks = self._document_store.get_all_index_id()
        vec_ids = self._vector_store.get_all_ids()
        if set(doc_chunks) != set(vec_ids):
            logger.error(f"the Docstore has {len(doc_chunks)} chunks in {self._document_store.db_path},"
                         f"but the VectorStore has {len(vec_ids)} vectors, that will cause some error,"
                         "please ensure that the data of this two databases is consistent")
            raise KnowledgeError("VectorStore is not accordance to Docstore !")


class KnowledgeMgrStore:
    FREE_SPACE_LIMIT = 200 * 1024 * 1024

    @validate_params(
        db_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                     message=STR_TYPE_CHECK_TIP_1024)
    )
    def __init__(self, db_path: str):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def add(self, knowledge_name: str):
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            raise KnowledgeError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        with self.session() as session:
            try:
                session.add(KnowledgeMgrModel(knowledge_name=knowledge_name))
                session.commit()
                logger.debug(f"success add {knowledge_name} in knowledgeMgr_table.")
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(f"Database error while adding knowledge: '{knowledge_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to add knowledge: '{knowledge_name}' due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add knowledge failed, {err}") from err

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def delete(self, knowledge_name: str):
        with self.session() as session:
            try:
                knowledge_to_delete = session.query(KnowledgeMgrModel
                                                    ).filter_by(knowledge_name=knowledge_name).first()
                if not knowledge_to_delete:
                    logger.debug(f"{knowledge_name} does not exist in db, no need delete.")
                else:
                    session.delete(knowledge_to_delete)
                    session.commit()
                    logger.debug(f"success delete {knowledge_name} in knowledgeMgr_table.")
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(f"Database error while deleting knowledge: '{knowledge_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to delete knowledge: '{knowledge_name}' due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

    def get_all(self):
        with self.session() as session:
            names = []
            for k in session.query(KnowledgeMgrModel).all():
                names.append(k.knowledge_name)
            return names


class KnowledgeMgr:
    @validate_params(
        mgr_store=dict(validator=lambda x: isinstance(x, KnowledgeMgrStore),
                       message="param must be instance of KnowledgeMgrStore")
    )
    def __init__(self, mgr_store: KnowledgeMgrStore):
        self.mgr_store = mgr_store

    @validate_params(
        knowledge=dict(validator=lambda x: isinstance(x, KnowledgeDB), message="param must be type KnowledgeDB")
    )
    def register(self, knowledge: KnowledgeDB):
        if knowledge.knowledge_name in self.mgr_store.get_all():
            raise KnowledgeError(f"knowledge {knowledge.knowledge_name} has been registered")
        self.mgr_store.add(knowledge.knowledge_name)

    @validate_params(
        knowledge=dict(validator=lambda x: isinstance(x, KnowledgeDB), message="param must be type KnowledgeDB")
    )
    def delete(self, knowledge: KnowledgeDB):
        if knowledge.knowledge_name not in self.mgr_store.get_all():
            raise KnowledgeError(f"knowledge {knowledge.knowledge_name} is not be registered")
        if knowledge.get_all_documents():
            raise KnowledgeError(f"please clear knowledge, before delete")
        self.mgr_store.delete(knowledge.knowledge_name)

    def get_all(self):
        return self.mgr_store.get_all()
