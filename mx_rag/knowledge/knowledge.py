# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import datetime
import os
import pathlib
import re
from typing import List, Callable, Optional, NoReturn

import numpy as np
import sqlalchemy
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String, DateTime, UniqueConstraint, func, JSON
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

from mx_rag.knowledge.base_knowledge import KnowledgeBase, KnowledgeError
from mx_rag.storage.document_store import Docstore, MxDocument
from mx_rag.storage.vectorstore import VectorStore
from mx_rag.utils.common import validate_params, FILE_COUNT_MAX, MAX_SQLITE_FILE_NAME_LEN, \
    check_db_file_limit, validata_list_str, TEXT_MAX_LEN, STR_TYPE_CHECK_TIP_1024, validate_sequence, STR_MAX_LEN, \
    check_pathlib_path, validate_lock
from mx_rag.utils.file_check import FileCheck, check_disk_free_space

Base = declarative_base()


class KnowledgeModel(Base):
    __tablename__ = "knowledge_table"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(Integer, autoincrement=True, nullable=False)
    knowledge_name = Column(String, comment="知识库名称")
    user_id = Column(String, comment="用户id")
    member_id = Column(JSON, comment="成员id")
    create_time = Column(DateTime, comment="创建时间", default=datetime.datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('knowledge_name', 'user_id', name="knowledge_name"),
        {"sqlite_autoincrement": True}
    )


class DocumentModel(Base):
    __tablename__ = "document_table"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(Integer, comment="知识库ID", nullable=False)
    knowledge_name = Column(String, comment="知识库名称")
    document_name = Column(String, comment="文档名称")
    document_file_path = Column(String, comment="文档路径")
    create_time = Column(DateTime, comment="创建时间", default=datetime.datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('knowledge_id', 'document_name', name="knowledge_id"),
        {"sqlite_autoincrement": True}
    )


class KnowledgeStore:
    FREE_SPACE_LIMIT = 200 * 1024 * 1024

    @validate_params(
        db_path=dict(validator=lambda x: 0 < len(x) <= 1024 and isinstance(x, str), message=STR_TYPE_CHECK_TIP_1024)
    )
    def __init__(self, db_path: str):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        FileCheck.check_filename_valid(db_path, MAX_SQLITE_FILE_NAME_LEN)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = scoped_session(sessionmaker(bind=engine))
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        file_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                       message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'")

    )
    def add_doc_info(self, knowledge_name: str, doc_name: str, file_path: str, user_id: str):
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            logger.error("Insufficient remaining space. Please clear disk space.")
            raise KnowledgeError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        chunk = self.check_document_exist(knowledge_name, doc_name, user_id)
        with self.session() as session:
            try:
                if chunk:
                    document_model = DocumentModel(knowledge_id=chunk.knowledge_id, knowledge_name=knowledge_name,
                                                   document_name=doc_name)
                    logger.debug(f"{doc_name} already exist in {knowledge_name}")
                    return document_model.document_id

                knowledge = session.query(KnowledgeModel
                                          ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
                if not knowledge:
                    knowledge_id = self.add_knowledge(knowledge_name, user_id)
                knowledge_id = knowledge.knowledge_id if knowledge else knowledge_id
                # 创建新的文档
                document_model = DocumentModel(knowledge_id=knowledge_id, knowledge_name=knowledge_name,
                                               document_name=doc_name, document_file_path=file_path)
                session.add(document_model)
                session.commit()
                logger.debug(f"success add (knowledge_name={knowledge_name}, "
                             f"doc_name={doc_name}, user_id={user_id}) in knowledge_table.")
                return document_model.document_id
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(
                    f"Database error while adding knowledge: '{knowledge_name}', document: '{doc_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to add knowledge: '{knowledge_name}', document: '{doc_name}' "
                    "due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add chunk failed") from err

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'")
    )
    def delete_doc_info(self, knowledge_name: str, doc_name: str, user_id: str = "Default"):
        with self.session() as session:
            try:
                knowledge = session.query(KnowledgeModel
                                          ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
                if not knowledge:
                    logger.debug(f"{knowledge_name} does not exist in knowledge_table, no need delete.")
                    return None
                # 删除文档信息
                doc_to_delete = session.query(DocumentModel).filter_by(document_name=doc_name,
                                                                       knowledge_id=knowledge.knowledge_id).first()
                if not doc_to_delete:
                    logger.debug(f"{doc_name} does not exist in {knowledge_name}, no need delete.")
                    return None
                else:
                    session.delete(doc_to_delete)
                    session.commit()
                    logger.debug(f"success delete (knowledge_name={knowledge_name}, "
                                 f"document_name={doc_name}, user_id={user_id}) in document_table.")
                return doc_to_delete.document_id
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
                            message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'")
    )
    def get_all_documents_by_knowledge(self, knowledge_name: str, user_id: str = None, member_id=None):
        with self.session() as session:
            if user_id is not None:
                knowledge = session.query(KnowledgeModel
                                          ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
            if knowledge is None:
                logger.debug(f"(knowledge_name={knowledge_name}, user_id={user_id}) does not exist in knowledge_table")
            if member_id is not None:
                knowledge = session.query(KnowledgeModel
                                          ).filter_by(knowledge_name=knowledge_name).first()
                if knowledge and knowledge.member_id is not None and member_id not in knowledge.member_id:
                    knowledge = None
                    logger.debug(f"(knowledge_name={knowledge_name}, member_id={member_id}) "
                                 f"does not exist in knowledge_table")
                elif knowledge and knowledge.member_id is None:
                    knowledge = None
                    logger.debug(f"(knowledge_name={knowledge_name}, member_id={member_id}) "
                                 f"does not exist in knowledge_table")
            if knowledge:
                return session.query(DocumentModel).filter_by(knowledge_id=knowledge.knowledge_id).all()
            return []

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'")
    )
    def check_document_exist(self, knowledge_name: str, doc_name: str, user_id: str):
        with self.session() as session:
            # 同一个user_id下知识库名称不能重复
            knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
            if not knowledge:
                return False
            chunk = session.query(DocumentModel).filter_by(
                knowledge_id=knowledge.knowledge_id, document_name=doc_name).first()
            return chunk

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'")
    )
    def get_all_knowledge_info(self, user_id=None, member_id=None):
        with self.session() as session:
            if user_id is not None:
                knowledge_list = session.query(KnowledgeModel).filter_by(user_id=user_id).all()
            if member_id is not None:
                knowledge_list = session.query(KnowledgeModel).filter(
                    KnowledgeModel.member_id.contains([member_id])).all()
        return knowledge_list or []

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'")
    )
    def delete_knowledge(self, knowledge_name, user_id):
        with self.session() as session:
            session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name, user_id=user_id).delete()
            session.commit()
            logger.debug(f"success delete (knowledge_name={knowledge_name}, user_id={user_id}) in knowledge_table.")

    def add_knowledge(self, knowledge_name, user_id, member_id=None):
        if member_id is None:
            member_id = []
        is_exist = self.check_knowledge_exist(knowledge_name=knowledge_name, user_id=user_id)

        with self.session() as session:
            if is_exist:
                knowledge_model = KnowledgeModel(knowledge_name=knowledge_name, user_id=user_id)
                logger.debug(f"(knowledge_name={knowledge_name}, user_id={user_id}) "
                             f"already exist in knowledge_table")
                return knowledge_model.knowledge_id

            max_id = session.query(KnowledgeModel).with_entities(
                func.max(KnowledgeModel.knowledge_id)).scalar() or 0
            knowledge_id = max_id + 1
            knowledge_model = KnowledgeModel(knowledge_id=knowledge_id, member_id=member_id,
                                             knowledge_name=knowledge_name, user_id=user_id)
            session.add(knowledge_model)
            session.commit()
            return knowledge_id

    def add_member_id_to_knowledge(self, knowledge_name, member_id):
        if isinstance(member_id, str):
            member_id = [member_id]
        try:
            with self.session() as session:
                knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name).first()
                if not knowledge:
                    raise KnowledgeError(f"(knowledge_name={knowledge_name}, member_id={member_id})"
                                         f" does not exist in knowledge_table")
                if knowledge.member_id is None:
                    knowledge.member_id = []
                new_members_added = False
                updated_member_id = knowledge.member_id.copy()
                for mid in member_id:
                    if mid not in updated_member_id:
                        updated_member_id.append(mid)
                        new_members_added = True
                        logger.info(f"successfully added member_id {mid} to knowledge {knowledge_name}")

                if new_members_added:
                    knowledge.member_id = updated_member_id
                    session.commit()

        except sqlalchemy.exc.IntegrityError as e:
            logger.error(f"failed to add, the {member_id} to {knowledge_name} "
                         f"have same member_id in {knowledge_name}")
            raise KnowledgeError(f"failed to add {member_id} to {knowledge_name}") from e
        except Exception as e:
            raise KnowledgeError(f"failed to add {member_id} to {knowledge_name}") from e

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def delete_usr_id_from_knowledge(self, user_id, knowledge_name):
        with self.session() as session:
            knowledge = session.query(KnowledgeModel
                                      ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
            if not knowledge:
                raise KnowledgeError(f"(user_id={user_id}, knowledge_name={knowledge_name})"
                                     f" does not exist in knowledge_table")

            knowledges = session.query(KnowledgeModel
                                       ).filter_by(knowledge_id=knowledge.knowledge_id).all()
            if len(knowledges) == 1:
                raise KnowledgeError(
                    f"The knowledge {knowledge_name} now only belongs to user {user_id}, not support delete. "
                    f"please use KnowledgeDB.delete_knowledge to clear, that operation will delete all documents "
                    f"of {knowledge_name}, and the vector database.")
            session.delete(knowledge)
            session.commit()

    def delete_member_from_knowledge(self, knowledge_name, member_id):
        if isinstance(member_id, str):
            member_id = [member_id]
        with self.session() as session:
            knowledge = session.query(KnowledgeModel
                                      ).filter_by(knowledge_name=knowledge_name).first()
            if not knowledge:
                raise KnowledgeError(f"(member_id={member_id}, knowledge_name={knowledge_name}) "
                                     f"does not exist in knowledge_table")

            if knowledge.member_id is None:
                knowledge.member_id = []
            members_deleted = False
            updated_member_id = knowledge.member_id.copy()
            for mid in member_id:
                if mid in updated_member_id:
                    updated_member_id.remove(mid)
                    members_deleted = True
                    logger.info(f"successfully deleted member_id {member_id} from knowledge {knowledge_name}")

            if members_deleted:
                knowledge.member_id = updated_member_id
                session.commit()

    def check_knowledge_exist(self, knowledge_name: str, user_id: str, member_id: str = None) -> bool:
        return knowledge_name in self.get_all_knowledge_name(user_id, member_id)

    def get_all_knowledge_name(self, user_id: str, member_id: str = None) -> List[str]:
        knowledge_list = self.get_all_knowledge_info(user_id, member_id)
        knowledge_name_list = [knowledge.knowledge_name for knowledge in knowledge_list]
        return knowledge_name_list


def _check_metadatas(metadatas) -> bool:
    if metadatas is None:
        return True
    if not isinstance(metadatas, list) or not (0 < len(metadatas) <= TEXT_MAX_LEN):
        logger.error(f"metadatas type incorrect or length over {TEXT_MAX_LEN}")
        return False
    for item in metadatas:
        if not isinstance(item, dict):
            logger.error("metadata type is not dict")
            return False
        if not validate_sequence(item):
            return False

    return True


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
                            message=f"param value range must be [1, {FILE_COUNT_MAX}]"),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_]{6,16}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_]{6,16}$'"),
        lock=dict(validator=lambda x: x is None or validate_lock(x),
                  message="param must be one of None, multiprocessing.Lock(), threading.Lock()")
    )
    def __init__(
            self,
            knowledge_store: KnowledgeStore,
            chunk_store: Docstore,
            vector_store: VectorStore,
            knowledge_name: str,
            white_paths: List[str],
            max_file_count: int = 1000,
            user_id: str = "Default",
            member_id: Optional[List[str]] = None,
            lock=None
    ):
        super().__init__(white_paths)
        self._knowledge_store = knowledge_store
        self._vector_store = vector_store
        self._document_store = chunk_store
        self.max_file_count = max_file_count
        self.knowledge_name = knowledge_name
        self.user_id = user_id
        self.member_id = member_id
        self.lock = lock
        if self.lock:
            with self.lock:
                self._check_store_accordance()
        else:
            self._check_store_accordance()

    def get_all_documents(self):
        """获取当前已上传的所有文档"""
        return self._knowledge_store.get_all_documents_by_knowledge(self.knowledge_name, self.user_id, self.member_id)

    @validate_params(
        file=dict(validator=lambda x: check_pathlib_path(x), message="param check failed, please see the log"),
        texts=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], "
                           f"list length range [1, {TEXT_MAX_LEN}], str length range [1, {STR_MAX_LEN}]"),
        metadatas=dict(validator=lambda x: _check_metadatas(x),
                       message='param must meets: Type is List[dict] or None,'
                               f' list length range [1, {TEXT_MAX_LEN}], other check please see the log')
    )
    def add_file(self, file: pathlib.Path, texts: List[str], embed_func: Callable[[List[str]], List[List[float]]],
                 metadatas: Optional[List[dict]]) -> NoReturn:
        embeddings = embed_func(texts)
        if not isinstance(embeddings, List):
            raise KnowledgeError("The data type of embedding should be List[float]")
        metadatas = metadatas or [{} for _ in texts]
        if not len(texts) == len(metadatas) == len(embeddings):
            raise KnowledgeError("texts, metadatas, embeddings expected to be equal length")
        documents = [MxDocument(page_content=t, metadata=m, document_name=file.name) for t, m in zip(texts, metadatas)]
        if self.lock:
            with self.lock:
                self._storage_and_vector_add(file.name, file.as_posix(), documents, embeddings)
        else:
            self._storage_and_vector_add(file.name, file.as_posix(), documents, embeddings)

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def delete_file(self, doc_name: str):
        if self.lock:
            with self.lock:
                self._storage_and_vector_delete(doc_name)
        else:
            self._storage_and_vector_delete(doc_name)

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def check_document_exist(self, doc_name: str) -> bool:
        return self._knowledge_store.check_document_exist(self.knowledge_name, doc_name, self.user_id) is not None

    def _check_store_accordance(self) -> None:
        chunk_ids = set(self._document_store.get_all_chunk_id())
        vec_ids = set(self._vector_store.get_all_ids())
        if chunk_ids != vec_ids:
            raise KnowledgeError("Vector store does not comply with the document store: different ids")

    def _storage_and_vector_delete(self, doc_name: str):
        document_id = self._knowledge_store.delete_doc_info(self.knowledge_name, doc_name, self.user_id)
        if document_id is None:
            return
        ids = self._document_store.delete(document_id)
        num_removed = self._vector_store.delete(ids)
        if len(ids) != num_removed:
            logger.warning("the number of documents does not match the number of vectors")

    def _storage_and_vector_add(self, doc_name: str, file_path: str, documents: List, embeddings: List):
        document_id = self._knowledge_store.add_doc_info(self.knowledge_name, doc_name, file_path, self.user_id)
        ids = self._document_store.add(documents, document_id)
        self._vector_store.add(np.array(embeddings), ids)
