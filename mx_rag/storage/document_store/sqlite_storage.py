# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import datetime
import os
from typing import List, Optional, Callable

from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import StorageError, Docstore
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM, MAX_SQLITE_FILE_NAME_LEN, \
    check_db_file_limit
from mx_rag.utils.file_check import FileCheck, check_disk_free_space

Base = declarative_base()


class ChunkModel(Base):
    __tablename__ = "chunks_table"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, comment="文档id")
    document_name = Column(String, comment="对应原始文档文件名")
    chunk_content = Column(String, comment="chunk内容")
    chunk_metadata = Column(JSON, comment="文档metadata")
    create_time = Column(DateTime, comment="创建时间", default=datetime.datetime.utcnow)
    __table_args__ = (
        {"sqlite_autoincrement": True}
    )

    def encrypt_chunk(self, encrypt_fun: Callable):
        if not isinstance(encrypt_fun, Callable):
            logger.error("encrypt_fun is not callable function, take no effect.")
            return
        try:
            encrypted_chunk = encrypt_fun(self.chunk_content)
            if not isinstance(encrypted_chunk, str):
                logger.error("encrypt_fun return value is str, take no effect.")
                return
            self.chunk_content = encrypted_chunk
        except Exception:
            logger.error("encrypt chunk content failed.")

    def decrypt_chunk(self, decrypt_fun: Callable):
        if not isinstance(decrypt_fun, Callable):
            logger.error("decrypt_fun is not callable function, take no effect.")
            return
        try:
            decrypted_chunk = decrypt_fun(self.chunk_content)
            if not isinstance(decrypted_chunk, str):
                logger.error("decrypted_chunk return value is str, take no effect.")
                return
            self.chunk_content = decrypted_chunk
        except Exception:
            logger.error("encrypt chunk content failed.")


class SQLiteDocstore(Docstore):
    FREE_SPACE_LIMIT = 200 * 1024 * 1024
    MAX_DOC_NAME_LEN = 1024

    @validate_params(
        encrypt_fun=dict(validator=lambda x: x is None or isinstance(x, Callable),
                         message="encrypt_fun must be None or callable function"),
        decrypt_fun=dict(validator=lambda x: x is None or isinstance(x, Callable),
                         message="decrypt_fun must be None or callable function")
    )
    def __init__(self, db_path: str, encrypt_fun: Callable = None, decrypt_fun: Callable = None):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        FileCheck.check_filename_valid(db_path, max_length=MAX_SQLITE_FILE_NAME_LEN)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = scoped_session(sessionmaker(bind=engine))
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)
        self.__encrypt_fun = encrypt_fun
        self.__decrypt_fun = decrypt_fun

    @validate_params(
        documents=dict(
            validator=lambda x: 0 < len(x) <= MAX_CHUNKS_NUM and all(isinstance(it, MxDocument) for it in x),
            message="param must be List[MxDocument] and length range in (0, 1000 * 1000]")
    )
    def add(self, documents: List[MxDocument], document_id: int) -> List[int]:
        FileCheck.check_input_path_valid(self.db_path, check_blacklist=True)
        FileCheck.check_filename_valid(self.db_path, max_length=MAX_SQLITE_FILE_NAME_LEN)
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            raise StorageError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        with self.session() as session:
            try:
                chunks = [
                    ChunkModel(document_id=document_id, document_name=doc.document_name, chunk_content=doc.page_content,
                               chunk_metadata=doc.metadata) for doc in documents
                ]
                if self.__encrypt_fun:
                    [chunk.encrypt_chunk(self.__encrypt_fun) for chunk in chunks]
                session.add_all(chunks)
                session.commit()
                logger.debug(f"success add {chunks[0].document_name} in chunks_table.")
                idxs = [chunk.chunk_id for chunk in chunks]
                return idxs
            except SQLAlchemyError as sql_err:
                session.rollback()
                logger.error(f"Database error while adding chunks: {sql_err}")
                raise StorageError(f"Failed to add chunks due to database error: {sql_err}") from sql_err
            except Exception as err:
                session.rollback()
                raise StorageError(f"add chunk failed, {err}") from err

    def delete(self, document_id: int) -> List[int]:
        with self.session() as session:
            try:
                chunks = session.query(ChunkModel).filter_by(document_id=document_id).all()
                idxs = [c.chunk_id for c in chunks]
                for chunk in chunks:
                    session.delete(chunk)
                session.commit()
                logger.debug(f"successfully delete document in chunks_table.(document_id: {document_id})")
                return idxs
            except SQLAlchemyError as sql_err:
                session.rollback()
                logger.error(f"Database error while deleting doc: {sql_err}")
                raise StorageError(f"Failed to delete chunks due to database error: {sql_err}") from sql_err
            except Exception as err:
                session.rollback()
                raise StorageError(f"delete chunk failed, {err}") from err

    @validate_params(chunk_id=dict(validator=lambda x: x >= 0, message="param must greater equal than 0"))
    def search(self, chunk_id: int) -> Optional[MxDocument]:
        with self.session() as session:
            chunk = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
            if self.__decrypt_fun:
                chunk.decrypt_chunk(self.__decrypt_fun)
            if chunk is not None:
                return MxDocument(
                    page_content=chunk.chunk_content,
                    metadata=chunk.chunk_metadata,
                    document_name=chunk.document_name
                )
            return chunk

    def get_all_index_id(self) -> List[int]:
        with self.session() as session:
            chunks = session.query(ChunkModel)
            ids = [chunk.chunk_id for chunk in chunks.all()]
            return ids
