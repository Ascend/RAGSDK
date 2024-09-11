# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from mx_rag.storage.document_store.base_storage import Docstore, MxDocument, StorageError
from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.file_operate import check_disk_free_space
from mx_rag.utils.common import validate_params

Base = declarative_base()


class ChunkIdxModel(Base):
    __tablename__ = "chunks_meta"

    id = Column(Integer, primary_key=True)
    chunk_nums = Column(Integer, comment="累计添加的文档数量，用于设置索引的id", default=0)

    # 三个预留字段
    reserved1 = Column(String, default=None, nullable=True, comment="预留字段1")
    reserved2 = Column(String, default=None, nullable=True, comment="预留字段2")
    reserved3 = Column(String, default=None, nullable=True, comment="预留字段3")


class ChunkModel(Base):
    __tablename__ = "chunks_table"

    chunk_id = Column(Integer, primary_key=True)
    index_id = Column(Integer, comment="对应索引的id")
    chunk_content = Column(String, comment="chunk内容")
    document_name = Column(String, comment="对应原始文档文件名")
    chunk_metadata = Column(JSON, comment="文档metadata")
    # 三个预留字段
    reserved1 = Column(String, default=None, nullable=True, comment="预留字段1")
    reserved2 = Column(String, default=None, nullable=True, comment="预留字段2")
    reserved3 = Column(String, default=None, nullable=True, comment="预留字段3")


class SQLiteDocstore(Docstore):

    FREE_SPACE_LIMIT = 200 * 1024 * 1024
    MAX_DOC_NAME_LEN = 1024

    def __init__(self, db_path: str):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    @validate_params(documents=dict(validator=lambda x: all(isinstance(it, MxDocument) for it in x)))
    def add(self, documents: List[MxDocument]) -> List[int]:
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            raise StorageError("Insufficient remaining space, please clear disk space")
        with self.session() as session:
            try:
                chunk_idx = session.query(ChunkIdxModel).filter_by(id=1).first()
                # 数据表不存在数据时，创建第一条数据
                if chunk_idx is None:
                    chunk_idx = ChunkIdxModel()
                    session.add(chunk_idx)
                    session.flush()

                idxs = [chunk_idx.chunk_nums + i for i in range(1, len(documents) + 1)]
                chunks = [
                    ChunkModel(chunk_content=doc.page_content, document_name=doc.document_name,
                               chunk_metadata=doc.metadata, index_id=idx)
                    for doc, idx in zip(documents, idxs)
                ]
                chunk_idx.chunk_nums += len(chunks)
                session.bulk_save_objects(chunks)
                session.commit()
                return idxs
            except SQLAlchemyError as sql_err:
                session.rollback()
                logger.error(f"Database error while adding chunks: {sql_err}")
                raise StorageError(f"Failed to add chunks due to database error: {sql_err}") from sql_err
            except Exception as err:
                session.rollback()
                raise StorageError(f"add chunk failed, {err}") from err

    @validate_params(doc_name=dict(validator=lambda x: 0 < len(x) <= SQLiteDocstore.MAX_DOC_NAME_LEN))
    def delete(self, doc_name: str) -> List[int]:
        with self.session() as session:
            try:
                chunks = session.query(ChunkModel).filter_by(document_name=doc_name)
                idxs = [c.index_id for c in chunks]
                chunks.delete(synchronize_session=False)
                session.commit()
                return idxs
            except SQLAlchemyError as sql_err:
                session.rollback()
                logger.error(f"Database error while deleting doc: {sql_err}")
                raise StorageError(f"Failed to add chunks due to database error: {sql_err}") from sql_err
            except Exception as err:
                session.rollback()
                raise StorageError(f"delete chunk failed, {err}") from err

    @validate_params(index_id=dict(validator=lambda x: x >= 0))
    def search(self, index_id: int) -> Optional[MxDocument]:
        with self.session() as session:
            chunk = session.query(ChunkModel).filter_by(index_id=index_id).first()
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
            ids = [chunk.index_id for chunk in chunks.all()]
            return ids

