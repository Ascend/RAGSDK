# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

from .base_storage import Docstore, Document, StorageError

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

    def __init__(self, db_path: str):
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    def add(self, documents: List[Document], *args, **kwargs) -> List[int]:
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
            except Exception as err:
                session.rollback()
                raise StorageError(f"add chunk failed, {err}") from err

    def delete(self, doc_name: str, *args, **kwargs) -> List[int]:
        with self.session() as session:
            try:
                chunks = session.query(ChunkModel).filter_by(document_name=doc_name)
                idxs = [c.index_id for c in chunks]
                chunks.delete(synchronize_session=False)
                session.commit()
                return idxs
            except Exception as err:
                session.rollback()
                raise StorageError(f"delete chunk failed, {err}") from err

    def search(self, index_id: int, *args, **kwargs) -> Optional[Document]:
        with self.session() as session:
            chunk = session.query(ChunkModel).filter_by(index_id=index_id).first()
            if chunk is not None:
                return Document(
                    page_content=chunk.chunk_content,
                    metadata=chunk.chunk_metadata,
                    document_name=chunk.document_name
                )
            return chunk

    def check_document_exist(self, doc_name: str) -> bool:
        with self.session() as session:
            chunk = session.query(ChunkModel).filter_by(document_name=doc_name).first()
            return True if chunk is not None else False
