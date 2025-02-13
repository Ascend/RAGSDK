# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from contextlib import contextmanager
from typing import List, Optional, Callable, Iterator, Iterable
from pydantic import validate_call
from sqlalchemy import create_engine, delete, URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from loguru import logger

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import StorageError, Docstore
from mx_rag.storage.document_store.models import Base, ChunkModel
from mx_rag.utils.common import MAX_CHUNKS_NUM


class _DocStoreHelper(Docstore):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
            self,
            url: URL,
            encrypt_fn: Optional[Callable[[str], str]] = None,
            decrypt_fn: Optional[Callable[[str], str]] = None,
            batch_size: int = 500
    ):
        """
        文档存储实现

        Args:
            url: 数据库连接字符串
            encrypt_fn: 内容加密函数 (str -> str)
            decrypt_fn: 内容解密函数 (str -> str)
            batch_size: 批量操作大小
        """
        self.engine = create_engine(
            url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True
        )
        self.session_factory = scoped_session(
            sessionmaker(
                bind=self.engine,
                autoflush=False,
                expire_on_commit=False
            )
        )
        self._init_db()
        self.batch_size = batch_size
        self.encrypt_fn = encrypt_fn
        self.decrypt_fn = decrypt_fn

    def add(
            self,
            documents: List[MxDocument],
            document_id: int
    ) -> List[int]:
        """分批次添加文档块"""
        if not 0 < len(documents) <= MAX_CHUNKS_NUM:
            raise ValueError(f"Documents count must be between 1 and {MAX_CHUNKS_NUM}")

        def batch_insert(chunk_batch, session):
            # 构造模型对象时同步加密
            chunks = [
                ChunkModel(
                    document_id=document_id,
                    document_name=doc.document_name,
                    chunk_content=self.encrypt_fn(doc.page_content) if self.encrypt_fn else doc.page_content,
                    chunk_metadata=doc.metadata
                ) for doc in chunk_batch
            ]
            session.bulk_save_objects(chunks, return_defaults=True)

        try:
            # 分批次处理原始文档
            self._batch_operation(
                iterable=documents,
                operation=batch_insert,
                desc=f"for document {document_id}"
            )

            # 获取生成的ID需要特殊处理（批量插入返回ID的限制）
            with self._transaction() as session:
                last_chunk = session.query(ChunkModel) \
                    .filter_by(document_id=document_id) \
                    .order_by(ChunkModel.chunk_id.desc()) \
                    .limit(len(documents)).all()
                inserted_ids = [c.chunk_id for c in reversed(last_chunk)]

            logger.info("Inserted {} chunks for doc {}", len(inserted_ids), document_id)
            return inserted_ids

        except SQLAlchemyError as e:
            raise StorageError(f"Bulk insert failed: {e}") from e

    def delete(self, document_id: int) -> List[int]:
        """分批次删除文档"""
        try:
            # 先查询所有需要删除的ID
            with self._transaction() as session:
                target_ids = session.query(ChunkModel.chunk_id) \
                    .filter_by(document_id=document_id) \
                    .all()
                target_ids = [id_[0] for id_ in target_ids]

            # 分批次执行删除
            def batch_delete(id_batch, session):
                session.execute(
                    delete(ChunkModel)
                    .where(ChunkModel.chunk_id.in_(id_batch))
                )

            self._batch_operation(
                iterable=target_ids,
                operation=batch_delete,
                desc=f"deleting doc {document_id}"
            )

            logger.info("Deleted {} chunks for doc {}", len(target_ids), document_id)
            return target_ids

        except SQLAlchemyError as e:
            raise StorageError(f"Delete failed: {e}") from e

    def search(self, chunk_id: int) -> Optional[MxDocument]:
        """
        根据chunk_id检索文档

        Args:
            chunk_id: 要查询的块ID

        Returns:
            MxDocument对象或None
        """
        with self._transaction() as session:
            chunk = session.get(ChunkModel, chunk_id)
            if not chunk:
                return None

            content = chunk.chunk_content
            if self.decrypt_fn:
                try:
                    content = self.decrypt_fn(content)
                except Exception as e:
                    logger.error("Decryption failed for chunk {}", chunk_id)
                    raise StorageError("Decryption failed") from e

            return MxDocument(
                page_content=content,
                metadata=chunk.chunk_metadata,
                document_name=chunk.document_name
            )

    def get_all_index_id(self) -> List[int]:
        """获取所有chunk_id的生成器实现"""
        with self._transaction() as session:
            query = session.query(ChunkModel.chunk_id).yield_per(1000)
            return [chunk_id for (chunk_id,) in query]

    @contextmanager
    def _transaction(self) -> Iterator[scoped_session]:
        """提供事务上下文管理的会话"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Database operation failed: {}", e)
            raise StorageError(f"Database operation failed: {e}") from e
        finally:
            session.close()

    def _init_db(self):
        """初始化数据库表结构"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables initialized")
        except SQLAlchemyError as e:
            logger.critical("Database initialization failed: {}", e)
            raise StorageError("Database setup failed") from e

    def _batch_operation(self, iterable: Iterable, operation: Callable, desc: str = ""):
        """通用分批次操作执行器"""

        total = 0
        batch = []

        def commit_batch(session: Session):
            nonlocal batch, total
            if batch:
                operation(batch, session)
                session.commit()
                total += len(batch)
                logger.debug(f"Processed {total} items {desc}")
                batch = []

        def commit_all(iterable: Iterable, session: Session):
            nonlocal batch
            for i, item in enumerate(iterable, 1):
                batch.append(item)
                if i % self.batch_size == 0:
                    commit_batch(session)
            commit_batch(session)  # 提交最后一批

        try:
            with self._transaction() as session:  # 使用统一的会话上下文
                commit_all(iterable, session)
                logger.info(f"Successfully processed {total} items {desc}")
                return total
        except Exception as e:
            logger.error(f"Batch operation failed at {total}: {str(e)}")
            raise StorageError(f"Batch operation failed: {e}") from e
