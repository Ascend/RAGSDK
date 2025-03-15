# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import multiprocessing
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Dict, Union, Any, Iterator, Tuple

import numpy as np
from loguru import logger
from opengauss_sqlalchemy.usertype import Vector, SPARSEVEC, SparseVector
from sqlalchemy import Column, text, BigInteger, Index, MetaData, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

from mx_rag.storage.vectorstore import VectorStore, SearchMode
from mx_rag.storage.document_store.base_storage import StorageError
from mx_rag.utils.common import validate_params
from mx_rag.utils.common import MAX_COLLECTION_NAME_LENGTH, MAX_TOP_K

DEFAULT_INDEX_OPTIONS = {'m': 16, 'ef_construction': 200}
Base = declarative_base()


def _vector_model_factory(
        table_name: str,
        search_mode: SearchMode,
        dense_dim: Optional[int] = None,
        sparse_dim: Optional[int] = None
) -> Any:
    """Factory function to create vector table model based on search mode."""

    class BaseModel(Base):
        __abstract__ = True
        __table_args__ = {'extend_existing': True}
        id = Column(BigInteger, primary_key=True, comment="向量ID")

    if search_mode == SearchMode.DENSE:
        class DenseModel(BaseModel):
            __tablename__ = table_name
            vector = Column(Vector(dense_dim))

        return DenseModel

    if search_mode == SearchMode.SPARSE:
        class SparseModel(BaseModel):
            __tablename__ = table_name
            sparse_vector = Column(SPARSEVEC(sparse_dim))

        return SparseModel

    class HybridModel(BaseModel):
        __tablename__ = table_name
        vector = Column(Vector(dense_dim))
        sparse_vector = Column(SPARSEVEC(sparse_dim))

    return HybridModel


def _metric_to_func_op(metric):
    metric_map = {
        "vector_l2_ops": "<->",  # 欧几里得距离L2
        "vector_ip_ops": "<#>",  # 负内积
        "vector_cosine_ops": "<=>",  # 余弦距离
        "sparsevec_ip_ops": "<#>"  # 负内积
    }
    op = metric_map.get(metric, None)
    if op is None:
        raise OpenGaussError(f"not supported metric: {metric}")
    return op


def _serialize_sparse(emb: Dict[int, float], dim: int) -> str:
    """Serialize sparse vector to database format."""
    return f'{{{",".join(f"{k}:{v}" for k, v in emb.items())}}}/{dim}'


class OpenGaussError(Exception):
    """Base exception for OpenGaussDB errors."""


class OpenGaussDB(VectorStore):
    """OpenGauss vector database implementation."""

    SCALE_MAP = {
        "IP": lambda x: -x,  # 负内积
        "L2": lambda x: max(1.0 - x / 2.0, 0.0),
        "COSINE": lambda x: max(1.0 - x / 2.0, 0.0),
    }

    METRIC_MAP = {
        "IP": "vector_ip_ops",
        "L2": "vector_l2_ops",
        "COSINE": "vector_cosine_ops",
    }

    INDEX_MAP = {
        "HNSW": "hnsw",
        "IVFFLAT": "ivfflat",
    }

    @validate_params(
        engine=dict(validator=lambda x: isinstance(x, Engine), message="param must be instance of Engine"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_COLLECTION_NAME_LENGTH and x.isidentifier(),
            message="param must be str, length range (0, 1024] and valid identifier"),
        search_mode=dict(validator=lambda x: isinstance(x, SearchMode), message="param must be instance of SearchMode"),
        index_type=dict(
            validator=lambda x: isinstance(x, str) and x in ("HNSW", "IVFFLAT"),
            message="param must be none or instance of str"),
        metric_type=dict(validator=lambda x: isinstance(x, str) and x in ("IP", "L2", "COSINE"),
                         message="param must be none or instance of str"),
    )
    def __init__(
            self,
            engine: Engine,
            collection_name: str = "vectorstore",
            search_mode: SearchMode = SearchMode.DENSE,
            index_type="HNSW",
            metric_type="IP"
    ):
        super().__init__()
        self.engine = engine
        self.table_name = collection_name
        self.search_mode = search_mode
        self.sparse_dim: Optional[int] = None
        self.vector_model: Optional[Any] = None
        self._index_type = index_type
        self._metric_type = metric_type

        self.session_factory = scoped_session(
            sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        )

    @classmethod
    def create(cls, **kwargs):
        if "engine" not in kwargs:
            logger.error(f"Missing required parameters: engine")
            return None

        try:
            instance = cls(
                engine=kwargs.pop("engine"),
                collection_name=kwargs.pop("collection_name", "vectorstore"),
                search_mode=kwargs.pop("search_mode", SearchMode.DENSE),
                index_type=kwargs.pop("index_type", "HNSW"),
                metric_type=kwargs.pop("metric_type", "IP")
            )
            instance.create_collection(
                dense_dim=kwargs.get("dense_dim"),
                sparse_dim=kwargs.get("sparse_dim", 100000),
                params=kwargs.get("params")
            )
            logger.info("Successfully create database instance")
            return instance
        except Exception as e:
            logger.error(f"Instance creation failed: {str(e)}")
            return None

    @validate_params(dense_dim=dict(validator=lambda x: x is None or isinstance(x, int),
                                    message="param requires to be None or int")
                     )
    def create_collection(
            self,
            dense_dim: Optional[int] = None,
            sparse_dim: int = 100000,
            params: Optional[Dict] = None
    ) -> None:
        """Initialize database schema and indexes."""
        if self.search_mode in [SearchMode.DENSE, SearchMode.HYBRID] and not dense_dim:
            raise OpenGaussError("param 'dense_dim' required for DENSE/HYBRID search mode")

        self.sparse_dim = sparse_dim
        self.vector_model = _vector_model_factory(
            self.table_name, self.search_mode, dense_dim, sparse_dim
        )

        try:
            with self._transaction():
                if not self.engine.dialect.has_table(self.engine.connect(), self.table_name):
                    Base.metadata.create_all(self.engine)
                    self._create_indexes(params or {})
                    logger.info(f"Create table: {self.table_name}")
        except SQLAlchemyError as e:
            raise StorageError(f"Collection creation failed: {str(e)}") from e

    def drop_collection(self):
        """Drops the table associated with the current object.

        Handles potential exceptions and checks for table existence before dropping.
        """
        table_name = self.table_name

        # Validate table name to prevent SQL injection
        if not table_name.isidentifier():
            raise StorageError(f"Invalid table name: {table_name}")

        # Quote the table name properly using SQLAlchemy's quoting mechanism
        quoted_table_name = self.engine.dialect.identifier_preparer.quote_identifier(table_name)
        logger.info(f"Dropping table: {quoted_table_name}")

        try:
            # Drop indexes first
            with self._transaction() as session:
                # Get all non-primary key indexes for this table using safe parameter binding
                indexes = session.execute(text(
                    """
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = :table_name 
                    AND indexname NOT LIKE '%_pkey'
                    """
                ), {"table_name": quoted_table_name}).fetchall()

                # Drop each index
                for idx in indexes:
                    index_name = idx[0]
                    # Force drop the index and its dependencies
                    session.execute(text(f"DROP INDEX IF EXISTS {index_name} CASCADE"))
                    logger.info(f"Dropped index '{index_name}'")

            # Then drop the table
            metadata = MetaData()
            metadata.reflect(bind=self.engine)

            if table_name in metadata.tables:
                table = metadata.tables[table_name]
                table.drop(self.engine, checkfirst=True)
                metadata.clear()
                logger.info(f"Table '{table_name}' dropped successfully.")
            else:
                logger.warning(f"Table '{table_name}' does not exist. Skipping drop.")

        except Exception as e:
            raise StorageError(f"Failed to drop collection: {str(e)}") from e

    @validate_params(
        embeddings=dict(validator=lambda x: isinstance(x, np.ndarray), message="param requires to be np.ndarray"),
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add(self, embeddings: np.ndarray, ids: List[int]):
        if self.search_mode != SearchMode.DENSE:
            raise ValueError("Add requires DENSE mode")
        self._internal_add(ids, embeddings)

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]"),
        sparse_embeddings=dict(validator=lambda x: isinstance(x, list) and all(isinstance(it, dict) for it in x),
                               message="param requires to be a list of dicts")
    )
    def add_sparse(self, ids: List[int], sparse_embeddings: List[Dict[int, float]]):
        if self.search_mode != SearchMode.SPARSE:
            raise ValueError("Add sparse requires SPARSE mode")
        self._internal_add(ids, sparse=sparse_embeddings)

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]"),
        dense_embeddings=dict(validator=lambda x: isinstance(x, np.ndarray),
                              message="param requires to be np.ndarray"),
        sparse_embeddings=dict(validator=lambda x: isinstance(x, list) and all(isinstance(it, dict) for it in x),
                               message="param requires to be a list of dicts")
    )
    def add_dense_and_sparse(self, ids: List[int],
                             dense_embeddings: np.ndarray,
                             sparse_embeddings: List[Dict[int, float]]):
        if self.search_mode != SearchMode.HYBRID:
            raise ValueError("Adding dense and sparse requires HYBRID mode")
        self._internal_add(ids, dense_embeddings, sparse_embeddings)

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]"))
    def delete(self, ids: List[int]):
        if len(ids) == 0:
            logger.warning("no id need be deleted")
            return 0

        try:
            with self._transaction() as session:
                delete_count = session.query(self.vector_model) \
                    .filter(self.vector_model.id.in_(ids)) \
                    .delete(synchronize_session=False)
                logger.info(f"Deleted {delete_count} vectors.")
                return delete_count
        except SQLAlchemyError as e:
            raise StorageError(f"Delete failed: {e}") from e

    @validate_params(
        k={
            "validator": lambda x: 0 < x <= MAX_TOP_K,
            "message": "param length range (0, 10000]"
        }
    )
    def search(self, embeddings: Union[List[List[float]], List[Dict[int, float]]], k: int = 3):
        """
        Searches for the k-nearest neighbors of the given embeddings.

        Args:
            embeddings: A list of dense vectors (list of lists) or sparse vectors (list of dictionaries).
            k: The number of nearest neighbors to return.

        Raises:
            ValueError: If embeddings is not a non-empty list of vectors or sparse vectors.

        Returns:
            The result of the parallel search.
        """
        if not (
                isinstance(embeddings, list)
                and len(embeddings) > 0
                and all(isinstance(e, (list, dict)) and len(e) > 0 for e in embeddings)
        ):
            raise ValueError(
                "embeddings must be a non-empty list of vectors (list) or sparse vectors (dict)"
            )

        return self._parallel_search(embeddings, k)

    def get_all_ids(self) -> List[int]:
        try:
            with self._transaction() as session:
                result = session.query(self.vector_model.id).all()
                ids = [i[0] for i in result]
                return ids

        except SQLAlchemyError as e:
            raise OpenGaussError("Failed to get all ids") from e

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        """Provide transactional scope around a series of operations."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction failed: {str(e)}")
            raise StorageError("Database operation failed") from e
        finally:
            session.close()

    def _internal_add(
            self,
            ids: List[int],
            dense: Optional[np.ndarray] = None,
            sparse: Optional[List[Dict[int, float]]] = None
    ) -> None:
        """Unified method for adding embeddings."""
        data = self._prepare_insert_data(ids, dense, sparse)
        self._bulk_insert(data)

    def _prepare_insert_data(
            self,
            ids: List[int],
            dense: Optional[np.ndarray] = None,
            sparse: Optional[List[Dict[int, float]]] = None
    ) -> List[Dict]:
        """Prepare data for bulk insertion."""
        data = [{"id": id_} for id_ in ids]
        if dense is not None:
            if len(ids) != len(dense):
                raise ValueError("Input lengths mismatch")
            for i, x in enumerate(data):
                x["vector"] = dense[i].tolist()
        if sparse is not None:
            if len(ids) != len(sparse):
                raise ValueError("Input lengths mismatch")
            for i, x in enumerate(data):
                x["sparse_vector"] = SparseVector(sparse[i], self.sparse_dim)
        return data

    def _bulk_insert(self, data: List[Dict]) -> None:
        """Execute bulk insert operation."""
        try:
            with self._transaction() as session:
                session.bulk_insert_mappings(self.vector_model, data)
                logger.info(f"Inserted {len(data)} vectors")
        except SQLAlchemyError as e:
            logger.error(f"Insert failed: {str(e)}")
            raise StorageError("Bulk insert failed") from e

    def _create_indexes(self, params: Dict) -> None:
        """Create appropriate indexes for the table."""
        index_options = {**DEFAULT_INDEX_OPTIONS, **params.get("index_creation_with_options", {})}

        with self._transaction() as session:
            # First, ensure no stale indexes exist
            session.execute(text(f"DROP INDEX IF EXISTS ix_dense_index CASCADE"))
            session.execute(text(f"DROP INDEX IF EXISTS ix_sparse_index CASCADE"))

        if hasattr(self.vector_model, "vector"):
            Index(
                "ix_dense_index",
                self.vector_model.vector,
                opengauss_using=self.INDEX_MAP.get(self._index_type),
                opengauss_with=index_options,
                opengauss_ops={'vector': self.METRIC_MAP.get(self._metric_type)}
            ).create(self.engine)

        if hasattr(self.vector_model, "sparse_vector"):
            Index(
                "ix_sparse_index",
                self.vector_model.sparse_vector,
                opengauss_using="hnsw",
                opengauss_with=index_options,
                opengauss_ops={'sparse_vector': "sparsevec_ip_ops"}
            ).create(self.engine)

    def _do_search(
            self,
            emb: Union[List[float], Dict[int, float]],
            k: int,
            metric_func_op: str
    ) -> Tuple[List[Any], List[float]]:
        """Execute single search query."""
        if isinstance(emb, list):
            emb = np.array(emb)
        with self._transaction() as session:
            field, param_key, order_dir = self._get_search_params(emb)
            emb_str = self._serialize_embedding(emb)

            query = session.query(
                self.vector_model,
                text(f"{field} {metric_func_op} :{param_key} AS score")
            ).order_by(text(f"score {order_dir}")).params(**{param_key: emb_str}).limit(k)

            results = query.all()
            return [item[0] for item in results], [item[1] for item in results]

    def _get_search_params(
            self,
            emb: Union[np.ndarray, Dict[int, float]]
    ) -> Tuple[str, str, str]:
        """Determine search parameters based on input type."""
        if isinstance(emb, np.ndarray):
            if self.search_mode not in [SearchMode.DENSE, SearchMode.HYBRID]:
                raise ValueError("Dense search requires DENSE/HYBRID mode")
            return "vector", "vector", "ASC"
        else:
            if self.search_mode not in [SearchMode.SPARSE, SearchMode.HYBRID]:
                raise ValueError("Sparse search requires SPARSE/HYBRID mode")
            return "sparse_vector", "sparsevec", "ASC"

    def _serialize_embedding(self, emb: Union[np.ndarray, Dict[int, float]]) -> str:
        """Convert embedding to database format."""
        if isinstance(emb, np.ndarray):
            return str(emb.tolist())
        return _serialize_sparse(emb, self.sparse_dim)

    def _parallel_search(
            self,
            embeddings: Union[List[List[float]], List[Dict[int, float]]],
            k: int = 3
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Execute parallel searches using thread pool."""
        metric_type = "sparsevec_ip_ops" if self.search_mode == SearchMode.SPARSE \
            else self.METRIC_MAP.get(self._metric_type)

        metric_func_op = _metric_to_func_op(metric_type)
        pool_size = self._calculate_pool_size()

        score_scale = self.SCALE_MAP.get("IP" if self.search_mode == SearchMode.SPARSE else self._metric_type)
        try:
            with ThreadPool(pool_size) as pool:
                results = pool.starmap(
                    self._do_search,
                    [(emb, k, metric_func_op) for emb in embeddings]
                )
            scores = [[score_scale(i) for i in s] for _, s in results]
            ids = [[item.id for item in r] for r, _ in results]
            return scores, ids
        except Exception as e:
            logger.error(f"Parallel search failed: {str(e)}")
            raise StorageError("Search operation failed") from e

    def _calculate_pool_size(self) -> int:
        """Determine optimal thread pool size."""
        cpu_count = multiprocessing.cpu_count()
        return min(
            self.engine.pool.size(),
            max(4, cpu_count - 4)
        )
