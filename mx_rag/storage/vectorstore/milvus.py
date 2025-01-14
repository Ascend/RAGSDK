# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

from typing import List, Dict, Optional, Any, Union

import numpy as np
from loguru import logger
from pymilvus import MilvusClient, DataType, MilvusException

from mx_rag.storage.vectorstore.vectorstore import VectorStore, SimilarityStrategy, SearchMode
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, MAX_TOP_K, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.common import validata_list_str


class MilvusError(Exception):
    pass


class SchemaBuilder:
    def __init__(self):
        self.schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        self._add_base_fields()

    def add_vector_field(self, field_name: str, datatype: DataType, dim: int):
        self.schema.add_field(field_name=field_name, datatype=datatype, dim=dim)

    def add_sparse_vector_field(self, field_name: str, datatype: DataType):
        self.schema.add_field(field_name=field_name, datatype=datatype)

    def build(self):
        return self.schema

    def _add_base_fields(self):
        self.schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)


class IndexParamsBuilder:
    def __init__(self, client, search_mode: SearchMode):
        self.client = client
        self.search_mode = search_mode
        self.index_params = self.client.prepare_index_params()

    def add_sparse_index(self, params: Dict[str, Any]):
        self.index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params=params,
        )

    def add_dense_index(self, similarity: Dict[str, Any], params: Dict[str, Any]):
        self.index_params.add_index(
            field_name="vector",
            index_name="dense_index",
            index_type=similarity.get("index"),
            metric_type=similarity.get("metric"),
            params=params,
        )

    def build(self):
        return self.index_params

    def prepare_index_params(self, similarity: Dict[str, Any], params: Dict[str, Any]):
        # Build the index parameters based on search_mode
        if self.search_mode == SearchMode.DENSE:
            self.add_dense_index(similarity, params)
        elif self.search_mode == SearchMode.SPARSE:
            self.add_sparse_index(params)
        else:
            self.add_sparse_index(params.get("sparse", {}))
            self.add_dense_index(similarity, params.get("dense", {}))

        return self.build()


class MilvusDB(VectorStore):
    MAX_COLLECTION_NAME_LENGTH = 1024
    MAX_URL_LENGTH = 1024
    MAX_QUERY_LENGTH = 1000
    MAX_DICT_LENGTH = 100000

    SIMILARITY_STRATEGY_MAP = {
        SimilarityStrategy.FLAT_IP:
            {
                "index": "FLAT",
                "metric": "IP",
                "scale": lambda x: min(x, 1.0)
            },
        SimilarityStrategy.FLAT_L2:
            {
                "index": "FLAT",
                "metric": "L2",
                "scale": lambda x: max(1.0 - x / 2.0, 0.0),
            },
        SimilarityStrategy.FLAT_COS:
            {
                "index": "FLAT",
                "metric": "COSINE",
                "scale": lambda x: min(x, 1.0)
            }
    }

    @validate_params(
        client=dict(validator=lambda x: isinstance(x, MilvusClient),
                    message="param must be instance of MilvusClient"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
            message="param must be str and length range (0, 1024]"),
        search_mode=dict(validator=lambda x: isinstance(x, SearchMode),
                         message="param must be instance of SearchMode")
    )
    def __init__(self, client: MilvusClient, collection_name: str = "mxRag",
                 search_mode: SearchMode = SearchMode.DENSE):
        super().__init__()
        self._client = client
        self._collection_name = collection_name
        self._search_mode = search_mode

    @property
    def search_mode(self):
        return self._search_mode

    @property
    def client(self):
        return self._client

    @staticmethod
    def create(**kwargs):
        client_field = "client"
        if client_field not in kwargs or not isinstance(kwargs.get(client_field), MilvusClient):
            logger.error(f"param error: {client_field} must be specified")
            return None

        client = kwargs.pop("client")
        x_dim = kwargs.pop("x_dim", None)

        ss = kwargs.pop("similarity_strategy", None)
        if ss is not None and not isinstance(ss, SimilarityStrategy):
            logger.error("param error: similarity_strategy must be SimilarityStrategy")
            return None

        params = kwargs.pop("params", {})
        if not isinstance(params, dict):
            logger.error("param error: params must be dict. ")
            return None

        collection_name = kwargs.pop("collection_name", "rag_sdk")
        search_mode = kwargs.pop("search_mode", SearchMode.DENSE)

        milvus_db = MilvusDB(client, collection_name=collection_name, search_mode=search_mode)
        if (milvus_db.search_mode == SearchMode.DENSE or milvus_db.search_mode == SearchMode.HYBRID) and x_dim is None:
            logger.error("param error: x_dim can't be None under DENSE or HYBRID search mode")
            return None

        try:
            milvus_db.create_collection(x_dim=x_dim, similarity_strategy=ss, params=params)
        except KeyError:
            logger.error("milvus create collection meet key error")
        except Exception as e:
            logger.error(f"milvus create collection failed: {e}")

        return milvus_db

    @staticmethod
    def _create_schema_dense(x_dim):
        builder = SchemaBuilder()
        builder.add_vector_field("vector", DataType.FLOAT_VECTOR, x_dim)
        return builder.build()

    @staticmethod
    def _create_schema_sparse():
        builder = SchemaBuilder()
        builder.add_sparse_vector_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        return builder.build()

    @staticmethod
    def _create_schema_hybrid(x_dim):
        builder = SchemaBuilder()
        builder.add_vector_field("vector", DataType.FLOAT_VECTOR, x_dim)
        builder.add_sparse_vector_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        return builder.build()

    @validate_params(collection_name=dict(validator=lambda x: 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
                                          message="param length range (0, 1024]"))
    def set_collection_name(self, collection_name: str):
        self._collection_name = collection_name

    @validate_params(
        x_dim=dict(validator=lambda x: x is None or (isinstance(x, int) and 0 < x <= MAX_VEC_DIM),
                   message="param value range (0, 1024 * 1024]"))
    def create_collection(self, x_dim: Optional[int] = None,
                          similarity_strategy: Optional[SimilarityStrategy] = None, params=None):
        if (self.search_mode == SearchMode.DENSE or self.search_mode == SearchMode.HYBRID) and x_dim is None:
            raise MilvusError("x_dim can't be None in mode DENSE or HYBRID")
        if similarity_strategy is None:
            similarity_strategy = SimilarityStrategy.FLAT_IP
        similarity = self.SIMILARITY_STRATEGY_MAP.get(similarity_strategy)
        self.score_scale = similarity.get("scale")
        if params is None:
            params = {}
        schema = self._create_schema(x_dim)
        index_params = self._prepare_index_params(similarity, params)
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )

    def drop_collection(self):
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")
        self.client.drop_collection(self._collection_name)

    @validate_params(ids=dict(validator=lambda x: all(isinstance(it, int) for it in x),
                              message="param must be List[int]"))
    def delete(self, ids: List[int]):
        self._validate_collection_existence()
        if len(ids) >= self.MAX_VEC_NUM:
            raise MilvusError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        res = self.client.delete(collection_name=self._collection_name, ids=ids).get("delete_count")
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success delete ids {ids} in MilvusDB.")
        return res

    @validate_params(
        k=dict(validator=lambda x: 0 < x <= MAX_TOP_K, message="param length range (0, 10000]")
    )
    def search(self, embeddings: Union[List[List[float]], List[Dict[int, float]]], k: int = 3, **kwargs):
        self._validate_collection_existence()
        # Retrieval additional arguments
        output_fields = kwargs.pop("output_fields", [])

        if isinstance(embeddings, list) and all(isinstance(x, dict) for x in embeddings):
            return self._perform_sparse_search(embeddings, k, output_fields, **kwargs)
        else:
            return self._perform_dense_search(np.array(embeddings), k, output_fields, **kwargs)

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add(self, embeddings: np.ndarray, ids: List[int], docs: Optional[List[str]] = None):
        """往向量数据库添加稠密向量，仅适用于稠密模式或混合模式
        """
        self._validate_collection_existence()
        if self.search_mode != SearchMode.DENSE:
            raise MilvusError("search mode needs to be DENSE")
        data = self._init_insert_data(ids, docs)
        self._handle_dense_input(embeddings, ids, data)
        self.client.insert(collection_name=self._collection_name, data=data)
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success add ids {ids} in MilvusDB.")

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add_sparse(self, ids, sparse_embeddings, docs: Optional[List[str]] = None):
        self._validate_collection_existence()
        if self.search_mode != SearchMode.SPARSE:
            raise MilvusError("search mode must be SPARSE")

        data = self._init_insert_data(ids, docs)
        self._handle_sparse_input(sparse_embeddings, ids, data)
        self.client.insert(collection_name=self._collection_name, data=data)
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success add ids {ids} in MilvusDB.")

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add_dense_and_sparse(self, ids: List[int], dense_embeddings: np.ndarray,
                             sparse_embeddings: List[Dict[int, float]], docs: Optional[List[str]] = None):
        self._validate_collection_existence()
        if self.search_mode != SearchMode.HYBRID:
            raise MilvusError("search mode must be HYBRID")

        data = self._init_insert_data(ids, docs)
        self._handle_sparse_input(sparse_embeddings, ids, data)
        self._handle_dense_input(dense_embeddings, ids, data)
        self.client.insert(collection_name=self._collection_name, data=data)
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success add ids {ids} in MilvusDB.")

    def get_all_ids(self) -> List[int]:
        all_id = self.client.query(self._collection_name, filter="id == 0 or id != 0", output_fields=["id"])
        ids = [idx['id'] for idx in all_id]
        return ids

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, str) for it in x), message="param must be List[str]")
    )
    def get_data_by_ids(self, ids):
        res = self.client.get(
            collection_name=self._collection_name,
            ids=ids
        )
        ids = []
        docs = []
        for data in res:
            ids.append(data["id"])
            docs.append(data["document"])
        return ids, docs

    @validate_params(collection_name=dict(validator=lambda x: 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
                                          message="param length range (0, 1024]"))
    def has_collection(self, collection_name):
        return self.client.has_collection(collection_name)

    def _create_schema(self, x_dim: Optional[int] = None):
        if self.search_mode == SearchMode.DENSE:
            return self._create_schema_dense(x_dim)
        if self.search_mode == SearchMode.SPARSE:
            return self._create_schema_sparse()
        return self._create_schema_hybrid(x_dim)

    def _prepare_index_params(self, similarity, params):
        builder = IndexParamsBuilder(self.client, self.search_mode)
        return builder.prepare_index_params(similarity, params)

    def _init_insert_data(self, ids, docs):
        data = [{"id": i} for i in ids]
        if docs is not None:
            self._validate_docs(docs)
            for i, doc in enumerate(docs):
                data[i]["document"] = doc
        return data

    def _validate_collection_existence(self):
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")

    def _validate_dense_input(self, data, search=True):
        if not isinstance(data, np.ndarray):
            raise ValueError("param must be np.ndarray")
        if len(data.shape) != 2:
            raise ValueError("shape of embedding must equal to 2")
        limit = self.MAX_SEARCH_BATCH if search else self.MAX_VEC_NUM
        if data.shape[0] >= limit:
            raise ValueError(f"num of embeddings must less {limit}")

    def _validate_sparse_input(self, data):
        if not (isinstance(data, list) and all(isinstance(x, dict) for x in data)):
            raise ValueError(
                f"param must be List[Dict] with max length {self.MAX_DICT_LENGTH}"
            )

    def _validate_docs(self, data):
        ret = validata_list_str(
            data,
            [1, self.MAX_SEARCH_BATCH],
            [1, self.MAX_QUERY_LENGTH]
        )
        if not ret:
            raise ValueError(
                f"param must be List[str] with max length {self.MAX_SEARCH_BATCH} "
                f"and each string length in [1, {self.MAX_QUERY_LENGTH}]"
            )

    def _handle_dense_input(self, embeddings: Optional[np.ndarray], ids: List[int], data: List[Dict]):
        self._validate_dense_input(embeddings, search=False)
        if embeddings.shape[0] != len(ids):
            raise MilvusError("Length of embeddings is not equal to number of ids")
        for i, e in enumerate(embeddings.astype(np.float32)):
            data[i]["vector"] = e

    def _handle_sparse_input(self, sparse_embeddings: Optional[List[Dict[int, float]]],
                             ids: List[int], data: List[Dict]):
        self._validate_sparse_input(sparse_embeddings)
        if len(sparse_embeddings) != len(ids):
            raise MilvusError("Length of sparse_embeddings is not equal to number of ids")
        for i, x in enumerate(sparse_embeddings):
            data[i]["sparse_vector"] = x

    def _perform_dense_search(self, embeddings: np.ndarray, k: int, output_fields: list, **kwargs):
        """Handle dense search logic."""
        self._validate_dense_input(embeddings)
        embeddings = embeddings.astype(np.float32)
        res = self.client.search(
            collection_name=self._collection_name,
            anns_field="vector",
            limit=k,
            data=embeddings,
            output_fields=output_fields,
            **kwargs
        )
        return self._process_search_results(res, output_fields)

    def _perform_sparse_search(self, sparse_embeddings: List[Dict[int, float]], k: int, output_fields: list, **kwargs):
        """Handle sparse search logic."""
        if self.search_mode not in (SearchMode.SPARSE, SearchMode.HYBRID):
            raise ValueError("Sparse search only supports SPARSE or HYBRID mode")
        self._validate_sparse_input(sparse_embeddings)
        res = self.client.search(
            collection_name=self._collection_name,
            anns_field="sparse_vector",
            limit=k,
            data=sparse_embeddings,
            output_fields=output_fields,
            **kwargs
        )
        return self._process_search_results(res, output_fields)

    def _process_search_results(self, data: List[List[Dict]], output_fields: Optional[List[str]] = None):
        if output_fields is None:
            output_fields = []
        filtered_fields = [field for field in output_fields if field not in ["id", "distance"]]

        scores, ids, extra_data = [], [], []
        for top_k in data:
            k_score, k_id = [], []
            k_extra = [[] for _ in filtered_fields]
            for entity in top_k:
                k_score.append(entity["distance"])
                k_id.append(entity["id"])
                for idx, field in enumerate(filtered_fields):
                    k_extra[idx].append(entity["entity"].get(field, None))
            scores.append(k_score)
            ids.append(k_id)
            extra_data.append(k_extra)
        return self._score_scale(scores), ids, extra_data
