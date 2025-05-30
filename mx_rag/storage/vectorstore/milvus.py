# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

from typing import List, Dict, Optional, Any, Union

import numpy as np
from loguru import logger
from pymilvus import MilvusClient, DataType
from pymilvus.client.types import ExtraList

from mx_rag.storage.vectorstore.vectorstore import VectorStore, SearchMode
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, MAX_TOP_K, _check_sparse_embedding, \
    _check_sparse_and_dense, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.common import validate_list_str


class MilvusError(Exception):
    pass


class SchemaBuilder:
    def __init__(self, auto_id=False):
        self.schema = MilvusClient.create_schema(auto_id=auto_id, enable_dynamic_field=True)
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

    def add_dense_index(self, index_type, metric_type, params: Dict[str, Any]):
        self.index_params.add_index(
            field_name="vector",
            index_name="dense_index",
            index_type=index_type,
            metric_type=metric_type,
            params=params,
        )

    def build(self):
        return self.index_params

    def prepare_index_params(self, index_type, metric_type, params: Dict[str, Any]):
        # Build the index parameters based on search_mode
        if self.search_mode == SearchMode.DENSE:
            self.add_dense_index(index_type, metric_type, params)
        elif self.search_mode == SearchMode.SPARSE:
            self.add_sparse_index(params)
        else:
            self.add_sparse_index(params.get("sparse", {}))
            self.add_dense_index(index_type, metric_type, params.get("dense", {}))

        return self.build()


class MilvusDB(VectorStore):
    MAX_COLLECTION_NAME_LENGTH = 1024
    MAX_URL_LENGTH = 1024
    MAX_QUERY_LENGTH = 1000
    MAX_DICT_LENGTH = 100000

    SCALE_MAP = {
        "IP": lambda x: min(x, 1.0),
        "L2": lambda x: max(1.0 - x / 2.0, 0.0),
        "COSINE": lambda x: min(x, 1.0)
    }

    @validate_params(
        client=dict(validator=lambda x: isinstance(x, MilvusClient),
                    message="param must be instance of MilvusClient"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
            message="param must be str and length range (0, 1024]"),
        search_mode=dict(validator=lambda x: isinstance(x, SearchMode),
                         message="param must be instance of SearchMode"),
        index_type=dict(validator=lambda x: isinstance(x, str) and x in ("FLAT", "IVF_FLAT", "IVF_PQ", "HNSW"),
                        message="param must str and one of [FLAT, IVF_FLAT, IVF_PQ, HNSW]"),
        metric_type=dict(validator=lambda x: isinstance(x, str) and x in ("IP", "L2", "COSINE"),
                         message="param must str and one of  [IP, L2, COSINE]"),
        auto_flush=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
    )
    def __init__(self, client: MilvusClient, collection_name: str = "rag_sdk",
                 search_mode: SearchMode = SearchMode.DENSE, auto_id=False,
                 index_type: str = "FLAT", metric_type: str = "L2", auto_flush=True):
        super().__init__()
        self._client = client
        self._collection_name = collection_name
        self._search_mode = search_mode
        self._auto_id = auto_id
        self._index_type = index_type
        self._metric_type = metric_type
        self._filter_dict = None
        self._auto_flush = auto_flush

        if self.search_mode == SearchMode.SPARSE and (self._index_type != "HNSW" or self._metric_type != "IP"):
            raise ValueError("sparse vector index_type only support HNSW, metric_type only support IP")

    @property
    def search_mode(self):
        return self._search_mode

    @property
    def collection_name(self):
        return self._collection_name

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

        index_type = kwargs.pop("index_type", "FLAT")
        if index_type not in ("FLAT", "IVF_FLAT", "IVF_PQ", "HNSW"):
            logger.error("param error: index_type must be one of [FLAT, IVF_FLAT, IVF_PQ, HNSW]")
            return None

        metric_type = kwargs.pop("metric_type", "L2")
        if metric_type not in ("IP", "L2", "COSINE"):
            logger.error("param error: metric_type must be one of [IP, L2, COSINE]")
            return None

        params = kwargs.pop("params", {})
        if not isinstance(params, dict):
            logger.error("param error: params must be dict. ")
            return None

        collection_name = kwargs.pop("collection_name", "rag_sdk")
        search_mode = kwargs.pop("search_mode", SearchMode.DENSE)
        auto_id = kwargs.pop("auto_id", False)
        auto_flush = kwargs.pop("auto_flush", True)
        milvus_db = MilvusDB(client, collection_name=collection_name, search_mode=search_mode,
                             auto_id=auto_id, index_type=index_type, metric_type=metric_type,
                             auto_flush=auto_flush)

        try:
            milvus_db.create_collection(x_dim=x_dim, params=params)
        except KeyError:
            logger.error("milvus create collection meet key error")
            milvus_db = None
        except Exception as e:
            logger.error(f"milvus create collection failed: {e}")
            milvus_db = None

        return milvus_db

    @validate_params(collection_name=dict(validator=lambda x: 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
                                          message="param length range (0, 1024]"))
    def set_collection_name(self, collection_name: str):
        self._collection_name = collection_name

    @validate_params(
        x_dim=dict(validator=lambda x: x is None or (isinstance(x, int) and 0 < x <= MAX_VEC_DIM),
                   message="param value range (0, 1024 * 1024]"))
    def create_collection(self, x_dim: Optional[int] = None, params=None):
        if (self.search_mode == SearchMode.DENSE or self.search_mode == SearchMode.HYBRID) and x_dim is None:
            raise MilvusError("x_dim can't be None in mode DENSE or HYBRID")

        self.score_scale = self.SCALE_MAP.get(self._metric_type)
        if params is None:
            params = {}
        schema = self._create_schema(x_dim)
        index_params = self._prepare_index_params(params)
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )

    def drop_collection(self):
        if not self.client.has_collection(self._collection_name):
            logger.warning(f"collection {self._collection_name} does not existed")
        else:
            self.client.drop_collection(self._collection_name)

    @validate_params(ids=dict(validator=lambda x: all(isinstance(it, int) for it in x),
                              message="param must be List[int]"))
    def delete(self, ids: List[int]):
        if len(ids) == 0:
            logger.warning("no id need be deleted")
            return 0

        self._validate_collection_existence()
        if len(ids) >= self.MAX_VEC_NUM:
            raise MilvusError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        res = self.client.delete(collection_name=self._collection_name, ids=ids).get("delete_count")
        if self._auto_flush:
            self.flush()
        logger.info(f"success delete {len(ids)} vectors in MilvusDB.")
        return res

    @validate_params(
        k=dict(validator=lambda x: 0 < x <= MAX_TOP_K, message="param length range (0, 10000]"),
        filter_dict=dict(validator=lambda x: isinstance(x, dict) or x is None,
                         message="param filter_dict must be dict or None"))
    def search(self, embeddings: Union[List[List[float]], List[Dict[int, float]]],
               k: int = 3, filter_dict=None, **kwargs):
        self._filter_dict = filter_dict
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
    def add(self, embeddings: np.ndarray, ids: List[int], document_id: int = 0, docs: Optional[List[str]] = None,
            metadatas: Optional[List[Dict]] = None):
        """往向量数据库添加稠密向量，仅适用于稠密模式或混合模式
        """
        self._validate_collection_existence()
        if self.search_mode != SearchMode.DENSE:
            raise MilvusError("search mode needs to be DENSE")
        data = self._init_insert_data(ids, docs, metadatas, document_id)
        self._handle_dense_input(embeddings, ids, data)
        self.client.insert(collection_name=self._collection_name, data=data)
        if self._auto_flush:
            self.flush()
        logger.info(f"success add {len(ids)} ids in MilvusDB.")

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add_sparse(self, ids, sparse_embeddings, document_id: int = 0, docs: Optional[List[str]] = None,
                   metadatas: Optional[List[Dict]] = None):
        self._validate_collection_existence()
        if self.search_mode != SearchMode.SPARSE:
            raise MilvusError("search mode must be SPARSE")

        data = self._init_insert_data(ids, docs, metadatas, document_id)
        self._handle_sparse_input(sparse_embeddings, ids, data)
        self.client.insert(collection_name=self._collection_name, data=data)
        if self._auto_flush:
            self.flush()
        logger.info(f"successfully add {len(ids)} vectors in MilvusDB.")

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add_dense_and_sparse(self, ids: List[int], dense_embeddings: np.ndarray,
                             sparse_embeddings: List[Dict[int, float]], docs: Optional[List[str]] = None,
                             metadatas: Optional[List[Dict]] = None, **kwargs):
        self._validate_collection_existence()
        if self.search_mode != SearchMode.HYBRID:
            raise MilvusError("search mode must be HYBRID")
        document_id = kwargs.pop("document_id", 0)
        if not isinstance(document_id, int):
            raise MilvusError("param document_id must be int")
        data = self._init_insert_data(ids, docs, metadatas, document_id=document_id)
        self._handle_sparse_input(sparse_embeddings, ids, data)
        self._handle_dense_input(dense_embeddings, ids, data)
        self.client.insert(collection_name=self._collection_name, data=data)
        if self._auto_flush:
            self.flush()
        logger.info(f"successfully add {len(ids)} vectors in MilvusDB.")

    def get_all_ids(self) -> List[int]:
        all_id = self.client.query(self._collection_name, filter="id == 0 or id != 0", output_fields=["id"])
        ids = [idx['id'] for idx in all_id]
        return ids

    @validate_params(
        vec_ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="vec_ids must be List[int]"),
        dense=dict(validator=lambda x: x is None or isinstance(x, np.ndarray),
                   message="dense must be Optional[np.ndarray]"),
        sparse=dict(validator=lambda x: x is None or _check_sparse_embedding(x),
                    message="sparse must to be Optional[List[Dict[int, float]]]")
    )
    def update(self, vec_ids: List[int], dense: Optional[np.ndarray] = None,
               sparse: Optional[List[Dict[int, float]]] = None):
        _check_sparse_and_dense(vec_ids, dense, sparse)
        responses = self.client.get(
            collection_name=self.collection_name,
            ids=vec_ids
        )
        if len(responses) != len(vec_ids):
            queried_ids = [res.get("id") for res in responses]
            raise MilvusError(f"the input id {set(vec_ids) - set(queried_ids)} in vec_ids not exists in milvus")
        if dense is None:
            dense = [None] * len(vec_ids)
        if sparse is None:
            sparse = [None] * len(vec_ids)
        for response in responses:
            dense_vector = dense[vec_ids.index(response.get("id"))]
            sparse_vector = sparse[vec_ids.index(response.get("id"))]
            if dense_vector is not None:
                response["vector"] = dense_vector
            if sparse_vector is not None:
                response["sparse_vector"] = sparse_vector
        if responses:
            self.client.upsert(collection_name=self.collection_name, data=responses)
            if self._auto_flush:
                self.flush()
            logger.info(f"Successfully updated chunk ids {vec_ids}")

    @validate_params(collection_name=dict(validator=lambda x: 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
                                          message="param length range (0, 1024]"))
    def has_collection(self, collection_name):
        return self.client.has_collection(collection_name)

    def flush(self):
        self.client.refresh_load(collection_name=self.collection_name)

    def _create_schema_dense(self, x_dim):
        builder = SchemaBuilder(self._auto_id)
        builder.add_vector_field("vector", DataType.FLOAT_VECTOR, x_dim)
        return builder.build()

    def _create_schema_sparse(self):
        builder = SchemaBuilder(self._auto_id)
        builder.add_sparse_vector_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        return builder.build()

    def _create_schema_hybrid(self, x_dim):
        builder = SchemaBuilder(self._auto_id)
        builder.add_vector_field("vector", DataType.FLOAT_VECTOR, x_dim)
        builder.add_sparse_vector_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        return builder.build()

    def _create_schema(self, x_dim: Optional[int] = None):
        if self.search_mode == SearchMode.DENSE:
            return self._create_schema_dense(x_dim)
        if self.search_mode == SearchMode.SPARSE:
            return self._create_schema_sparse()
        return self._create_schema_hybrid(x_dim)

    def _prepare_index_params(self, params):
        builder = IndexParamsBuilder(self.client, self.search_mode)
        return builder.prepare_index_params(self._index_type, self._metric_type, params)

    def _init_insert_data(self, ids, docs, metadatas, document_id):
        data = [{"id": i, "document_id": document_id} for i in ids]
        if docs is not None:
            self._validate_docs(docs)
            if len(ids) != len(docs):
                raise MilvusError("#id must be equal #doc")
            for i, doc in enumerate(docs):
                data[i]["document"] = doc
        if metadatas is not None:
            self._validate_metadatas(metadatas)
            if len(ids) != len(metadatas):
                raise MilvusError("#id must be equal #metadata")
            for i, metadata in enumerate(metadatas):
                data[i]["metadata"] = metadata
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
        ret = validate_list_str(
            data,
            [1, self.MAX_SEARCH_BATCH],
            [1, self.MAX_QUERY_LENGTH]
        )
        if not ret:
            raise ValueError(
                f"param must be List[str] with max length {self.MAX_SEARCH_BATCH} "
                f"and each string length in [1, {self.MAX_QUERY_LENGTH}]"
            )

    def _validate_metadatas(self, data):
        if not (isinstance(data, list) and all(isinstance(it, dict) for it in data)):
            raise MilvusError("param error: param must be list[dict]")
        if len(data) > self.MAX_SEARCH_BATCH:
            raise MilvusError(f"param error: length of list must be less or equal {self.MAX_SEARCH_BATCH}")

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

    def _check_doc_filter(self, doc_filter):
        if not isinstance(doc_filter, list) or not all(isinstance(item, int) for item in doc_filter):
            raise MilvusError("value of 'document_id' in filter_dict must be List[int]")
        doc_filter = list(set(doc_filter))  # 去重
        max_ids_len = len(self.get_all_ids())
        if len(doc_filter) > max_ids_len:
            raise MilvusError(f"length of 'document_id' in filter_dict over than length of ids({max_ids_len})")
        return doc_filter

    def _perform_dense_search(self, embeddings: np.ndarray, k: int, output_fields: list, **kwargs):
        """Handle dense search logic."""
        if self.search_mode not in (SearchMode.DENSE, SearchMode.HYBRID):
            raise ValueError("Sparse search only supports DENSE or HYBRID mode")
        self._validate_dense_input(embeddings)
        embeddings = embeddings.astype(np.float32)
        doc_filter = self._filter_dict.get("document_id", []) if self._filter_dict else []
        doc_filter = self._check_doc_filter(doc_filter)
        search_kwargs = {
            "collection_name": self._collection_name,
            "anns_field": "vector",
            "limit": k,
            "data": embeddings,
            "output_fields": output_fields
        }
        if doc_filter:
            search_kwargs["filter"] = "document_id IN {document_list}"
            search_kwargs["filter_params"] = {"document_list": doc_filter}
        res = self.client.search(**search_kwargs, **kwargs)
        return self._process_search_results(res, output_fields)

    def _perform_sparse_search(self, sparse_embeddings: List[Dict[int, float]], k: int, output_fields: list, **kwargs):
        """Handle sparse search logic."""
        if self.search_mode not in (SearchMode.SPARSE, SearchMode.HYBRID):
            raise ValueError("Sparse search only supports SPARSE or HYBRID mode")
        self._validate_sparse_input(sparse_embeddings)
        doc_filter = self._filter_dict.get("document_id", []) if self._filter_dict else []
        doc_filter = self._check_doc_filter(doc_filter)
        if doc_filter:
            res = self.client.search(
                collection_name=self._collection_name,
                anns_field="sparse_vector",
                limit=k,
                data=sparse_embeddings,
                output_fields=output_fields,
                filter="document_id IN {document_list}",
                filter_params={"document_list": doc_filter},
                **kwargs
            )
        else:
            res = self.client.search(
                collection_name=self._collection_name,
                anns_field="sparse_vector",
                limit=k,
                data=sparse_embeddings,
                output_fields=output_fields,
                **kwargs
            )
        return self._process_search_results(res, output_fields)

    def _process_search_results(self, data: ExtraList, output_fields: Optional[List[str]] = None):
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
                self._append_search_outfields(entity, filtered_fields, k_extra)
            scores.append(k_score)
            ids.append(k_id)
            extra_data.append(k_extra)
        return self._score_scale(scores), ids, extra_data

    def _append_search_outfields(self, entity, filtered_fields, k_extra):
        for idx, field in enumerate(filtered_fields):
            k_extra_value = entity["entity"].get(field, None)
            if k_extra_value is not None:
                k_extra[idx].append({field: k_extra_value})
