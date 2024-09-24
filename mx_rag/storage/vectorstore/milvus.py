# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from pymilvus import MilvusClient, DataType

from mx_rag.storage.vectorstore.vectorstore import VectorStore, SimilarityStrategy
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, MAX_TOP_K, BOOL_TYPE_CHECK_TIP


class MilvusError(Exception):
    pass


class MilvusDB(VectorStore):
    MAX_COLLECTION_NAME_LENGTH = 1024
    MAX_URL_LENGTH = 1024

    SIMILARITY_STRATEGY_MAP = {
        SimilarityStrategy.FLAT_IP:
            {
                "index": "FLAT",
                "metric": "IP",
                "scale": lambda x: x if x <= 1.0 else 1.0
            },
        SimilarityStrategy.FLAT_L2:
            {
                "index": "FLAT",
                "metric": "L2",
                "scale": lambda x: (1.0 - x / 2.0) if x <= 2.0 else 0.0
            },
        SimilarityStrategy.FLAT_COS:
            {
                "index": "FLAT",
                "metric": "COSINE",
                "scale": lambda x: x if x <= 1.0 else 1.0
            }
    }

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_URL_LENGTH,
                 message="param must be str and length range (0, 1024]"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
            message="param must be str and length range (0, 1024]"),
        use_http=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, url: str, collection_name: str = "mxRag", use_http: bool = False, **kwargs):
        super().__init__()
        if url.startswith("http:") and not use_http:
            raise MilvusError("http protocol is not support")
        self.client = MilvusClient(url, **kwargs)
        self._collection_name = collection_name

    @staticmethod
    def create(**kwargs):
        x_dim_name = "x_dim"
        url_name = "url"
        similarity_strategy_name = "similarity_strategy"
        param = "param"

        if x_dim_name not in kwargs or not isinstance(kwargs.get(x_dim_name), int):
            raise KeyError("x_dim param error. ")

        if similarity_strategy_name not in kwargs or \
                not isinstance(kwargs.get(similarity_strategy_name), SimilarityStrategy):
            raise KeyError("similarity_strategy param error. ")

        if url_name not in kwargs or not isinstance(kwargs.get(url_name), str):
            raise KeyError("url param error. ")

        url = kwargs.pop(url_name)
        vector_dims = kwargs.pop(x_dim_name)
        param = kwargs.pop(param)
        similarity_strategy = kwargs.pop(similarity_strategy_name)

        milvus_db = MilvusDB(url, **kwargs)
        milvus_db.create_collection(x_dim=vector_dims, similarity_strategy=similarity_strategy, param=param)
        return milvus_db

    @validate_params(collection_name=dict(validator=lambda x: 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
                                          message="param length range (0, 1024]"))
    def set_collection_name(self, collection_name: str):
        self._collection_name = collection_name

    @validate_params(
        x_dim=dict(validator=lambda x: 0 < x <= MAX_VEC_DIM, message="param value range (0, 1024 * 1024]"),
        similarity_strategy=dict(validator=lambda x: x in MilvusDB.SIMILARITY_STRATEGY_MAP,
                                 message="param must be enum of SimilarityStrategy")
    )
    def create_collection(self, x_dim: int, similarity_strategy: SimilarityStrategy, param=None):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=x_dim)

        similarity = self.SIMILARITY_STRATEGY_MAP.get(similarity_strategy, None)
        if similarity is None:
            raise KeyError(f"index type {similarity_strategy} not support")

        self.score_scale = similarity.get("scale", None)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=similarity.get("index"),
            metric_type=similarity.get("metric"),
            param=param
        )
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
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")
        res = self.client.delete(collection_name=self._collection_name, ids=ids).get("delete_count")
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success delete ids {ids} in MilvusDB.")
        return res

    @validate_params(k=dict(validator=lambda x: 0 < x <= MAX_TOP_K, message="param length range (0, 10000]"))
    def search(self, embeddings: np.ndarray, k: int = 3):
        embeddings = embeddings.astype(np.float32)
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")

        data = self.client.search(
            collection_name=self._collection_name,
            limit=k,
            data=embeddings,
        )
        scores = []
        ids = []
        for top_k in data:
            k_score = []
            k_id = []
            for entity in top_k:
                k_score.append(entity["distance"])
                k_id.append(entity["id"])
            scores.append(k_score)
            ids.append(k_id)
        return self._score_scale(scores), ids

    @validate_params(
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]")
    )
    def add(self, embeddings: np.ndarray, ids: List[int]):
        if embeddings.shape[0] != len(ids):
            raise MilvusError("Length of embeddings is not equal to number of ids")
        embeddings = embeddings.astype(np.float32)
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")

        data = []
        for e, i in zip(embeddings, ids):
            data.append({"vector": e, "id": i})
        self.client.insert(collection_name=self._collection_name, data=data)
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success add ids {ids} in MilvusDB.")

    def get_all_ids(self) -> List[int]:
        all_id = self.client.query(self._collection_name, filter="id == 0 or id != 0", output_fields=["id"])
        ids = [idx['id'] for idx in all_id]
        return ids
