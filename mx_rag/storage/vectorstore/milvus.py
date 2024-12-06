# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from pymilvus import MilvusClient, DataType, MilvusException

from mx_rag.storage.vectorstore.vectorstore import VectorStore, SimilarityStrategy
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, MAX_TOP_K, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.file_check import SecFileCheck, FileCheckError
from mx_rag.utils.cert_check import MAX_CERT_LIMIT
from mx_rag.utils.url import is_url_valid


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

        self.client = None

        if not is_url_valid(url, use_http):
            logger.error("check milvus url failed")
            return

        server_pem_path = kwargs.get("server_pem_path", None)
        kwargs.pop("user", None)
        kwargs.pop("password", None)
        kwargs.pop("token", None)

        # 由于milvus 不支持读取加密的私钥文件，直接丢弃
        kwargs.pop("client_key_path", None)

        if not use_http:
            try:
                SecFileCheck(server_pem_path, MAX_CERT_LIMIT).check()
            except FileCheckError as e:
                logger.error(f"check milvus sever pem file failed: {e}")
                return
            except Exception as e:
                logger.error(f"check milvus server pem failed")
                return

        try:
            self.client = MilvusClient(url, **kwargs)
        except MilvusException as e:
            logger.error(f"init milvus client meet exception")
        except Exception as e:
            logger.error(f"init milvus client failed")
            self.client = None
            return

        self._collection_name = collection_name

    @staticmethod
    def create(**kwargs):
        x_dim_name = "x_dim"
        url_name = "url"
        similarity_strategy_name = "similarity_strategy"
        params = "params"

        if x_dim_name not in kwargs or not isinstance(kwargs.get(x_dim_name), int):
            logger.error("x_dim param error. ")
            return None

        if similarity_strategy_name not in kwargs or \
                not isinstance(kwargs.get(similarity_strategy_name), SimilarityStrategy):
            logger.error("similarity_strategy param error. ")
            return None

        if url_name not in kwargs or not isinstance(kwargs.get(url_name), str):
            logger.error("url param error. ")
            return None

        url = kwargs.pop(url_name)
        vector_dims = kwargs.pop(x_dim_name)
        params = kwargs.pop(params, None)
        similarity_strategy = kwargs.pop(similarity_strategy_name)

        milvus_db = MilvusDB(url, **kwargs)

        try:
            milvus_db.create_collection(x_dim=vector_dims, similarity_strategy=similarity_strategy, params=params)
        except KeyError:
            logger.error("milvus create collection meet key error")
        except Exception:
            logger.error("milvus create collection failed")

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
    def create_collection(self, x_dim: int, similarity_strategy: SimilarityStrategy, params=None):
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
            params=params
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
        if len(ids) >= self.MAX_VEC_NUM:
            raise MilvusError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")
        res = self.client.delete(collection_name=self._collection_name, ids=ids).get("delete_count")
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success delete {len(ids)} ids in MilvusDB.")
        return res

    @validate_params(
        k=dict(validator=lambda x: 0 < x <= MAX_TOP_K, message="param length range (0, 10000]"),
        embeddings=dict(validator=lambda x: isinstance(x, np.ndarray), message="embeddings must be np.ndarray type"))
    def search(self, embeddings: np.ndarray, k: int = 3):
        if len(embeddings.shape) != 2:
            raise MilvusError("shape of embedding must equal to 2")
        if embeddings.shape[0] >= self.MAX_SEARCH_BATCH:
            raise MilvusError(f"num of embeddings must less {self.MAX_SEARCH_BATCH}")
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
        ids=dict(validator=lambda x: all(isinstance(it, int) for it in x), message="param must be List[int]"),
        embeddings=dict(validator=lambda x: isinstance(x, np.ndarray), message="embeddings must be np.ndarray type")
    )
    def add(self, embeddings: np.ndarray, ids: List[int]):
        if len(embeddings.shape) != 2:
            raise MilvusError("shape of embedding must equal to 2")
        if embeddings.shape[0] != len(ids):
            raise MilvusError("Length of embeddings is not equal to number of ids")
        if len(ids) >= self.MAX_VEC_NUM:
            raise MilvusError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        embeddings = embeddings.astype(np.float32)
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")

        data = []
        for e, i in zip(embeddings, ids):
            data.append({"vector": e, "id": i})
        self.client.insert(collection_name=self._collection_name, data=data)
        self.client.refresh_load(self._collection_name)
        logger.debug(f"success add {len(ids)} ids in MilvusDB.")

    def get_all_ids(self) -> List[int]:
        all_id = self.client.query(self._collection_name, filter="id == 0 or id != 0", output_fields=["id"])
        ids = [idx['id'] for idx in all_id]
        return ids
