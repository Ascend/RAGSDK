# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Optional, List, Any, Sized, Tuple, Callable, NoReturn, Union

import numpy as np
from pymilvus import MilvusClient, DataType

from mx_rag.vectorstore.vectorstore import VectorStore


class MilvusError(Exception):
    pass


class MilvusDB(VectorStore):

    def __init__(self, url: str, collection_name="mxRag", **kwargs):
        if url.startswith("http:"):
            raise MilvusError("http protocol is not support")
        self.client = MilvusClient(url, **kwargs)
        self._collection_name = collection_name

    def set_collection_name(self, collection_name: str):
        self._collection_name = collection_name

    def create_collection(self, x_dim, index_type, metric_type, param=None):
        if self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} has been created")

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=x_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type=metric_type,
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

    def delete(self, ids):
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")
        res = self.client.delete(collection_name=self._collection_name, ids=ids).get("delete_count")
        self.client.refresh_load(self._collection_name)
        return res

    def search(self, embeddings: Union[np.ndarray, List[list], list], k: int = 3):
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
        return scores, ids

    def add(self, embeddings, ids):
        if not self.client.has_collection(self._collection_name):
            raise MilvusError(f"collection {self._collection_name} is not existed")

        data = []
        for e, i in zip(embeddings, ids):
            data.append({"vector": e, "id": i})
        self.client.insert(collection_name=self._collection_name, data=data)
        self.client.refresh_load(self._collection_name)
