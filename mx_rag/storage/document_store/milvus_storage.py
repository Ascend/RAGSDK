# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from typing import List, Union
from loguru import logger
from pymilvus import MilvusClient, DataType

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import Docstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM, KB


class MilvusDocstore(Docstore):
    @validate_params(
        client=dict(validator=lambda x: isinstance(x, MilvusClient),
                    message="param must be instance of MilvusClient"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
            message="param must be str and length range (0, 1024]"))
    def __init__(self, client: MilvusClient, collection_name: str = "doc_store"):
        self._client = client
        self._collection_name = collection_name
        self._create_collection()

    @property
    def client(self):
        return self._client

    @property
    def collection_name(self):
        return self._collection_name

    def drop_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

    @validate_params(
        documents=dict(
            validator=lambda x: 0 < len(x) <= MAX_CHUNKS_NUM and all(isinstance(it, MxDocument) for it in x),
            message="param must be List[MxDocument] and length range in (0, 1000 * 1000]")
    )
    def add(self, documents: List[MxDocument], document_id: int) -> List[int]:
        data = []
        for doc in documents:
            data.append(dict(
                page_content=doc.page_content,
                sparse_vector={1: 0.1, 2: 0.3},  # add dummy sparse vector
                document_id=document_id,
                document_name=doc.document_name,
                metadata=doc.metadata
            ))
        res = self.client.insert(collection_name=self.collection_name, data=data)
        self.client.refresh_load(collection_name=self.collection_name)
        logger.info(f"Successfully added {res['insert_count']} documents")
        return list(res["ids"])

    def search(self, chunk_id) -> Union[MxDocument, None]:
        res = self.client.get(
            collection_name=self.collection_name,
            ids=chunk_id,
            output_fields=["page_content", "metadata", "document_name"]
        )
        result = None
        if res:
            doc = res[0]
            result = MxDocument(
                page_content=doc["page_content"],
                metadata=doc["metadata"],
                document_name=doc["document_name"]
            )
        return result

    def delete(self, document_id):
        """
        Delete all chunks having document_id `document_id`.
        Args:
            document_id: int

        Returns:

        """
        res = self.client.query(
            collection_name=self.collection_name,
            filter=f"document_id == {document_id}",
            output_fields=["id"]
        )
        ids = [x.get("id") for x in res]
        if ids:
            self.client.delete(self.collection_name, ids)
            self.client.refresh_load(self.collection_name)
        return ids

    def get_all_chunk_id(self):
        res = self.client.query(self.collection_name, filter="id == 0 or id != 0", output_fields=["id"])
        return [x.get("id") for x in res]

    def get_all_document_id(self) -> List[int]:
        res = self.client.query(self.collection_name, filter="id == 0 or id != 0", output_fields=["document_id"])
        return [x.get("document_id") for x in res]

    def _create_collection(self):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="document_id", datatype=DataType.INT64)
        schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=60 * KB)
        schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )
