# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import json
from typing import List, Union
from loguru import logger
from pymilvus import MilvusClient, DataType, Function, FunctionType

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import Docstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM, KB, TEXT_MAX_LEN, MAX_TOP_K


class MilvusDocstore(Docstore):
    @validate_params(
        client=dict(validator=lambda x: isinstance(x, MilvusClient),
                    message="param must be instance of MilvusClient"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
            message="param must be str and length range (0, 1024]"),
        enable_bm25=dict(validator=lambda x: isinstance(x, bool),
                    message="param must be instance of bool"),
        bm25_k1=dict(validator=lambda x: isinstance(x, float) and 1.2 <= x <= 2.0,
                     message="param must be be range of [1.2, 2]"),
        bm25_b=dict(validator=lambda x: isinstance(x, float) and 0 <= x <= 1,
                    message="param must be range of [0, 1]"))
    def __init__(self, client: MilvusClient, collection_name: str = "doc_store",
                 enable_bm25=True, bm25_k1: float = 1.2, bm25_b: float = 0.75):
        self._client = client
        self._collection_name = collection_name
        self._enable_bm25 = enable_bm25
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        if not self._client.has_collection(self._collection_name):
            self._create_collection()
        else:
            logger.warning(f"Collection {self._collection_name} already exists")

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
            info = dict(
                page_content=doc.page_content,
                document_id=document_id,
                document_name=doc.document_name,
                metadata=doc.metadata
            )
            if not self._enable_bm25:
                info["sparse_vector"] = {1: 0.1, 2: 0.3}
            data.append(info)
        res = self.client.insert(collection_name=self.collection_name, data=data)
        self.client.refresh_load(collection_name=self.collection_name)
        logger.info(f"Successfully added {res['insert_count']} documents")
        return list(res["ids"])

    def search(self, chunk_id: int) -> Union[MxDocument, None]:
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

    @validate_params(
        query=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
            message=f"param must be str and length range (0, {TEXT_MAX_LEN}]"),
        top_k=dict(
            validator=lambda x: 0 < x <= MAX_TOP_K,
            message="param must be int and must in range (0, 10000]"),
        drop_ratio_search=dict(validator=lambda x: isinstance(x, float) and 0 <= x < 1,
                               message="param must be range of [0, 1)"))
    def full_text_search(self, query: str, top_k: int = 3, drop_ratio_search: float = 0.2) -> List[MxDocument]:
        if not self._enable_bm25:
            logger.error("MilvusDocstore full_text_search failed due to enable_bm25 is False")
            return []
        search_params = {
            # Proportion of small vector values to ignore during the search
            'params': {'drop_ratio_search': drop_ratio_search},
        }
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field='sparse_vector',
            limit=top_k,
            search_params=search_params,
            output_fields=["page_content", "metadata", "document_name"]
        )

        result = []
        if not res:
            return []
        try:
            res = json.dumps(res)
            res = json.loads(res)
            for item in res[0]:
                item["entity"]["metadata"]["score"] = item["distance"]
                result.append(MxDocument(
                    page_content=item["entity"]["page_content"],
                    metadata=item["entity"]["metadata"],
                    document_name=item["entity"]["document_name"]
                ))
            return result
        except json.JSONDecodeError as e:
            logger.error(f"parse data from json format failed!: {e}")
            return []
        except KeyError as e:
            logger.error(f"get result from item failed!: {e}")
            return []
        except Exception as e:
            logger.error(f"exception occurred while full text search: {e}")
            return []

    def delete(self, document_id: int):
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
        if self._enable_bm25:
            analyzer_params_built_in = {
                "type": "chinese",
                "filter": ["cnalphanumonly"],
            }
            schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=60 * KB,
                             enable_analyzer=True, analyzer_params=analyzer_params_built_in)
        else:
            schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=60 * KB)
        schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["page_content"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse_vector"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )

        if self._enable_bm25:
            schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        if not self._enable_bm25:
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"

            )
        else:
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                    "bm25_k1": self.bm25_k1,
                    "bm25_b": self.bm25_b
                },
            )
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )
