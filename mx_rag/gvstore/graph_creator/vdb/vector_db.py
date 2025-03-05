# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC, abstractmethod

import os
from typing import List

import numpy as np
from loguru import logger
from networkx import DiGraph
from sqlalchemy import create_engine, Column, Integer, String, URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from langchain_core.embeddings import Embeddings

from mx_rag.gvstore.util.utils import GraphUpdatedData
from mx_rag.utils.common import check_db_file_limit, MIN_SQLITE_FREE_SPACE
from mx_rag.utils.file_check import check_disk_free_space
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.vectorstore import SimilarityStrategy


class VectorDBBase(ABC):

    @abstractmethod
    def initialize(self, collection_name="", **kwargs):
        pass

    @abstractmethod
    def get_data(self, ids: list, **kwargs):
        pass

    @abstractmethod
    def create_index(self, graph: DiGraph, **kwargs):
        pass

    @abstractmethod
    def search_indexes(self, query: str, k: int):
        pass

    @abstractmethod
    def add_embedding(self, entity_list: list, id_list: list, partition_name: str):
        pass

    @abstractmethod
    def query_embedding(self, collection_name, entity_list: list, **kwargs):
        pass

    @abstractmethod
    def row_count(self):
        pass

    @abstractmethod
    def update_index(self, updated_data: GraphUpdatedData):
        pass


class VectorDBError(Exception):
    pass


Base = declarative_base()


class GraphChunkModel(Base):
    __tablename__ = "graph_chunks"
    id = Column(Integer, primary_key=True)
    chunk_content = Column(String, comment="chunk内容")
    label = Column(String, comment="label标签")


class MilvusVecDB(VectorDBBase):
    MAX_VEC_NUM = 100 * 1000 * 1000 * 1000
    MAX_SEARCH_BATCH = 1024 * 1024
    MAX_PARTITION_NAME_LENGTH = 1024

    def __init__(self, milvus_db, **kwargs):
        self.embedding_model = kwargs.get("embedding_model", None)
        self.milvus_db = milvus_db
        self.chunk_size = 2000

    def initialize(self, collection_name: str, **kwargs):
        if "x_dim" in kwargs and not isinstance(kwargs.get("x_dim"), int):
            raise KeyError("x_dim param error, it should be integer type")
        if "similarity_strategy" in kwargs and not isinstance(kwargs.get("similarity_strategy"), SimilarityStrategy):
            raise KeyError("similarity_strategy parameter error, it should be SimilarityStrategy type")
        if "param" in kwargs and not isinstance(kwargs.get("param"), dict):
            raise KeyError("param param error, it should be valid dict type")
        if "force" in kwargs and not isinstance(kwargs.get("force"), bool):
            raise KeyError("param force error, it should be bool type")
        x_dim = kwargs.get("x_dim", 1024)
        similar_strategy = kwargs.get("similarity_strategy", SimilarityStrategy.FLAT_IP)
        param = kwargs.get("param", None)
        self.milvus_db.set_collection_name(collection_name)
        force = kwargs.get("force", False)
        has_collection_flag = self.milvus_db.has_collection(collection_name)
        if not has_collection_flag:
            self.milvus_db.create_collection(x_dim, similar_strategy, param)
        if has_collection_flag:
            if force:
                logger.warning(f"force overwrite collection {collection_name} in milvus.")
                self.milvus_db.drop_collection()
                self.milvus_db.create_collection(x_dim, similar_strategy, param)
            else:
                logger.warning(f"collection {collection_name} in milvus already exists.")

    def get_data(self, ids: list, **kwargs):
        db_ids, docs = self.get_data_by_ids(ids)
        return [str(i) for i in db_ids], docs

    def create_index(self, graph, **kwargs):
        if self.embedding_model is None:
            raise VectorDBError("Embedding model is none")
        if not isinstance(self.embedding_model, Embeddings):
            raise VectorDBError("Embedding model type is not instance of langchain_core.embeddings.Embeddings")

        node_list = [data for _, data in graph.nodes.data()]
        self._insert_data(node_list)

    def search_indexes(self, query, k):
        partition_names = ["text"]
        score_list, id_list, doc_list = \
            self.search_with_docs(np.array(self.embedding_model.embed_documents([query])), k, partition_names)
        return score_list, id_list, doc_list

    def add_embedding(self, entity_list: list, id_list: list, partition_name: str):
        if not self.has_partition(partition_name=partition_name):
            self.create_partition(partition_name)
        for i in range(0, len(entity_list), self.chunk_size):
            self.add_with_docs(
                np.array(self.embedding_model.embed_documents(entity_list[i:i + self.chunk_size])),
                ids=id_list[i:i + self.chunk_size],
                docs=entity_list[i:i + self.chunk_size],
                partition_name=partition_name
            )

    def query_embedding(self, collection_name, entity_list: list, **kwargs):
        partition_names = kwargs.get("partition_names", ["text"])
        score_list, id_list, doc_list = self.search_with_docs(
            np.array(self.embedding_model.embed_documents(entity_list)), 1, partition_names)
        return list(zip(id_list, entity_list, doc_list, score_list))

    def row_count(self):
        res = self.collection_stats()
        return res[0]["count(*)"]

    def update_index(self, updated_data: GraphUpdatedData):
        added_nodes = updated_data.added_nodes
        self._insert_data(added_nodes)

    def add_with_docs(self, embeddings: np.ndarray, ids, docs, partition_name=None):
        if len(embeddings.shape) != 2:
            raise VectorDBError("shape of embedding must equal to 2")
        if embeddings.shape[0] != len(ids):
            raise VectorDBError("Length of embeddings is not equal to number of ids")
        if len(ids) >= self.MAX_VEC_NUM:
            raise VectorDBError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        if len(docs) >= self.MAX_VEC_NUM:
            raise VectorDBError(f"Length of docs is over limit, {len(docs)} >= {self.MAX_VEC_NUM}")
        if partition_name and len(partition_name) >= self.MAX_PARTITION_NAME_LENGTH:
            raise VectorDBError(
                f"length of partition_name is over limit, {len(partition_name)} >= {self.MAX_PARTITION_NAME_LENGTH}")

        embeddings = embeddings.astype(np.float32)
        if not self.milvus_db.client.has_collection(self.milvus_db.collection_name):
            raise VectorDBError(f"collection {self.milvus_db.collection_name} is not existed")

        data = []
        for e, i, doc in zip(embeddings, ids, docs):
            data.append({"vector": e, "id": i, "document": doc})
        if not partition_name:
            self.milvus_db.client.insert(collection_name=self.milvus_db.collection_name, data=data)
        else:
            self.milvus_db.client.insert(collection_name=self.milvus_db.collection_name, data=data,
                                         partition_name=partition_name)
        self.milvus_db.client.refresh_load(self.milvus_db.collection_name)
        logger.debug(f"success add ids {ids} in MilvusDB.")

    def search_with_docs(self, embeddings: np.ndarray, k: int = 3, partition_names=None):
        if len(embeddings.shape) != 2:
            raise VectorDBError("shape of embedding must equal to 2")
        if embeddings.shape[0] >= self.MAX_SEARCH_BATCH:
            raise VectorDBError(f"num of embeddings must less {self.MAX_SEARCH_BATCH}")
        embeddings = embeddings.astype(np.float32)
        if not self.milvus_db.client.has_collection(self.milvus_db.collection_name):
            raise VectorDBError(f"collection {self.milvus_db.collection_name} is not existed")

        if partition_names:
            for partition in partition_names:
                if len(partition) >= self.MAX_PARTITION_NAME_LENGTH:
                    raise VectorDBError(f"length of partition_name is over limit, "
                                        f"{len(partition)} >= {self.MAX_PARTITION_NAME_LENGTH}")
        if not partition_names:
            data = self.milvus_db.client.search(
                collection_name=self.milvus_db.collection_name, limit=k,
                data=embeddings, output_fields=["id", "distance", "document"])
        else:
            data = self.milvus_db.client.search(
                collection_name=self.milvus_db.collection_name, limit=k, partition_names=partition_names,
                data=embeddings, output_fields=["id", "distance", "document"])
        scores = []
        ids = []
        docs = []
        for top_k in data:
            k_score = []
            k_id = []
            k_doc = []
            for entity in top_k:
                k_score.append(entity["distance"])
                k_id.append(entity["id"])
                k_doc.append(entity["entity"]["document"])
            scores.extend(k_score)
            ids.extend(k_id)
            docs.extend(k_doc)
        return scores, ids, docs

    def get_data_by_ids(self, ids):
        res = self.milvus_db.client.get(
            collection_name=self.milvus_db.collection_name,
            ids=ids
        )
        ids = []
        docs = []
        for data in res:
            ids.append(data["id"])
            docs.append(data["document"])
        return ids, docs

    def collection_stats(self):
        counts = self.milvus_db.client.query(collection_name=self.milvus_db.collection_name, output_fields=["count(*)"])
        return counts

    def create_partition(self, partition_name: str):
        res = self.milvus_db.client.has_partition(collection_name=self.milvus_db.collection_name,
                                                  partition_name=partition_name)
        if res:
            raise VectorDBError(f"Partition: {partition_name} already exists in {self.milvus_db.collection_name}")
        self.milvus_db.client.create_partition(collection_name=self.milvus_db.collection_name,
                                               partition_name=partition_name)

    def has_partition(self, partition_name: str):
        return self.milvus_db.client.has_partition(collection_name=self.milvus_db.collection_name,
                                                   partition_name=partition_name)

    def _insert_data(self, nodes: list):
        partitions = ["text", "entity"]
        chunks = {}
        indexes = {}
        for name in partitions:
            if not self.has_partition(name):
                self.create_partition(name)
            chunks[name] = []
            indexes[name] = []

        for data in nodes:
            node_id = data.get("id", None)
            node_info = data.get("info", None)
            if node_id is not None and node_info:
                node_label = data.get("label", None)
                if node_label and node_label in ["text", "table"]:
                    indexes["text"].append(node_id)
                    chunks["text"].append(node_info)
                else:
                    indexes["entity"].append(node_id)
                    chunks["entity"].append(node_info)
        for data in partitions:
            cur_chunks = chunks.get(data, None)
            cur_indexes = indexes.get(data, None)
            if not cur_chunks or not cur_indexes:
                logger.error("data to be inserted milvus not correct")
                continue
            for i in range(0, len(cur_chunks), self.chunk_size):
                self.add_with_docs(
                    np.array(self.embedding_model.embed_documents(cur_chunks[i:i + self.chunk_size])),
                    ids=[x for x in cur_indexes[i:i + self.chunk_size]],
                    docs=cur_chunks[i:i + self.chunk_size],
                    partition_name=data
                )


class GraphVecMindfaissDB(VectorDBBase):
    def __init__(self, mind_faiss: MindFAISS, embedding_model: Embeddings, db_path: str, dev_list: list[int] = None):
        self.embedding_model = embedding_model
        check_db_file_limit(db_path)
        self.db_path = db_path
        engine = create_engine(url=URL.create("sqlite", database=db_path))
        self.session = scoped_session(sessionmaker(bind=engine))
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)
        self.dev_list = dev_list if dev_list else [0]
        self.vec_store = mind_faiss
        self.chunk_size = 1024

    def initialize(self, collection_name="", **kwargs):
        self._check_store_accordance()

    def query_embedding(self, collection_name, entity_list: list, **kwargs):
        scores = []
        ids = []
        docs = []
        for entity in entity_list:
            index, score, chunk = self.search_indexes(entity, 1)
            # 默认取top1
            scores.append(score[0])
            ids.append(index[0])
            docs.extend(chunk)
        return list(zip(ids, entity_list, docs, scores))

    def row_count(self):
        return len(self.get_all_ids())

    def create_index(self, graph: DiGraph, **kwargs):
        chunks = []
        ids = []
        labels = []
        for _, data in graph.nodes.data():
            if "id" in data and "info" in data and data["info"]:
                index = data['id']
                ids.append(index)
                info = data['info']
                chunks.append(info)
                labels.append(data['label'])
        self._add_index(ids, chunks, labels)

    def get_data(self, ids: list, **kwargs):
        with self.session() as session:
            chunk_table = session.query(GraphChunkModel).filter(GraphChunkModel.id.in_(ids)).all()
            chunks = []
            searched_id = []
            if chunk_table:
                for data in chunk_table:
                    chunks.append(data.chunk_content)
                    searched_id.append(data.id)
            return [str(i) for i in searched_id], chunks

    def search_indexes(self, query, k):
        lables = ["text"]
        scores, ids = self.vec_store.search(np.array(self.embedding_model.embed_documents([query])), k)
        chunks = []
        with (self.session() as session):
            chunk_table = session.query(GraphChunkModel).filter(GraphChunkModel.id.in_(ids[0])
                                                                ).filter(GraphChunkModel.label.in_(lables))
            if chunk_table.all():
                for data in chunk_table.all():
                    chunks.append(data.chunk_content)
        return scores[0], ids[0], chunks

    def add_embedding(self, entity_list: list, id_list: list, partition_name: str):

        for i in range(0, len(entity_list), self.chunk_size):
            chunks = entity_list[i:i + self.chunk_size]
            embed_list = np.array(self.embedding_model.embed_documents(chunks))
            ids = id_list[i:i + self.chunk_size]
            self.vec_store.add(embed_list, ids)
            labels = [partition_name] * len(ids)
            self._add_graph_chunk(chunks, ids, labels)

    def get_all_ids(self):
        with self.session() as session:
            query = session.query(GraphChunkModel.id).yield_per(self.chunk_size)
        return [chunk_id for (chunk_id,) in query]

    def update_index(self, updated_data: GraphUpdatedData):
        chunks = []
        ids = []
        labels = []
        for node in updated_data.added_nodes:
            node_id = node.get("id", None)
            node_info = node.get("info", None)
            labels.append(node.get("label", ""))
            if node_id and node_info:
                chunks.append(node_info)
                ids.append(node_id)
        self._add_index(ids, chunks, labels)

    def _add_graph_chunk(self, batch_chunks: list, id_batch: list, labels: list):
        if check_disk_free_space(os.path.dirname(self.db_path), MIN_SQLITE_FREE_SPACE):
            raise VectorDBError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        with self.session() as session:
            try:
                # id已存在时覆盖
                exist_chunk = session.query(GraphChunkModel).filter(GraphChunkModel.id.in_(id_batch))
                if exist_chunk.all():
                    exist_chunk.delete()
                chunks_table = [GraphChunkModel(chunk_content=doc, id=idx, label=label)
                                for doc, idx, label in zip(batch_chunks, id_batch, labels)]
                session.bulk_save_objects(chunks_table)
                session.commit()
            except SQLAlchemyError as sql_err:
                session.rollback()
                logger.error(f"Database error while adding chunks: {sql_err}")
                raise VectorDBError(f"Failed to add chunks due to database error: {sql_err}") from sql_err
            except Exception as err:
                session.rollback()
                raise VectorDBError(f"add chunk failed, {err}") from err

    def _add_index(self, ids: List[int], chunks: List[str], labels: List[str]):
        if self.embedding_model is None:
            raise VectorDBError("Embedding model is none")
        if not isinstance(self.embedding_model, Embeddings):
            raise VectorDBError("Embedding model type is not supported")
        for i in range(0, len(chunks), self.chunk_size):
            emb_vec = np.array(self.embedding_model.embed_documents(chunks[i:i + self.chunk_size]))
            id_batch = ids[i:i + self.chunk_size]
            self.vec_store.add(emb_vec, id_batch)
            self._add_graph_chunk(chunks[i:i + self.chunk_size], id_batch, labels[i:i + self.chunk_size])

    def _check_store_accordance(self) -> None:
        doc_chunks = self.get_all_ids()
        vec_ids = self.vec_store.get_all_ids()
        if set(doc_chunks) != set(vec_ids):
            logger.error(f"The id of chunk in the sqlite database is different from id in the vector database. "
                         f"the sqlite db path: {self.db_path}, the vectore db path: {self.vec_store.get_save_file()}.")
            raise VectorDBError("The IDs of the sqlite database and vector database are inconsistent.")
