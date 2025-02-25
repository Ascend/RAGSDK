# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC

import os
import numpy as np
from loguru import logger
from networkx import DiGraph
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.embeddings import Embeddings
from mx_rag.utils.common import check_db_file_limit, MIN_SQLITE_FREE_SPACE
from mx_rag.utils.file_check import check_disk_free_space
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.vectorstore import SimilarityStrategy


class VectorDBBase(ABC):

    @staticmethod
    def initialize(self, collection_name="", **kwargs):
        pass

    @staticmethod
    def get_data(self, ids: list, **kwargs):
        pass

    @staticmethod
    def create_index(self, graph: DiGraph, **kwargs):
        pass

    @staticmethod
    def search_indexes(self, query: str, k: int, **kwargs):
        pass


class VectorDBError(Exception):
    pass


Base = declarative_base()


class GraphChunkModel(Base):
    __tablename__ = "graph_chunks"
    id = Column(Integer, primary_key=True)
    chunk_content = Column(String, comment="chunk内容")


class MilvusVecDB(VectorDBBase):
    MAX_VEC_NUM = 100 * 1000 * 1000 * 1000
    MAX_SEARCH_BATCH = 1024 * 1024

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
        x_dim = kwargs.get("x_dim", 1024)
        similar_strategy = kwargs.get("similarity_strategy", SimilarityStrategy.FLAT_IP)
        param = kwargs.get("param", None)
        self.milvus_db.set_collection_name(collection_name)
        if self.milvus_db.has_collection(collection_name):
            self.milvus_db.drop_collection()
        self.milvus_db.create_collection(x_dim, similar_strategy, param)

    def get_data(self, ids: list, **kwargs):
        db_ids, docs = self.get_data_by_ids(ids)
        return [str(i) for i in db_ids], docs

    def create_index(self, graph, **kwargs):
        if self.embedding_model is None:
            raise VectorDBError("Embedding model is none")
        if not isinstance(self.embedding_model, Embeddings):
            raise VectorDBError("Embedding model type is not instance of langchain_core.embeddings.Embeddings")

        chunks = []
        indexes = []
        for _, data in graph.nodes.data():
            if "id" in data and "info" in data and data["info"]:
                index = data['id']
                indexes.append(index)
                info = data['info']
                chunks.append(info)
        for i in range(0, len(chunks), self.chunk_size):
            self.add_with_docs(
                np.array(self.embedding_model.embed_documents(chunks[i:i + self.chunk_size])),
                ids=[x + i for x in range(len(indexes[i:i + self.chunk_size]))],
                docs=chunks[i:i + self.chunk_size]
            )

    def search_indexes(self, query, k, **kwargs):
        score_list, id_list, doc_list = \
            self.search_with_docs(np.array(self.embedding_model.embed_documents([query])), k)
        return [str(i) for i in id_list], doc_list

    def add_with_docs(self, embeddings: np.ndarray, ids, docs):
        if len(embeddings.shape) != 2:
            raise VectorDBError("shape of embedding must equal to 2")
        if embeddings.shape[0] != len(ids):
            raise VectorDBError("Length of embeddings is not equal to number of ids")
        if len(ids) >= self.MAX_VEC_NUM:
            raise VectorDBError(f"Length of ids is over limit, {len(ids)} >= {self.MAX_VEC_NUM}")
        if len(docs) >= self.MAX_VEC_NUM:
            raise VectorDBError(f"Length of docs is over limit, {len(docs)} >= {self.MAX_VEC_NUM}")

        embeddings = embeddings.astype(np.float32)
        if not self.milvus_db.client.has_collection(self.milvus_db.collection_name):
            raise VectorDBError(f"collection {self.milvus_db.collection_name} is not existed")

        data = []
        for e, i, doc in zip(embeddings, ids, docs):
            data.append({"vector": e, "id": i, "document": doc})
        self.milvus_db.client.insert(collection_name=self.milvus_db.collection_name, data=data)
        self.milvus_db.client.refresh_load(self.milvus_db.collection_name)
        logger.debug(f"success add ids {ids} in MilvusDB.")

    def search_with_docs(self, embeddings: np.ndarray, k: int = 3):
        if len(embeddings.shape) != 2:
            raise VectorDBError("shape of embedding must equal to 2")
        if embeddings.shape[0] >= self.MAX_SEARCH_BATCH:
            raise VectorDBError(f"num of embeddings must less {self.MAX_SEARCH_BATCH}")
        embeddings = embeddings.astype(np.float32)
        if not self.milvus_db.client.has_collection(self.milvus_db.collection_name):
            raise VectorDBError(f"collection {self.milvus_db.collection_name} is not existed")

        data = self.milvus_db.client.search(
            collection_name=self.milvus_db.collection_name,
            limit=k,
            data=embeddings,
            output_fields=["id", "distance", "document"]
        )
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


class GraphVecMindfaissDB(VectorDBBase):
    def __init__(self, mind_faiss: MindFAISS, embedding_model: Embeddings, db_path: str, dev_list: list[int] = None):
        self.embedding_model = embedding_model
        check_db_file_limit(db_path)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)
        self.dev_list = dev_list if dev_list else [0]
        self.vec_store = mind_faiss
        self.chunk_size = 1024

    def create_index(self, graph: DiGraph, **kwargs):
        if self.embedding_model is None:
            raise VectorDBError("Embedding model is none")
        if not isinstance(self.embedding_model, Embeddings):
            raise VectorDBError("Embedding model type is not supported")
        chunks = []
        ids = []
        for _, data in graph.nodes.data():
            if "id" in data and "info" in data and data["info"]:
                index = data['id']
                ids.append(index)
                info = data['info']
                chunks.append(info)

        for i in range(0, len(chunks), self.chunk_size):
            emb_vec = np.array(self.embedding_model.embed_documents(chunks[i:i + self.chunk_size]))
            id_batch = [x + i for x in range(len(ids[i:i + self.chunk_size]))]
            self.vec_store.add(emb_vec, id_batch)
            self._add_graph_chunk(chunks[i:i + self.chunk_size], id_batch)

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

    def search_indexes(self, query, k, **kwargs):
        score, ids = self.vec_store.search(np.array(self.embedding_model.embed_documents([query])), k)
        chunks = []
        with self.session() as session:
            chunk_table = session.query(GraphChunkModel).filter(GraphChunkModel.id.in_(ids[0]))

            if chunk_table.all():
                for data in chunk_table.all():
                    chunks.append(data.chunk_content)
        return [str(i) for i in ids[0]], chunks

    def _add_graph_chunk(self, batch_chunks, id_batch):
        if check_disk_free_space(os.path.dirname(self.db_path), MIN_SQLITE_FREE_SPACE):
            raise VectorDBError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        with self.session() as session:
            try:
                # id已存在时覆盖
                exist_chunk = session.query(GraphChunkModel).filter(GraphChunkModel.id.in_(id_batch))
                if exist_chunk.all():
                    exist_chunk.delete()
                chunks_table = [GraphChunkModel(chunk_content=doc, id=idx) for doc, idx in zip(batch_chunks, id_batch)]
                session.bulk_save_objects(chunks_table)
                session.commit()
            except SQLAlchemyError as sql_err:
                session.rollback()
                logger.error(f"Database error while adding chunks: {sql_err}")
                raise VectorDBError(f"Failed to add chunks due to database error: {sql_err}") from sql_err
            except Exception as err:
                session.rollback()
                raise VectorDBError(f"add chunk failed, {err}") from err
