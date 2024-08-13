# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
from typing import List, Callable, Optional, NoReturn

import numpy as np
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

from mx_rag.knowledge.base_knowledge import KnowledgeBase, KnowledgeError
from mx_rag.retrievers import TreeBuilderConfig, TreeBuilder
from mx_rag.retrievers.tree_retriever import Tree

from mx_rag.storage.document_store.base_storage import Docstore, MxDocument
from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.file_operate import check_disk_free_space
from mx_rag.storage.vectorstore import VectorStore

Base = declarative_base()


class KnowledgeMgrModel(Base):
    __tablename__ = "knowledgeMgr_table"

    id = Column(Integer, primary_key=True)
    knowledge_name = Column(String, comment="知识库名称", unique=True)


class KnowledgeModel(Base):
    __tablename__ = "knowledge_table"

    id = Column(Integer, primary_key=True)
    knowledge_name = Column(String, comment="知识库名称")
    document_name = Column(String, comment="文档名称", unique=True)


class KnowledgeStore:

    FREE_SPACE_LIMIT = 200 * 1024 * 1024

    def __init__(self, db_path):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    def add(self, knowledge_name, doc_name):
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            raise KnowledgeError("Insufficient remaining space, please clear disk space")
        with self.session() as session:
            try:
                session.add(KnowledgeModel(knowledge_name=knowledge_name, document_name=doc_name))
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add chunk failed, {err}") from err

    def delete(self, knowledge_name, doc_name):
        with self.session() as session:
            try:
                session.query(KnowledgeModel).filter_by(
                    knowledge_name=knowledge_name, document_name=doc_name).delete()
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

    def get_all(self, knowledge_name):
        with self.session() as session:
            ret = []
            for doc in session.query(
                    KnowledgeModel.document_name
            ).filter_by(knowledge_name=knowledge_name).distinct().all():
                ret.append(doc[0])
            return ret

    def check_document_exist(self, knowledge_name, doc_name) -> bool:
        with self.session() as session:
            chunk = session.query(KnowledgeModel).filter_by(
                knowledge_name=knowledge_name, document_name=doc_name).first()
            return True if chunk is not None else False


class KnowledgeDB(KnowledgeBase):

    def __init__(
            self,
            knowledge_store: KnowledgeStore,
            chunk_store: Docstore,
            vector_store: VectorStore,
            knowledge_name: str,
            white_paths: List[str],
            max_loop_limit: int = 1000,
    ):
        super().__init__(white_paths)
        self._knowledge_store = knowledge_store
        self._vector_store = vector_store
        self._document_store = chunk_store
        self.max_loop_limit = max_loop_limit
        self.knowledge_name = knowledge_name

    def get_all_documents(self):
        """获取当前已上传的所有文档"""
        return self._knowledge_store.get_all(self.knowledge_name)

    def add_file(
            self,
            doc_name: str,
            texts: List[str],
            embed_func: Callable[[List[str]], List[List[float]]],
            metadatas: Optional[List[dict]] = None
    ) -> NoReturn:
        embeddings = embed_func(texts)
        if not isinstance(embeddings, List):
            raise KnowledgeError("The data type of embedding should be np.ndarray")
        metadatas = metadatas or [{} for _ in texts]
        if len(texts) != len(metadatas) != len(embeddings):
            raise KnowledgeError("texts, metadatas, embeddings expected to be equal length")
        documents = [MxDocument(page_content=t, metadata=m, document_name=doc_name) for t, m in zip(texts, metadatas)]
        self._knowledge_store.add(self.knowledge_name, doc_name)
        ids = self._document_store.add(documents)
        self._vector_store.add(np.array(embeddings), np.array(ids))

    def delete_file(self, doc_name: str):
        self._knowledge_store.delete(self.knowledge_name, doc_name)
        ids = self._document_store.delete(doc_name)
        num_removed = self._vector_store.delete(ids)
        if len(ids) != num_removed:
            logger.warning("the number of documents does not match the number of vectors")

    def check_document_exist(self, doc_name: str) -> bool:
        return self._knowledge_store.check_document_exist(self.knowledge_name, doc_name)


class KnowledgeTreeDB(KnowledgeBase):
    def __init__(
            self,
            knowledge_store: KnowledgeStore,
            chunk_store: Docstore,
            knowledge_name: str,
            white_paths: List[str],
            tree_builder_config: TreeBuilderConfig
    ):
        super().__init__(white_paths)
        self._knowledge_store = knowledge_store
        self._document_store = chunk_store
        self.knowledge_name = knowledge_name
        self.tree_builder_config = tree_builder_config
        self.tree_builder = TreeBuilder(tree_builder_config)

    def get_all_documents(self):
        """获取当前已上传的所有文档"""
        return self._knowledge_store.get_all(self.knowledge_name)

    def add_files(self,
                  file_names: List[str],
                  texts: List[str],
                  embed_func: Callable[[List[str]], np.ndarray],
                  metadatas: List[dict]) -> Tree:
        if len(texts) != len(metadatas) != len(file_names):
            raise KnowledgeError("chunks, metadatas, file_names expected to be equal length")
        # 需要将索引tree返回
        tree = self.tree_builder.build_from_text(embed_func, chunks=texts)
        documents = []
        for text, metadata, file_name in zip(texts, metadatas, file_names):
            documents.append(MxDocument(page_content=text, metadata=metadata, document_name=file_name))
        for file_name in list(set(file_names)):
            self.add_file(file_name, [], None, [])
        self._document_store.add(documents)
        return tree

    def add_file(self, doc_name: str, texts: Optional[List[str]],
                 embed_func: Optional[Callable[[List[str]], np.ndarray]], metadatas: Optional[List[dict]]):
        self._knowledge_store.add(self.knowledge_name, doc_name)

    def delete_file(self, doc_name: str):
        self._knowledge_store.delete(self.knowledge_name, doc_name)
        self._document_store.delete(doc_name)

    def check_document_exist(self, doc_name: str) -> bool:
        return self._knowledge_store.check_document_exist(self.knowledge_name, doc_name)


class KnowledgeMgrStore:

    FREE_SPACE_LIMIT = 200 * 1024 * 1024

    def __init__(self, db_path):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    def add(self, knowledge_name):
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            raise KnowledgeError("Insufficient remaining space, please clear disk space")
        with self.session() as session:
            try:
                session.add(KnowledgeMgrModel(knowledge_name=knowledge_name))
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add knowledge failed, {err}") from err

    def delete(self, knowledge_name):
        with self.session() as session:
            try:
                session.query(KnowledgeMgrModel).filter_by(knowledge_name=knowledge_name).delete()
                session.commit()
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

    def get_all(self):
        with self.session() as session:
            names = []
            for k in session.query(KnowledgeMgrModel).all():
                names.append(k.knowledge_name)
            return names


class KnowledgeMgr:

    def __init__(self, mgr_store: KnowledgeMgrStore):
        self.mgr_store = mgr_store

    def register(self, knowledge: KnowledgeDB):
        if knowledge.knowledge_name in self.mgr_store.get_all():
            raise KnowledgeError(f"knowledge {knowledge.knowledge_name} has been registered")
        self.mgr_store.add(knowledge.knowledge_name)

    def delete(self, knowledge: KnowledgeDB):
        if knowledge.knowledge_name not in self.mgr_store.get_all():
            raise KnowledgeError(f"knowledge {knowledge.knowledge_name} is not be registered")
        if knowledge.get_all_documents():
            raise KnowledgeError(f"please clear knowledge, before delete")
        self.mgr_store.delete(knowledge.knowledge_name)

    def get_all(self):
        return self.mgr_store.get_all()
