# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import time
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger

from mx_rag.document import LoaderMng
from mx_rag.utils.common import validate_params
from mx_rag.utils.file_check import FileCheck
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker import Reranker
from mx_rag.storage.vectorstore.vectorstore import VectorStore
from mx_rag.storage.vectorstore import MilvusDB, MindFAISS
from mx_rag.gvstore.graph_creator.graph_core import GraphNX
from mx_rag.gvstore.graph_creator.graph_create import GraphCreation
from mx_rag.gvstore.graph_creator.vdb.vector_db import MilvusVecDB, GraphVecMindfaissDB
from mx_rag.gvstore.retrieval.retriever.graph_retrieval import GraphRetrieval
from mx_rag.libs.glib.utils.file_utils import FileCreate

GRAPHDB_TYPE_CLASS = \
    {
        "networkx": GraphNX
    }

MAX_GRAPH_NAME_LENGTH = 1024


class KGEngineError(Exception):
    pass


def _process_content(chunks: List[str]):
    index = 0
    chunks_dict = {}
    for chunk in chunks:
        para_node = {
            "id": index,
            "label": "text",
            "level": 0,
            "info": [],
            "parent": None,
            "children": [],
        }
        para_node["info"].append(chunk)
        chunks_dict[index] = para_node
        index += 1
    return chunks_dict


class KGEngine:
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        embedding_model=dict(validator=lambda x: isinstance(x, Embeddings),
                             message="param must be instance of subclass of langchain_core.embeddings.Embeddings"),
        rerank_model=dict(validator=lambda x: isinstance(x, Reranker) or x is None,
                          message="param must be instance of mx_rag.reranker.Reranker"),
        vector_db=dict(validator=lambda x: isinstance(x, VectorStore),
                       message="param must be instance of subclass of VectorStore"),
        work_dir=dict(validator=lambda x: isinstance(x, str), message="param must be instance of str")
    )
    def __init__(self, llm, embedding_model, rerank_model, vector_db, work_dir, **kwargs):
        self.llm = llm
        self.embedding_model = embedding_model
        FileCheck.check_path_is_exist_and_valid(work_dir, True, True)
        self.image_save_path = os.path.join(work_dir, "tmp_img")
        result = FileCreate.create_dir(self.image_save_path, 0o750)
        if result.error:
            logger.error(f"{result.error}")
            raise KGEngineError("work_dir invalid, check specification for details")
        self.graphml_save_path = work_dir
        self.contents = None
        self.vector_db = vector_db
        self.rerank_model = rerank_model
        if "lang" in kwargs:
            if not isinstance(kwargs.get("lang"), str):
                raise KeyError("lang param error, it should be str type")
            if kwargs.get("lang") not in ["zh", "en"]:
                raise ValueError(f"lang param error, value must be in [zh, en]")
        self.lang = kwargs.get("lang", "zh")
        self.max_file_count = 100

    @validate_params(
        file_list=dict(validator=lambda x: isinstance(x, list), message="param must be instance of list"),
        loader_mng=dict(validator=lambda x: isinstance(x, LoaderMng), message="param must be instance of LoaderMng")
    )
    def upload_kg_files(self, file_list: list, loader_mng: LoaderMng):
        if len(file_list) > self.max_file_count:
            raise KGEngineError(f'files list length must less than {self.max_file_count}, upload kg files failed')
        txt_files_contents = {}
        for file in file_list:
            FileCheck.check_path_is_exist_and_valid(file)
            if not os.path.isfile(file):
                raise KGEngineError(f"upload kg file failed: '{file}' is not file")
            file_obj = Path(file)
            loader_info = loader_mng.get_loader(file_obj.suffix)
            loader = loader_info.loader_class(file_path=file_obj.as_posix(), **loader_info.loader_params)
            splitter_info = loader_mng.get_splitter(file_obj.suffix)
            splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
            docs = loader.load_and_split(splitter)
            chunks = [doc.page_content for doc in docs if doc.page_content]
            txt_files_contents[file_obj.name] = _process_content(chunks)
        self.contents = txt_files_contents
        return txt_files_contents

    @validate_params(
        graph_name=dict(validator=lambda x: isinstance(x, str) and len(x) <= MAX_GRAPH_NAME_LENGTH,
                        message="graph_name must be instance of str and length range (0, 1024]"),
    )
    def create_kg_graph(self, graph_name: str, **kwargs):
        if not self.contents:
            raise KGEngineError("please call upload_kg_files first")
        FileCheck.check_path_is_exist_and_valid(self.graphml_save_path, True, True)
        graphml_data_path = os.path.join(self.graphml_save_path, f"{graph_name}.graphml")
        if "entity_types" in kwargs and not isinstance(kwargs.get("entity_types"), list):
            raise KeyError("entity_types param error, it should be list[str] type")
        entity_types = kwargs.pop("entity_types", None)
        kwargs["lang"] = self.lang
        graph_creation = GraphCreation(llm=self.llm, entity_types=entity_types, **kwargs)
        extract_graph_start_time = time.time()
        graph_creation.graph_create(graphml_data_path, self.contents)
        extract_graph_end_time = time.time()
        logger.info(f"Graph [{graph_name}] extract takes:{extract_graph_end_time - extract_graph_start_time}")
        index_start_time = time.time()
        graph_client = self._create_graph_client(graph_name, graphml_data_path, **kwargs)
        graph_client.create_graph_index(**kwargs)
        index_end_time = time.time()
        logger.info(f"Graph [{graph_name}] index takes:{index_end_time - index_start_time}")

    @validate_params(
        graph_name=dict(validator=lambda x: isinstance(x, str), message="param must be instance of str"),
        question=dict(validator=lambda x: isinstance(x, str), message="param must be instance of str")
    )
    def do_query(self, graph_name: str, question: str, **kwargs):
        kwargs["lang"] = self.lang
        graphml_data_path = os.path.join(self.graphml_save_path, f"{graph_name}.graphml")
        FileCheck.check_path_is_exist_and_valid(graphml_data_path, True, True)
        graph_client = self._create_graph_client(graph_name, graphml_data_path, **kwargs)
        retriever = GraphRetrieval(graph_name=graph_name, graph=graph_client)
        return retriever.invoke(question)

    def _create_graph_client(self, graph_name: str, graph_path: str, **kwargs):
        graph_client_type = kwargs.get("graph_client_type", "networkx")
        if isinstance(self.vector_db, MilvusDB):
            vec_db = MilvusVecDB(self.vector_db, embedding_model=self.embedding_model)
        elif isinstance(self.vector_db, MindFAISS):
            db_path = os.path.join(self.graphml_save_path, "sql.db")
            vec_db = GraphVecMindfaissDB(mind_faiss=self.vector_db,
                                         embedding_model=self.embedding_model, db_path=db_path)
        else:
            raise TypeError(f"vector_db {self.vector_db} is not support!")
        graph_client = GRAPHDB_TYPE_CLASS.get(graph_client_type)
        if graph_client is None:
            logger.error(f"graph_client_type [{graph_client_type}] unsupported")
            raise ValueError(f"graph type [{graph_client_type}] is unsupported")
        logger.info(f"graph client [{graph_client_type}] created")
        graph = graph_client(graph_path, graph_name, vector_db=vec_db, **kwargs)
        return graph
