#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import json
from typing import List, Optional
from pathlib import Path

from loguru import logger
from langchain_core.embeddings import Embeddings
from langchain_opengauss import OpenGaussSettings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from mx_rag.storage.document_store.base_storage import StorageError
from mx_rag.storage.vectorstore import VectorStorageFactory
from mx_rag.storage.vectorstore.vectorstore import VectorStore
from mx_rag.utils import Lang
from mx_rag.graphrag.relation_extraction import LLMRelationExtractor
from mx_rag.graphrag.graph_merger import GraphMerger
from mx_rag.graphrag.concept_graph_merger import ConceptGraphMerger
from mx_rag.graphrag.graphs.networkx_graph import NetworkxGraph
from mx_rag.graphrag.graphs.opengauss_graph import OpenGaussGraph
from mx_rag.graphrag.graph_conceptualizer import GraphConceptualizer
from mx_rag.graphrag.concept_clustering import ConceptCluster
from mx_rag.graphrag.concept_embedding import ConceptEmbedding
from mx_rag.graphrag.graph_rag_model import GraphRAGModel
from mx_rag.graphrag.vector_stores.vector_store_wrapper import VectorStoreWrapper
from mx_rag.document import LoaderMng
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params, FileCheck, TEXT_MAX_LEN
from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.file_check import check_disk_free_space, FileCheckError


def save_to_json(data, file_path: str):
    try:
        FileCheck.check_input_path_valid(file_path)
        FileCheck.check_filename_valid(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        logger.error(f"Error creating directory or writing file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {e}")


class GraphRAGError(Exception):
    pass


class GraphRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    graph_rag_model: GraphRAGModel

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List:
        return self.graph_rag_model.generate([query])[0]


class GraphRAGPipeline:
    FREE_SPACE_LIMIT = 5 * 1024 * 1024 * 1024  # 5GB

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM),
                 message="llm must be an instance of Text2TextLLM"),
        embedding_model=dict(validator=lambda x: isinstance(x, Embeddings),
                             message="embedding_model must be an instance of Embeddings"),
        rerank_model=dict(validator=lambda x: isinstance(x, Reranker),
                          message="rerank_model must be an instance of Reranker"),
        dim=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1024 * 1024,
                 message="dim must be an integer, value range [1, 1024 * 1024]"),
        graph_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) < 256 and x.isidentifier(),
                        message="graph_name must be a str and length range [1, 255]"),
        graph_type=dict(validator=lambda x: isinstance(x, str) and x in ["networkx", "opengauss"],
                        message="graph_type must be 'networkx' or 'opengauss'")
    )
    def __init__(self, work_dir: str, llm, embedding_model, rerank_model, dim: int,
                 graph_type="networkx", graph_name: str = "graph", **kwargs):
        FileCheck.check_input_path_valid(work_dir)
        FileCheck.check_filename_valid(work_dir)
        if check_disk_free_space(work_dir, self.FREE_SPACE_LIMIT):
            raise StorageError("Insufficient remaining space, please clear disk space")
        self.work_dir = work_dir
        self.graph_name = graph_name
        self.graph_conf = None
        self._setup_save_path(self.graph_name)
        self._setup_graph(self.graph_name, graph_type, **kwargs)
        self.llm = llm
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.concept_embedding = None
        self.docs = []
        self.dim = dim

        self.triple_instructions: Optional[dict] = None
        self.conceptualizer_prompts: Optional[dict] = None
        self.concept_vector_store = None
        self.node_vector_store = None
        self.devs: List[int] = kwargs.pop("devs", [0])
        self._init_vector_store(**kwargs)

    @validate_params(
        file_list=dict(validator=lambda x: isinstance(x, list) and 0 < len(x) <= 100,
                       message="file_list must be list, and length range [1, 100]"),
        loader_mng=dict(validator=lambda x: isinstance(x, LoaderMng), message="param must be instance of LoaderMng")
    )
    def upload_files(self, file_list: list, loader_mng: LoaderMng):
        failed_files = []
        for file in file_list:
            try:
                FileCheck.check_path_is_exist_and_valid(file)
                if not os.path.isfile(file):
                    failed_files.append(file)
                    continue
            except FileCheckError:
                failed_files.append(file)
                continue     
            file_obj = Path(file)
            loader_info = loader_mng.get_loader(file_obj.suffix)
            loader = loader_info.loader_class(file_path=file_obj.as_posix(), **loader_info.loader_params)
            splitter_info = loader_mng.get_splitter(file_obj.suffix)
            splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
            docs = loader.load_and_split(splitter)
            self.docs.extend(docs)
        if failed_files:
            logger.warning(f"{len(failed_files)} files failed to upload, please check: {','.join(failed_files)}")

    @validate_params(
        lang=dict(
            validator=lambda x: isinstance(x, Lang),
            message="param must be a Lang instance",
        ),
        pad_token=dict(
            validator=lambda x: isinstance(x, str) and len(x) < 256,
            message="param must be a string, range [0, 255]",
        ),
        conceptualize=dict(
            validator=lambda x: isinstance(x, bool), message="param must be a boolean"
        ),
    )
    def build_graph(
        self,
        lang: Lang = Lang.EN,
        pad_token: str = "",
        conceptualize: bool = False,
        **kwargs,
    ):
        max_workers = kwargs.pop("max_workers", None)
        top_k = kwargs.pop("top_k", 5)
        threshold = kwargs.pop("threshold", 0.5)
        self.triple_instructions = kwargs.pop("triple_instructions", self.triple_instructions)
        self.conceptualizer_prompts = kwargs.pop("conceptualizer_prompts", self.conceptualizer_prompts)

        if not self.docs:
            raise GraphRAGError("Empty documents, please first run upload_files")
        try:
            extractor = LLMRelationExtractor(
                llm=self.llm,
                pad_token=pad_token,
                language=lang,
                max_workers=max_workers,
                triple_instructions=self.triple_instructions,
            )
            relations = extractor.query(self.docs)
            self.docs = []
            save_to_json(relations, self.relations_save_path)
            logger.info(f"Relations saved: {self.relations_save_path}")

            merger = GraphMerger(self.graph)
            merger.merge(relations, lang)
            merger.save_graph(self.graph_save_path)

            if conceptualize:
                self._process_concepts_and_clusters(lang, top_k, threshold)
            logger.info("Graph built successfully")
        except Exception as e:
            raise GraphRAGError("Graph building failed") from e

    @validate_params(
        question=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                      message=f"param must be a str and its length meets (0, {TEXT_MAX_LEN}]"))
    def retrieve_graph(self, graph_name, graph_type, question: str, **kwargs):
        if self.graph.number_of_nodes() == 0:
            raise GraphRAGError("Empty graph, first build the graph")

        return self.as_retriever(graph_name, graph_type, **kwargs).invoke(question)

    @validate_params(
        graph_name=dict(
            validator=lambda x: isinstance(x, str) and x.isidentifier() and 0 < len(x) < 256,
            message="param must be a str and its length meets [1, 255], "
                    "and only contains letters, _ and digits."
        ),
        graph_type=dict(
            validator=lambda x: isinstance(x, str) and x in ["networkx", "opengauss"],
            message="param only takes from 'networkx' or 'opengauss'"
        )
    )
    def as_retriever(self, graph_name, graph_type, **kwargs):
        self._setup_save_path(graph_name)
        self._setup_graph(graph_name, graph_type)

        use_text = kwargs.pop("use_text", True)
        batch_size = kwargs.pop("batch_size", 512)
        similarity_tail_threshold = kwargs.pop("similarity_tail_threshold", 0.0)
        retrieval_top_k = kwargs.pop("retrieval_top_k", 40)
        reranker_top_k = kwargs.pop("reranker_top_k", 20)
        subgraph_depth = kwargs.pop("subgraph_depth", 2)
        node_vector_store_wrapper = VectorStoreWrapper(vector_store=self.node_vector_store)
        if self.concept_vector_store is not None:
            concept_vector_store_wrapper = VectorStoreWrapper(vector_store=self.concept_vector_store)
        else:
            concept_vector_store_wrapper = None

        rag_model = GraphRAGModel(
            llm=self.llm, llm_config=self.llm.llm_config,
            embed_func=self.embedding_model.embed_documents,
            graph_store=self.graph,
            vector_store=node_vector_store_wrapper,
            vector_store_concept=concept_vector_store_wrapper,
            reranker=self.rerank_model,
            use_text=use_text,
            batch_size=batch_size,
            similarity_tail_threshold=similarity_tail_threshold,
            retrieval_top_k=retrieval_top_k,
            reranker_top_k=reranker_top_k,
            subgraph_depth=subgraph_depth
        )
        return GraphRetriever(graph_rag_model=rag_model)

    def _init_vector_store(self, **kwargs):
        self.node_vector_store = kwargs.pop("node_vector_store", None)
        self.concept_vector_store = kwargs.pop("concept_vector_store", None)

        if self.node_vector_store is None:
            self.node_vector_store = VectorStorageFactory.create_storage(
                vector_type="npu_faiss_db",
                x_dim=self.dim,
                load_local_index=self.node_vectors_path,
                devs=self.devs
            )
        elif not isinstance(self.node_vector_store, VectorStore):
            raise GraphRAGError("node_vector_store must be an instance of VectorStore")
        if self.concept_vector_store is None:
            self.concept_vector_store = VectorStorageFactory.create_storage(
                vector_type="npu_faiss_db",
                x_dim=self.dim,
                load_local_index=self.concept_vectors_path,
                devs=self.devs
            )
        elif not isinstance(self.concept_vector_store, VectorStore):
            raise GraphRAGError("concept_vector_store must be an instance of VectorStore")

    def _setup_graph(self, graph_name, graph_type, **kwargs):
        if not isinstance(graph_type, str) or graph_type not in ["networkx", "opengauss"]:
            raise GraphRAGError("graph client supports only networkx or opengauss")
        if graph_type == "networkx":
            self.graph = NetworkxGraph(path=self.graph_save_path)
        elif self.graph_conf is None:
            if "graph_conf" not in kwargs:
                raise GraphRAGError("graph_conf must be specified in case of opengauss graph")
            self.graph_conf = kwargs.pop("graph_conf")
            if not isinstance(self.graph_conf, OpenGaussSettings):
                raise GraphRAGError("graph_conf must be an instance of OpenGaussSettings")
            self.graph = OpenGaussGraph(graph_name, self.graph_conf)

    def _setup_save_path(self, graph_name):
        self.graph_save_path = os.path.join(self.work_dir, f"{graph_name}.json")
        self.relations_save_path = os.path.join(self.work_dir, f"{graph_name}_relations.json")
        self.concepts_save_path = os.path.join(self.work_dir, f"{graph_name}_concepts.json")
        self.concept_cluster_path = os.path.join(self.work_dir, f"{graph_name}_concept_clusters.json")
        self.synset_save_path = os.path.join(self.work_dir, f"{graph_name}_synset.json")
        self.node_vectors_path = os.path.join(self.work_dir, f"{graph_name}_node_vectors.index")
        self.concept_vectors_path = os.path.join(self.work_dir, f"{graph_name}_concept_vectors.index")

    def _process_concepts_and_clusters(self, lang, top_k, threshold):
        if self.concept_embedding is None:
            self.concept_embedding = ConceptEmbedding(
                self.embedding_model.embed_documents
            )
        concepts = GraphConceptualizer(
            self.llm,
            self.graph,
            lang=lang,
            prompts=self.conceptualizer_prompts,
        ).conceptualize()
        save_to_json(concepts, self.concepts_save_path)
        logger.info(f"Concepts saved: {self.concepts_save_path}")

        embeddings = self.concept_embedding.embed(concepts)
        vector_store_wrapper = VectorStoreWrapper(self.concept_vector_store)
        graph = NetworkxGraph(is_digraph=False)
        cluster = ConceptCluster(vector_store=vector_store_wrapper, graph=graph)
        clusters = cluster.find_clusters(embeddings, top_k=top_k, threshold=threshold)
        save_to_json([list(c) for c in clusters], self.synset_save_path)
        logger.info(f"Clusters saved: {self.synset_save_path}")

        merger = ConceptGraphMerger(self.graph)
        merger.merge_concepts_and_synset(concepts, clusters)
        merger.save_graph(self.graph_save_path)
