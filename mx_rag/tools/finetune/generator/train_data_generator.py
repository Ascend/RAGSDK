# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from loguru import logger

from mx_rag.document.doc import Doc
from mx_rag.document.splitter import CharTextSplitter
from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess import generate_qa_embedding_pairs, improve_query, MineHardNegative
from mx_rag.tools.finetune.generator.common import BaseGenerator
from mx_rag.utils.file_check import FileCheck, SecFileCheck
from mx_rag.utils.file_operate import write_jsonl_to_file, read_jsonl_from_file

MAX_DATASET_LEN = 10000
MAX_FILE_SIZE_100M = 100 * 1024 * 1024
MAX_FILE_PROCESS_TIMES = 1000
SAMPLE_RANGE_MIN = 100
DEFAULT_TRUNC_SIZE = 512
DEFAULT_TRUNC_OVERLAP = 10


class TrainDataGenerator(BaseGenerator):
    def __init__(self, llm: Text2TextLLM, document_path: str, dataset_path: str, embedding: str, reranker: str):
        super().__init__(llm)
        self.document_path = document_path
        self.dataset_path = dataset_path
        self.embedding = embedding
        self.reranker = reranker

    def generate_train_data(self,
                            featured_percentage: float = 0.8,
                            llm_threshold_score: float = 0.8,
                            question_number: int = 10,
                            query_rewrite_numer: int = 2,
                            negative_number: int = 10):
        # 流程开始
        logger.info("step1 Split the document")
        split_doc_list = self.process_origin_document()
        logger.info("step1 Split the document finished")

        logger.info("step2 Generating rough problem documentation pairs")
        query_list, doc_list = self.generate_coarsest_qd_pairs(split_doc_list, question_number)
        logger.info("step2 Generated rough problem documentation pairs finished")

        logger.info("step3.1 bm25+reranker query document pair selection")
        featured_query_list, featured_doc_list = self._feature_qd_pair(query_list,
                                                                       doc_list,
                                                                       self.reranker,
                                                                       featured_percentage)
        logger.info("step3.1 bm25+reranker selection finished")

        logger.info("step3.2 LLM optimizing query document pair")
        preferred_query_list, preferred_doc_list = self._prefer_qd_pair(featured_query_list,
                                                                        featured_doc_list,
                                                                        llm_threshold_score)
        logger.info("step3.2 LLM optimizing query document pair finished")

        logger.info("step3.3 Enhancing query diversity and preserving training data")
        train_data = self.save_train_data_and_rewrite(preferred_query_list,
                                                      preferred_doc_list,
                                                      query_rewrite_numer)
        logger.info("step3.3 Enhancing query diversity and preserving training data finished")

        logger.info("step4 Hard Example Mining, enhance training data")
        train_data_mn_path = os.path.join(self.dataset_path, "train_data_mn.jsonl")
        sample_range = [1, min(SAMPLE_RANGE_MIN, len(preferred_query_list) * (query_rewrite_numer + 1))]
        mine_hard_negative = MineHardNegative(self.embedding)
        train_data = mine_hard_negative.find_knn_neg(train_data, sample_range, negative_number)
        write_jsonl_to_file(train_data, train_data_mn_path)
        logger.info("step4 Hard Example Mining Finished, the dataset has been prepared.")

        return train_data_mn_path

    def process_origin_document(self):
        """原始文档切分"""
        FileCheck.dir_check(self.document_path)
        FileCheck.dir_check(self.dataset_path)

        corpus_data_path = os.path.join(self.dataset_path, "corpus_data.jsonl")
        if not os.path.exists(corpus_data_path):
            return self._generate_origin_document(corpus_data_path)

        logger.info("The corpus data already exists and loaded from the file.")
        split_doc_list_from_file = read_jsonl_from_file(corpus_data_path)
        split_doc_list = []
        for doc in split_doc_list_from_file:
            split_doc_list.append(doc["content"])
        return split_doc_list

    def generate_coarsest_qd_pairs(self,
                                   split_doc_list: list[str],
                                   question_number: int) -> Tuple[List, List]:
        """文档对生成"""
        if len(split_doc_list) > MAX_DATASET_LEN:
            logger.error(f"inputs len should not bigger than {MAX_DATASET_LEN}, now is {len(split_doc_list)}")
            return [], []

        origin_train_data_path = os.path.join(self.dataset_path, "origin_train_data.jsonl")
        if not os.path.exists(origin_train_data_path):
            return self._generate_qd_pairs(split_doc_list, question_number, origin_train_data_path)

        logger.info("The qd file is existed, skip the generation process")
        qd_pairs = read_jsonl_from_file(origin_train_data_path)

        query_list = []
        doc_list = []
        for qd in qd_pairs:
            query_list.append(qd["query"])
            doc_list.extend(qd["pos"])

        return query_list, doc_list

    def save_train_data_and_rewrite(self,
                                    preferred_query_list: list[str],
                                    preferred_doc_list: list[str],
                                    query_rewrite_numer: int) -> List:
        if len(preferred_query_list) > MAX_DATASET_LEN or len(preferred_doc_list) > MAX_DATASET_LEN:
            logger.error(f"inputs len should not bigger than {MAX_DATASET_LEN}")
            return []

        if len(preferred_query_list) != len(preferred_doc_list):
            logger.error(f"preferred_query_list and preferred_doc_list has different len")
            return []

        FileCheck.dir_check(self.dataset_path)
        train_data_path = os.path.join(self.dataset_path, "train_data.jsonl")
        if os.path.exists(train_data_path):
            raise Exception(f"{train_data_path} exist, old data will be removed")

        qd_pairs = []
        for query, doc in zip(preferred_query_list, preferred_doc_list):
            qd_pairs.append({"query": query, "pos": [doc]})

        old_query_list = preferred_query_list[:]
        for i in range(query_rewrite_numer):
            logger.info(f"The {i + 1}st times rewrite the query")
            new_query_list = improve_query(self.llm, old_query_list)
            for query, doc in zip(new_query_list, preferred_doc_list):
                qd_pairs.append({"query": query, "pos": [doc]})

        write_jsonl_to_file(qd_pairs, train_data_path)

        return qd_pairs

    def _generate_origin_document(self, corpus_data_path: str):
        logger.info("Original document splitting")

        def md_load(file_path: str):
            SecFileCheck(file_path, MAX_FILE_SIZE_100M).check()

            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            docs = []
            for document in documents:
                docs.append(Doc(page_content=document.page_content, metadata=document.metadata))
            return docs

        doc_cnt = 0
        split_doc_list = []
        for mk in Path(self.document_path).rglob("*.md"):
            if doc_cnt > MAX_FILE_PROCESS_TIMES:
                logger.warning(f"unable to process files over {MAX_FILE_PROCESS_TIMES} times")
                break
            if not mk.is_file():
                continue
            for doc in md_load(mk.as_posix()):
                splitter = CharTextSplitter(chunk_size=DEFAULT_TRUNC_SIZE, chunk_overlap=DEFAULT_TRUNC_OVERLAP)
                split_doc_list.extend(splitter.split_text(doc.page_content))
            doc_cnt = doc_cnt + 1

        split_doc_list_to_save = []
        for doc in split_doc_list:
            split_doc_list_to_save.append({"content": doc})
        write_jsonl_to_file(split_doc_list_to_save, corpus_data_path)

        return split_doc_list

    def _generate_qd_pairs(self,
                           split_doc_list: list[str],
                           question_number: int,
                           origin_train_data_path: str,
                           chunk_size: int = 50):
        logger.info("query document pair generation")
        query_list = []
        doc_list = []
        count = 0
        for i in range(0, len(split_doc_list), chunk_size):
            chunk_doc_list = split_doc_list[i:i + chunk_size]  # 切片获取当前块的数据
            doc_queries = generate_qa_embedding_pairs(self.llm, chunk_doc_list, question_number)

            for doc, queries in doc_queries.items():
                query_list.extend(queries)
                doc_list.extend([doc] * len(queries))

            logger.info(f"The {count + 1}st query document pair generated success")
            count += 1

        qd_pairs = []
        for query, pos_doc in zip(query_list, doc_list):
            qd_pairs.append({"query": query, "pos": [pos_doc]})
        write_jsonl_to_file(qd_pairs, origin_train_data_path)

        return query_list, doc_list
