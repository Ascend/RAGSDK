# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import random

from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess.generate_qd import generate_qa_embedding_pairs
from mx_rag.tools.finetune.generator.common import BaseGenerator
from mx_rag.utils.file_check import FileCheck, SecFileCheck
from mx_rag.utils.file_operate import write_jsonl_to_file, read_jsonl_from_file
from mx_rag.utils.common import validate_params, INT_32_MAX

MAX_FILE_SIZE_100M = 100 * 1024 * 1024


class EvalDataGenerator(BaseGenerator):
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM)),
        dataset_path=dict(validator=lambda x: isinstance(x, str)),
        reranker=dict(validator=lambda x: isinstance(x, str))
    )
    def __init__(self, llm: Text2TextLLM, dataset_path: str, reranker: str):
        super().__init__(llm)
        self.dataset_path = dataset_path
        self.reranker = reranker

    @validate_params(
        max_samples=dict(validator=lambda x: 0 < x < INT_32_MAX),
        question_number=dict(validator=lambda x: 0 < x < INT_32_MAX),
        featured_percentage=dict(validator=lambda x: 0 <= x <= 1),
        llm_threshold_score=dict(validator=lambda x: 0 <= x <= 1)
    )
    def generate_eval_data(self,
                           max_samples: int = 500,
                           question_number: int = 2,
                           featured_percentage: float = 0.8,
                           llm_threshold_score: float = 0.8):
        logger.info("Start to generate evaluation data")
        FileCheck.dir_check(self.dataset_path)
        corpus_data_path = os.path.join(self.dataset_path, "corpus_data.jsonl")
        SecFileCheck(corpus_data_path, MAX_FILE_SIZE_100M).check()

        eval_data_path = os.path.join(self.dataset_path, "eval_data.jsonl")
        if os.path.exists(eval_data_path):
            raise Exception(f"'{eval_data_path}' exist, old data will be removed")

        corpus_list_from_file = read_jsonl_from_file(corpus_data_path)
        corpus_list = []
        for doc in corpus_list_from_file:
            corpus_list.append(doc["content"])

        sample_doc_list = random.sample(corpus_list, min(len(corpus_list), max_samples))
        doc_queries = generate_qa_embedding_pairs(self.llm, sample_doc_list, question_number)
        query_list = []
        doc_list = []
        for doc, queries in doc_queries.items():
            query_list.extend(queries)
            doc_list.extend([doc] * len(queries))

        logger.info("step1 Query document pair selecting")
        featured_query_list, featured_doc_list = \
            self._feature_qd_pair(query_list, doc_list, self.reranker, featured_percentage)
        logger.info("step1 Query document pair selecting finished")

        logger.info("step2 LLM optimizing query document pair")
        preferred_query_list, preferred_doc_list = \
            self._prefer_qd_pair(featured_query_list, featured_doc_list, llm_threshold_score)
        logger.info("step2 LLM optimizing query document pair finished")

        fd_pairs = []
        for query, pos_doc in zip(preferred_query_list, preferred_doc_list):
            fd_pairs.append({"query": query, "pos": [pos_doc]})
        write_jsonl_to_file(fd_pairs, eval_data_path)
        logger.info("The evaluation data is generated successfully")
