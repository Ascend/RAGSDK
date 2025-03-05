# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os

from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess.generate_qd import GENERATE_QA_PROMPT
from mx_rag.tools.finetune.generator.common import BaseGenerator
from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.file_operate import write_jsonl_to_file
from mx_rag.utils.common import (validate_params, validata_list_str, TEXT_MAX_LEN, STR_MAX_LEN,
                                 MAX_PATH_LENGTH, MAX_PROMPT_LENGTH)


class EvalDataGenerator(BaseGenerator):
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        dataset_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PATH_LENGTH,
                          message=f"param must be str and str length range (0, {MAX_PATH_LENGTH}]")
    )
    def __init__(self, llm: Text2TextLLM, dataset_path: str):
        super().__init__(llm, dataset_path)

    @validate_params(
        split_doc_list=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                            message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                                    f"str length range [1, {STR_MAX_LEN}]"),
        generate_qd_prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                                message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]"),
        question_number=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 20,
                             message="param must meets: Type is int, length range (0, 20]"),

    )
    def generate_evaluate_data(self,
                               split_doc_list: list[str],
                               generate_qd_prompt: str = GENERATE_QA_PROMPT,
                               question_number: int = 3,
                               batch_size: int = 8):
        FileCheck.dir_check(self.dataset_path)
        evaluate_data_path = os.path.join(self.dataset_path, "evaluate_data.jsonl")
        if os.path.exists(evaluate_data_path):
            logger.info("embedding evaluate data has been created.")
            return

        # 流程开始
        logger.info("step Generating rough problem documentation pairs")
        query_list, doc_list = self._generate_coarsest_qd_pairs(split_doc_list, question_number,
                                                                generate_qd_prompt, batch_size)
        logger.info("step Generated rough problem documentation pairs finished")

        evaluate_data = []
        for query, doc in zip(query_list, doc_list):
            evaluate_data.append({"query": query, "corpus": doc})

        write_jsonl_to_file(evaluate_data, evaluate_data_path)

        return
