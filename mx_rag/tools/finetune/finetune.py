# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import argparse
import os

from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.generator import TrainDataGenerator, EvalDataGenerator
from mx_rag.tools.finetune.train import train_embedding, train_reranker
from mx_rag.utils.file_check import FileCheck

DEFAULT_LLM_TIMEOUT = 10 * 60


class Finetune:
    def __init__(self,
                 origin_document_path: str,
                 generate_dataset_path: str,
                 llm: Text2TextLLM,
                 embed_model_path: str,
                 reranker_model_path: str,
                 finetune_output_path: str,
                 featured_percentage: float = 0.8,
                 llm_threshold_score: float = 0.8,
                 train_question_number: int = 2,
                 query_rewrite_number: int = 1,
                 negative_number: int = 2,
                 eval_samples: int = 500,
                 eval_question_number: int = 5):
        self.origin_document_path = origin_document_path
        self.generate_dataset_path = generate_dataset_path
        self.llm = llm
        self.embed_model_path = embed_model_path
        self.reranker_model_path = reranker_model_path
        self.finetune_output_path = finetune_output_path

        self.featured_percentage = featured_percentage
        self.llm_threshold_score = llm_threshold_score
        self.train_question_number = train_question_number
        self.query_rewrite_number = query_rewrite_number
        self.negative_number = negative_number

        self.eval_samples = eval_samples
        self.eval_question_number = eval_question_number

    def start(self):
        logger.info("--------------------Generating training data--------------------")
        train_data_generator = TrainDataGenerator(self.llm,
                                                  self.origin_document_path,
                                                  self.generate_dataset_path,
                                                  self.embed_model_path,
                                                  self.reranker_model_path)
        train_data_path = train_data_generator.generate_train_data(self.featured_percentage,
                                                                   self.llm_threshold_score,
                                                                   self.train_question_number,
                                                                   self.query_rewrite_number,
                                                                   self.negative_number)

        # 检查微调文件输出目录
        FileCheck.dir_check(self.finetune_output_path)

        logger.info("--------------------Fine-tuning embedding--------------------")
        output_embed_model_path = os.path.join(self.finetune_output_path, 'embedding-finetune')
        if not os.path.exists(output_embed_model_path):
            os.mkdir(output_embed_model_path)
            FileCheck.dir_check(output_embed_model_path)
        train_embedding(self.embed_model_path, output_embed_model_path, train_data_path)

        logger.info("--------------------Fine-tuning reranker--------------------")
        output_reranker_model_path = os.path.join(self.finetune_output_path, 'reranker-finetune')
        if not os.path.exists(output_reranker_model_path):
            os.mkdir(output_reranker_model_path)
            FileCheck.dir_check(output_reranker_model_path)
        train_reranker(self.reranker_model_path, output_reranker_model_path, train_data_path)

        logger.info("--------------------Generate evaluation data--------------------")
        eval_data_generator = EvalDataGenerator(self.llm, self.generate_dataset_path, self.reranker_model_path)
        eval_data_generator.generate_eval_data(self.eval_samples,
                                               self.eval_question_number,
                                               self.featured_percentage,
                                               self.llm_threshold_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--document_path", type=str, default="")
    parser.add_argument("--generate_dataset_path", type=str, default="")
    parser.add_argument("--llm_url", type=str, default="")
    parser.add_argument("--llm_model_name", type=str, default="")
    parser.add_argument("--use_http", type=bool, default=False)
    parser.add_argument("--embedding_model_path", type=str, default="")
    parser.add_argument("--reranker_model_path", type=str, default="")
    parser.add_argument("--finetune_output_path", type=str, default="")

    parser.add_argument("--featured_percentage", type=float, default=0.8)
    parser.add_argument("--llm_threshold_score", type=float, default=0.8)
    parser.add_argument("--train_question_number", type=int, default=10)
    parser.add_argument("--query_rewrite_number", type=int, default=10)
    parser.add_argument("--negative_number", type=int, default=5)

    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--eval_question_number", type=int, default=5)

    args = parser.parse_args()
    logger.info("Fine-tuning beginning")
    text_llm = Text2TextLLM(base_url=args.llm_url, model_name=args.llm_model_name, timeout=DEFAULT_LLM_TIMEOUT,
                            use_http=args.use_http)
    finetune = Finetune(args.document_path,
                        args.generate_dataset_path,
                        text_llm,
                        args.embedding_model_path,
                        args.reranker_model_path,
                        args.finetune_output_path,
                        args.featured_percentage,
                        args.llm_threshold_score,
                        args.train_question_number,
                        args.query_rewrite_number,
                        args.negative_number,
                        args.eval_samples,
                        args.eval_question_number)
    finetune.start()
