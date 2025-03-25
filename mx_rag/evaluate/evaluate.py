# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain.llms.base import LLM
from loguru import logger
from datasets import Dataset
from ragas.evaluation import Result
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    context_entity_recall,
    context_utilization,
    answer_correctness,
    answer_similarity,
    answer_relevancy
)
from ragas.metrics.critique import (
    harmfulness,
    maliciousness,
    coherence,
    correctness,
    conciseness
)

from mx_rag.utils.file_check import FileCheck, SecFileCheck
from mx_rag.utils.common import validate_params, validata_list_str, validata_list_list_str, TEXT_MAX_LEN, MB
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding


class Evaluate:
    RAG_TEST_METRIC = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "context_entity_recall": context_entity_recall,
        "context_utilization": context_utilization,
        "answer_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
        "harmfulness": harmfulness,
        "maliciousness": maliciousness,
        "coherence": coherence,
        "correctness": correctness,
        "conciseness": conciseness,
    }

    MAX_PROMPT_FILE_NUM:int = 30
    PROMPT_FILE_SUFFIX:str = ".json"
    PROMPT_FILE_MAX_SIZE:int = 1 * MB
    MAX_LIST_LEN:int = 128

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, LLM), message="param must be instance of LLM"),
        embedding=dict(validator=lambda x: isinstance(x, (TextEmbedding, TEIEmbedding)),
                       message="param must be instance of TextEmbedding or TEIEmbedding")
    )
    def __init__(self,
                 llm: LLM,
                 embedding: Embeddings):
        self.eval_embedding = embedding
        self.eval_llm = llm

    @classmethod
    def _check_metric_name(cls, metrics_name: list[str]):
        """
        metric name合法性校验
        Args:
            metrics_name: ragas metrics 列表

        Returns: None

        """
        for metric_name in metrics_name:
            if metric_name not in cls.RAG_TEST_METRIC:
                raise KeyError("find not support metric")

        if len(set(metrics_name)) != len(metrics_name):
            raise ValueError("duplicate metric")

    @classmethod
    def _check_datasets(cls, datasets: Dict[str, Any]) -> bool:
        check_attribute = {
            "question" : lambda x: validata_list_str(x, [1, cls.MAX_LIST_LEN], [1, TEXT_MAX_LEN]),
            "answer" : lambda x: validata_list_str(x, [1, cls.MAX_LIST_LEN], [1, TEXT_MAX_LEN]),
            "contexts":
                lambda x: validata_list_list_str(x, [1, cls.MAX_LIST_LEN], [1, cls.MAX_LIST_LEN], [1, TEXT_MAX_LEN]),
            "ground_truth": lambda x: validata_list_str(x, [1, cls.MAX_LIST_LEN], [1, TEXT_MAX_LEN])
        }

        if not (isinstance(datasets, Dict) and all(isinstance(key, str) for key in datasets)):
            logger.error("datasets should dict type and its key should str type")
            return False

        if not 1 <= len(datasets) <= len(check_attribute):
            logger.error(f"datasets key num range [1, {len(check_attribute)}]")
            return False

        return all(key in check_attribute and check_attribute.get(key)(value) for key, value in datasets.items())


    @validate_params(
        metrics_name=dict(validator=lambda x: validata_list_str(x, [1, len(Evaluate.RAG_TEST_METRIC)], [1, 50]),
                          message="param must meets: Type is List[str], list length range [1, 14], "
                                  "str length range [1, 50]"),
        datasets=dict(
            validator=lambda x: Evaluate._check_datasets(x), message="param check error detail see log"),
        language=dict(
            validator=lambda x: x is None or (isinstance(x, str) and 1 <= len(x) <= 64 and re.match(r'^[A-Za-z]+$', x)),
            message="param must be None or str, and str length range [1, 64] and must use alpha letter"),
        prompt_dir=dict(
            validator=lambda x: x is None or (isinstance(x, str) and 1 <= len(x) <= 256),
            message="param must be None or str, and str length range [1, 256]")
    )
    def evaluate(self,
                 metrics_name: list[str],
                 datasets: Dict[str, Any],
                 language: str = None,
                 prompt_dir: str = None) -> Optional[Result]:
        """
        根据metrics_name列表 计算得分
        Args:
            metrics_name: ragas metrics 列表
            datasets: 数据集 参考ragas官网
            language: 本地化语言
            prompt_dir: 本地化语言对应的prompt 路径

        Returns:ragas 评估结果 Result
        """
        self._check_metric_name(metrics_name)

        metrics = [self.RAG_TEST_METRIC.get(metric_name) for metric_name in metrics_name]

        self._metrics_local_adapt(metrics, language, prompt_dir)

        datesets = Dataset.from_dict(datasets)
        try:
            from ragas import evaluate as ragas_evaluate

            data = ragas_evaluate(dataset=datesets,
                                  metrics=metrics,
                                  llm=self.eval_llm,
                                  embeddings=self.eval_embedding)
        except ValueError:
            logger.error("ragas evaluate run failed value error")
            return None
        except Exception:
            logger.error("ragas evaluate run failed")
            return None

        return data

    def evaluate_scores(self,
                        metrics_name: list[str],
                        datasets: Dict[str, Any],
                        language: str = None,
                        prompt_dir: str = None) \
            -> Optional[Dict[str, List[float]]]:
        """
        根据metrics_name列表 计算得分
        Args:
            metrics_name: ragas metrics 列表
            datasets: 数据集 参考ragas官网
            language: 本地化语言
            prompt_dir: 本地化语言对应的prompt 路径

        Returns:Dict[str, List[float]]
        """
        data = self.evaluate(metrics_name, datasets, language, prompt_dir)
        if data is None:
            logger.error("evaluate fatal error")
            return None

        scores = data.scores.to_list()

        final_scores: Dict[str, List[float]] = {}
        for metric_name in metrics_name:
            final_scores[metric_name] = [score.get(metric_name, np.nan) for score in scores]
        return final_scores

    def _metrics_local_adapt(self, metrics, language: Optional[str], cache_dir: Optional[str]):
        """
        ragas metrics 本地化适配
        Args:
            metrics: 具体的ragas metrics
            language: 本地化语言
            cache_dir: 本地化语言对应的prompt 路径

        Returns: None
        """
        if language is None or cache_dir is None:
            logger.warning(f"because local param is None will not adapt local")
            return

        self._prompt_cache_check(cache_dir, language)

        logger.info(f"adapt to local!")

        try:
            for metric in metrics:
                self._metric_local_adapt(metric, language, cache_dir)
        except ValueError:
            logger.error("adapt to local run failed because value error")
        except Exception:
            logger.error("adapt to local fatal error")

    def _metric_local_adapt(self, metric, language: str, cache_dir :str):
        _exclude_adapt_metric: list[str] = [
            "context_entity_recall",
            "answer_similarity"
        ]

        from ragas.adaptation import adapt
        from ragas.llms.base import LangchainLLMWrapper

        if metric.name not in _exclude_adapt_metric:
            adapt(metrics=[metric], language=language, llm=self.eval_llm, cache_dir=cache_dir)
        else:
            if metric.name == "context_entity_recall":
                metric.llm = LangchainLLMWrapper(self.eval_llm)
                metric.context_entity_recall_prompt.adapt(
                    llm=metric.llm, language=language, cache_dir=cache_dir)
                metric.save(cache_dir=cache_dir)

    def _prompt_cache_check(self, cache_dir: str, language: str):
        prompt_dir = os.path.join(cache_dir, language)

        FileCheck.dir_check(prompt_dir)
        FileCheck.check_files_num_in_directory(prompt_dir, self.PROMPT_FILE_SUFFIX, self.MAX_PROMPT_FILE_NUM)

        files = os.listdir(prompt_dir)
        files = [os.path.join(prompt_dir, file) for file in files]
        filtered_files = [file for file in files if Path(file).suffix == self.PROMPT_FILE_SUFFIX]

        for file in filtered_files:
            SecFileCheck(file_path=file, max_size=self.PROMPT_FILE_MAX_SIZE).check()


