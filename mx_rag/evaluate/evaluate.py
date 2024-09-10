# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import ast
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain.llms.base import LLM
from loguru import logger
from datasets import Dataset
from ragas.evaluation import Result
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
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

from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.common import validate_params


class Evaluate:
    RAG_TEST_METRIC = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_relevancy": context_relevancy,
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

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, LLM)),
        embedding=dict(validator=lambda x: isinstance(x, Embeddings))
    )
    def __init__(self,
                 llm: LLM,
                 embedding: Embeddings):
        self.eval_embedding = embedding
        self.eval_llm = llm

    @staticmethod
    def load_data(file_path: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        加载本地用户数据集 要求是csv 格式
        Args:
            file_path: 本地数据集路径

        Returns:解析之后的用户数据集 Dict[str, Any]

        """
        FileCheck.check_path_is_exist_and_valid(file_path)
        FileCheck.check_file_size(file_path, 100 * 1024 * 1024)

        try:
            data = pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"load_data error {e}")
            return None

        context_key_words = 'contexts'
        if context_key_words in data:
            data[context_key_words] = data[context_key_words].apply(ast.literal_eval)
        datasets = data.to_dict(orient='list')
        return datasets

    @classmethod
    def save_data(cls, data: Result, save_path: str, **kwargs):
        """
        将ragas 评估结果存放在save_path的目录下
        Args:
            data: ragas的评估结果
            save_path: 存放目录

        Returns: None

        """
        FileCheck.check_path_is_exist_and_valid(save_path)

        current_time = datetime.now(tz=timezone.utc)
        formatted_time = current_time.strftime('%Y%m%d%H%M%S')
        filename = f'rag_evaluate_{formatted_time}.csv'
        filepath = os.path.join(save_path, filename)

        data.to_pandas().to_csv(filepath, **kwargs)
        logger.info(f"evaluate save data to {filepath}")

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
                raise KeyError(f"{metric_name} not support in Evaluate")

        if len(set(metrics_name)) != len(metrics_name):
            raise ValueError(f"duplicate metric {metrics_name}")

    @validate_params(
        metrics_name=dict(
            validator=lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x) and 0 < len(x) <= 14),
        datasets=dict(
            validator=lambda x: isinstance(x, Dict) and all(isinstance(key, str) for key in x) and 0 < len(x) <= 4096),
        language=dict(validator=lambda x: x is None or (isinstance(x, str) and 0 < len(x) <= 64))
    )
    def evaluate(self,
                 metrics_name: list[str],
                 datasets: Dict[str, Any],
                 language: str = None,
                 prompt_dir: str = None,
                 **kwargs) -> Optional[Result]:
        """
        根据metrics_name列表 计算得分
        Args:
            metrics_name: ragas metrics 列表
            datasets: 数据集 参考ragas官网
            language: 本地化语言
            prompt_dir: 本地化语言对应的prompt 路径
            **kwargs: ragas 运行时参数

        Returns:ragas 评估结果 Result
        """
        from ragas import evaluate as ragas_evaluate

        self._check_metric_name(metrics_name)

        metrics = [self.RAG_TEST_METRIC.get(metric_name) for metric_name in metrics_name]

        self._metrics_local_adapt(metrics, language, prompt_dir)

        datesets = Dataset.from_dict(datasets)
        try:
            data = ragas_evaluate(dataset=datesets,
                                  metrics=metrics,
                                  llm=self.eval_llm,
                                  embeddings=self.eval_embedding,
                                  **kwargs)
        except ValueError as e:
            logger.error(f"evaluate unexpect: {e}")
            return None

        return data

    @validate_params(
        metrics_name=dict(
            validator=lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x) and 0 < len(x) <= 14),
        datasets=dict(
            validator=lambda x: isinstance(x, Dict) and all(isinstance(key, str) for key in x) and 0 < len(x) <= 4096),
        language=dict(validator=lambda x: x is None or (isinstance(x, str) and 0 < len(x) <= 64))
    )
    def evaluate_scores(self,
                        metrics_name: list[str],
                        datasets: Dict[str, Any],
                        language: str = None,
                        prompt_dir: str = None,
                        **kwargs) \
            -> Dict[str, List[float]]:
        """
        根据metrics_name列表 计算得分
        Args:
            metrics_name: ragas metrics 列表
            datasets: 数据集 参考ragas官网
            language: 本地化语言
            prompt_dir: 本地化语言对应的prompt 路径
            **kwargs: ragas 运行时参数

        Returns:Dict[str, List[float]]
        """
        data = self.evaluate(metrics_name, datasets, language, prompt_dir, **kwargs)
        scores = data.scores.to_list()

        final_scores: Dict[str, List[float]] = {}
        for metric_name in metrics_name:
            final_scores[metric_name] = [score[metric_name] for score in scores]
        return final_scores

    def _metrics_local_adapt(self, metrics, language: str, cache_dir: str):
        """
        ragas metrics 本地化适配
        Args:
            metrics: 具体的ragas metrics
            language: 本地化语言
            cache_dir: 本地化语言对应的prompt 路径

        Returns: None
        """
        from ragas.adaptation import adapt
        from ragas.llms.base import LangchainLLMWrapper

        if language is None or cache_dir is None:
            logger.warning(f"because local param is None will not adapt local")
            return

        logger.info(f"local param language:{language} cache_dir:{cache_dir}")

        _exclude_adapt_metric: list[str] = [
            "context_entity_recall",
            "answer_similarity"
        ]

        FileCheck.check_path_is_exist_and_valid(cache_dir)

        for metric in metrics:
            if metric.name not in _exclude_adapt_metric:
                adapt(metrics=[metric], language=language, llm=self.eval_llm, cache_dir=cache_dir)
            else:
                if metric.name == "context_entity_recall":
                    metric.llm = LangchainLLMWrapper(self.eval_llm)
                    metric.context_entity_recall_prompt.adapt(llm=metric.llm, language=language, cache_dir=cache_dir)
                    metric.save(cache_dir=cache_dir)
