# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain.llms.base import LLM
from datasets import Dataset
from ragas import evaluate
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
    def load_data(file_path: str) -> Dict[str, Any]:
        """
        加载本地用户数据集 要求是csv 格式
        Args:
            file_path: 本地数据集路径

        Returns:解析之后的用户数据集 Dict[str, Any]

        """
        FileCheck.check_path_is_exist_and_valid(file_path)
        FileCheck.check_file_size(file_path, 100 * 1024 * 1024)

        df = pd.read_csv(file_path)
        question_list = df.loc[:, "question"].tolist()
        ground_truth_list = df.loc[:, "ground_truth"].tolist()
        answer_list = df.loc[:, "answer"].tolist()
        contexts_list = df.loc[:, "contexts"].tolist()

        for i, ground_truth in enumerate(ground_truth_list):
            ground_truth_list[i] = eval(ground_truth)

        for i, contexts in enumerate(contexts_list):
            contexts_list[i] = eval(contexts)

        datasets = {
            "question": question_list,
            "answer": answer_list,
            "contexts": contexts_list,
            "ground_truths": ground_truth_list
        }

        return datasets

    @classmethod
    def save_data(cls, data: Result, metrics_name: list[str], save_path: str):
        """
        将ragas 评估结果存放在save_path的目录下
        Args:
            data: ragas的评估结果
            metrics_name: 包括的测试方法
            save_path: 存放目录

        Returns: None

        """
        FileCheck.check_path_is_exist_and_valid(save_path)

        cls._check_metric_name(metrics_name)

        current_time = datetime.now(tz=timezone.utc)
        formatted_time = current_time.strftime('%Y%m%d%H%M%S')
        filename = f'rag_evaluate_{formatted_time}.csv'
        filepath = os.path.join(save_path, filename)
        df = data.to_pandas()
        row_number = df.shape[0]

        if row_number == 0:
            raise ValueError("row_number except greater than zero")

        for element in metrics_name:
            value = 0
            for i in range(row_number):
                value += df.loc[i, element]
            ave = value / row_number
            df.loc[row_number + 1, element] = ave
        df.to_csv(filepath, index=False)

    @classmethod
    def _check_metric_name(cls, metrics_name: list[str]):
        """
        metric name合法性校验
        Args:
            metrics_name: ragas metrics 列表

        Returns: None

        """
        for metric_name in metrics_name:
            if not isinstance(metric_name, str):
                raise TypeError("metrics_name element type must be string")

            if metric_name not in cls.RAG_TEST_METRIC:
                raise KeyError(f"{metric_name} not support in Evaluate")

    def evaluate(self,
                 metrics_name: list[str],
                 datasets: Dict[str, Any],
                 language: str = None,
                 prompt_dir: str = None,
                 **kwargs) -> Result:
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
        if not isinstance(metrics_name, list):
            raise TypeError("metrics_name must be list")

        self._check_metric_name(metrics_name)

        metrics = [self.RAG_TEST_METRIC.get(metric_name) for metric_name in metrics_name]

        self._metrics_local_adapt(metrics, language, prompt_dir)

        datesets = Dataset.from_dict(datasets)
        data = evaluate(dataset=datesets,
                        metrics=metrics,
                        llm=self.eval_llm,
                        embeddings=self.eval_embedding,
                        **kwargs)
        return data

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
