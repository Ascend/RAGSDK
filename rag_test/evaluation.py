# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import argparse
import datetime
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision, answer_correctness, answer_similarity
from ragas.metrics import critique, context_relevancy, answer_relevancy, context_entity_recall
from ragas.metrics import AnswerRelevancy, AnswerCorrectness, AnswerSimilarity
from datasets import load_dataset
from model.api_model import APILLM, APIEmbedding
from model.local_model import LocalLLM, LocalEmbedding

RAG_TEST_METRIC = {
    "faithfulness": faithfulness,
    "context_recall": context_recall,
    "context_precision": context_precision,
    "answer_correctness": answer_correctness,
    "answer_similarity": answer_similarity,
    "critique": critique,
    "context_relevancy": context_relevancy,
    "answer_relevancy": answer_relevancy,
    "context_entity_recall": context_entity_recall,
}


def dataset_preprocess(dataset_path: str):
    df = pd.read_csv(dataset_path)
    question_list = df.loc[:, "question"].tolist()
    ground_truth_list = df.loc[:, "ground_truth"].tolist()
    answer_list = df.loc[:, "answer"].tolist()
    contexts_lis = df.loc[:, "contexts"].tolist()

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

    datasets = Dataset.from_dict(datasets)
    return datasets


def result_postprocess(data, metrics: list, output_path: str):
    current_time = datetime.datetime.now(tz=timezone.utc)
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    filename = f'rag_test_{formatted_time}.csv'
    filepath = os.path.join(output_path, filename)
    df = data.to_pandas()
    row_number = df.shape[0]
    for element in metrics:
        value = 0
        for i in range(row_number):
            value += df.loc[i, element]
        ave = value / row_number
        df.loc[row_number+1, element] = ave
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load rag test parameter")
    parser.add_argument(
        "--language",
        type=str,
        default="chinese",
        help="the language of datasets, support chinese/english",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="url",
        help="the method of models, support url/local",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default="default",
        help="llm model url",
    )
    parser.add_argument(
        "--embed_url",
        type=str,
        default="default",
        help="embedding model url",
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default="llm model path",
        help="llm ",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default="default",
        help="embedding model path",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="faithfulness",
        help="rag test metric list",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="result",
        help="rag test result output path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/baseline.csv",
        help="rag test datasets path",
    )

    args = parser.parse_args()

    if args.language not in ["chinese", "english"]:
        raise ValueError("language only support chinese or english")

    if args.method not in ["url", "local"]:
        raise ValueError("method only support url or local")
    
    if not os.path.exists(args.llm_path):
        raise ValueError("llm path is invalid")
    
    if not os.path.exists(args.embed_path):
        raise ValueError("embed path is invalid")
    
    if not os.path.exists(args.output_path):
        raise ValueError("output path is invalid")
    
    if not os.path.exists(args.dataset_path):
        raise ValueError("dataset path is invalid")
    
    metric_list = args.metric.split(",")
    metric_support_list = list(RAG_TEST_METRIC.keys())
    for metric_name in metric_list:
        if metric_name not in metric_support_list:
            raise ValueError("metric is not support")

    evalsets = dataset_preprocess(args.dataset_path)

    if args.method == "url":
        embedding_model = APIEmbedding(args.llm_url)
        llm_model = APILLM(args.embed_url)
    elif args.method == "local":
        embedding_model = LocalEmbedding(args.llm_path)
        llm_model = LocalLLM(args.embed_path)

    metric_test_list = []
    for metric_name in metric_list:
        metric = RAG_TEST_METRIC[metric_name]
        metric.llm = llm_model
        if isinstance(metric, (AnswerRelevancy, AnswerCorrectness, AnswerSimilarity)):
            metric.embeddings = embedding_model
        if args.language == "chinese":
            metric.adapt(language=args.language, cache_dir="prompt")
        metric_test_list.append(metric)

    result = evaluate(evalsets, metrics=metric_test_list, is_async=False)
    result_postprocess(result, metric_list, args.output_path)