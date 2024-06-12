from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision, answer_correctness, answer_similarity
from ragas.metrics import critique, context_relevancy, answer_relevancy, context_entity_recall
from datasets import load_dataset
from .model.api_model import APILLM, APIEmbedding
from .model.local_model import LocalLLM, LocalEmbedding
import pandas as pd
import datetime
import argparse

RAG_TEST_METRIC = [faithfulness, context_recall, context_precision, answer_correctness, answer_similarity,
                   critique, context_relevancy, answer_relevancy, context_entity_recall]

def result_postprocess(result):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    filename = f'result/rag_test_{formatted_time}.csv'
    df = result.to_pandas()
    row_number=df.shape[0]
    for metric in ["context_recall", "context_precision", "answer_relevancy", "faithfulness", "answer_correctness",
                   "answer_similarity", "critique", "context_relevancy", "context_entity_recall"]:
        value=0
        for i in range(row_number):
            value+=df.loc[i, metric]
        ave=value/row_number
        df.loc[row_number+1, metric]=ave
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load rag test parameter")
    parser.add_argument(
        "--language",
        type=str,
        default="chinese",
        help="Language type of the evaluation dataset",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="url",
        help="The method of using llm and embeddings",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="default",
        help="model path or api url",
    )
    parser.add_argument(
        "--metric",
        type=list,
        default="context_recall, context_precision, answer_relevancy",
        help="max length of input sequence",
    )

    args = parser.parse_args()

    dataset = load_dataset('csv', data_files='datatest/base_dataset.csv')

    if args.method=="url":
        embedding_model = APIEmbedding(args.path)
        llm_model = APILLM(args.path)
    elif args.method=="local":
        embedding_model = LocalEmbedding(args.path)
        llm_model = LocalLLM(args.path)

    for metric in RAG_TEST_METRIC:
        metric.adapt(language=args.language, cache_dir="prompt/")
        metric.llm = llm_model
        if isinstance(metric, (AnswerRelevancy, AnswerCorrectness, AnswerSimilarity)):
            metric.embeddings = embedding_model

    result = evaluate(dataset, metrics=RAG_TEST_METRIC)
    result_postprocess(result)