import os
import unittest
from unittest.mock import MagicMock, patch

from datasets import Dataset
from ragas.evaluation import Result
from ragas.validation import handle_deprecated_ground_truths, remap_column_names

from mx_rag.embedding.local import TextEmbedding
from mx_rag.evaluate import Evaluate
from mx_rag.llm import Text2TextLLM

current_dir = os.path.dirname(os.path.realpath(__file__))
PROMPT_DIR = os.path.realpath(os.path.join(current_dir, "../../../mx_rag/evaluate/prompt"))


class TestEvaluate(unittest.TestCase):
    @patch("ragas.evaluate")
    def test_evaluate(self, evaluate_mock):
        ori_dataset = {
            "question": ["世界上最高的山峰是哪座？"],
            "answer": ["珠穆朗玛峰"],
            "contexts": [["世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8848米。"]]
        }
        datasets = Dataset.from_dict(ori_dataset)
        dataset = remap_column_names(datasets, {})
        dataset = handle_deprecated_ground_truths(dataset)
        llm = MagicMock(spec=Text2TextLLM)
        embedding = MagicMock(spec=TextEmbedding)
        evaluator = Evaluate(llm=llm, embedding=embedding)
        scores = [{"answer_relevancy": 0.09, "context_utilization": 0.01, "faithfulness": 0.5}]
        result = Result(
            scores=Dataset.from_list(scores),
            dataset=dataset,
            binary_columns=[],
        )

        evaluate_mock.return_value = result
        scores = evaluator.evaluate_scores(metrics_name=["answer_relevancy", "context_utilization", "faithfulness"],
                                           datasets=ori_dataset,
                                           language="chinese",
                                           prompt_dir=PROMPT_DIR)
        self.assertEqual(scores, {'answer_relevancy': [0.09], 'context_utilization': [0.01], 'faithfulness': [0.5]})
