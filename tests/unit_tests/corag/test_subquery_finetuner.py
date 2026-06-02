#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import torch
import unittest
from unittest.mock import patch, MagicMock
from transformers import TrainingArguments

from mx_rag.corag.subquery_finetuner import (
    _ChainOfRagCollator,
    FineTuneArguments,
    SubqueryFineTuner,
)


class TestChainOfRagCollator(unittest.TestCase):
    """测试 _ChainOfRagCollator 数据整理器"""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode = MagicMock(side_effect=lambda x, add_special_tokens=False: [ord(c) for c in x])
        self.mock_tokenizer.pad = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        })
        self.mock_tokenizer.pad_token = "<pad>"
        self.max_len = 2048

    def test_init_success(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        self.assertEqual(collator.max_len, self.max_len)
        self.assertEqual(collator.tokenizer, self.mock_tokenizer)

    def test_init_default_max_len(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer)
        self.assertEqual(collator.max_len, 2048)

    def test_call_single_message(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        features = [{
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"}
            ]
        }]
        result = collator(features)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)

    def test_call_with_observation_role(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        features = [{
            "messages": [
                {"role": "observation", "content": "retrieved content"}
            ]
        }]
        result = collator(features)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)

    def test_call_with_system_role(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        features = [{
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user query"},
                {"role": "assistant", "content": "assistant response"}
            ]
        }]
        result = collator(features)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)

    def test_call_multiple_features(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        features = [
            {"messages": [{"role": "user", "content": "query1"}, {"role": "assistant", "content": "answer1"}]},
            {"messages": [{"role": "user", "content": "query2"}, {"role": "assistant", "content": "answer2"}]}
        ]
        result = collator(features)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)

    def test_call_truncation(self):
        small_max_len = 10
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(side_effect=lambda x, add_special_tokens=False: [1] * len(x))
        mock_tokenizer.pad = MagicMock(return_value={
            "input_ids": torch.tensor([[1] * small_max_len]),
            "attention_mask": torch.tensor([[1] * small_max_len])
        })
        collator = _ChainOfRagCollator(tokenizer=mock_tokenizer, max_len=small_max_len)
        features = [{
            "messages": [
                {"role": "user", "content": "a" * 100},
                {"role": "assistant", "content": "b" * 100}
            ]
        }]
        result = collator(features)
        self.assertIn("input_ids", result)
        self.assertEqual(result["input_ids"].shape[1], small_max_len)

    def test_call_labels_mask_for_non_assistant(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        features = [{
            "messages": [
                {"role": "user", "content": "query"},
                {"role": "assistant", "content": "response"}
            ]
        }]
        result = collator(features)
        self.assertIn("labels", result)
        labels = result["labels"].tolist()[0]
        has_non_masked = any(l != -100 for l in labels)
        self.assertTrue(has_non_masked)

    def test_call_empty_messages(self):
        collator = _ChainOfRagCollator(tokenizer=self.mock_tokenizer, max_len=self.max_len)
        features = [{"messages": []}]
        result = collator(features)
        self.assertIn("input_ids", result)


class TestFineTuneArguments(unittest.TestCase):
    """测试 FineTuneArguments 参数类"""

    def test_default_values(self):
        args = FineTuneArguments()
        self.assertEqual(args.model_name_or_path, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(args.train_file, "data/aligned_train.jsonl")
        self.assertEqual(args.max_len, 2048)

    def test_custom_values(self):
        args = FineTuneArguments(
            model_name_or_path="/path/to/model",
            train_file="/path/to/train.jsonl",
            max_len=4096
        )
        self.assertEqual(args.model_name_or_path, "/path/to/model")
        self.assertEqual(args.train_file, "/path/to/train.jsonl")
        self.assertEqual(args.max_len, 4096)

    def test_none_train_file(self):
        args = FineTuneArguments(train_file=None)
        self.assertIsNone(args.train_file)

    def test_dataclass_immutability(self):
        args = FineTuneArguments()
        args.model_name_or_path = "new_model"
        self.assertEqual(args.model_name_or_path, "new_model")


class TestSubqueryFineTuner(unittest.TestCase):
    """测试 SubqueryFineTuner 微调器"""

    def setUp(self):
        self.finetune_args = FineTuneArguments(
            model_name_or_path="test_model",
            train_file="test_train.jsonl",
            max_len=512
        )
        self.train_args = TrainingArguments(
            output_dir="./test_output",
            do_train=False,
        )

    def test_init_success(self):
        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        self.assertEqual(tuner.finetune_args, self.finetune_args)
        self.assertEqual(tuner.train_args, self.train_args)
        self.assertIsNone(tuner.model)
        self.assertIsNone(tuner.tokenizer)
        self.assertIsNone(tuner.collator)
        self.assertIsNone(tuner.trainer)

    @patch("mx_rag.corag.subquery_finetuner.AutoModelForCausalLM.from_pretrained")
    @patch("mx_rag.corag.subquery_finetuner.AutoTokenizer.from_pretrained")
    def test_prepare_model_and_tokenizer_success(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        tuner._prepare_model_and_tokenizer()

        mock_model_from_pretrained.assert_called_once_with(
            self.finetune_args.model_name_or_path,
            local_files_only=True
        )
        mock_tokenizer_from_pretrained.assert_called_once_with(
            self.finetune_args.model_name_or_path,
            pad_token='</s>',
            local_files_only=True
        )
        self.assertEqual(tuner.model, mock_model)
        self.assertEqual(tuner.tokenizer, mock_tokenizer)

    @patch("mx_rag.corag.subquery_finetuner.AutoModelForCausalLM.from_pretrained")
    @patch("mx_rag.corag.subquery_finetuner.AutoTokenizer.from_pretrained")
    def test_prepare_model_and_tokenizer_set_pad_token(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        tuner._prepare_model_and_tokenizer()

        self.assertEqual(tuner.tokenizer.pad_token, "<eos>")

    @patch("mx_rag.corag.subquery_finetuner.AutoModelForCausalLM.from_pretrained")
    def test_prepare_model_and_tokenizer_failure(self, mock_model_from_pretrained):
        mock_model_from_pretrained.side_effect = FileNotFoundError("Model not found")

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        with self.assertRaises(FileNotFoundError):
            tuner._prepare_model_and_tokenizer()

    @patch("mx_rag.corag.subquery_finetuner.load_dataset")
    def test_prepare_datasets_success(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        result = tuner._prepare_datasets()

        mock_load_dataset.assert_called_once()
        self.assertEqual(result, mock_dataset)

    @patch("mx_rag.corag.subquery_finetuner.load_dataset")
    def test_prepare_datasets_with_none_train_file(self, mock_load_dataset):
        finetune_args = FineTuneArguments(train_file=None)
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        tuner = SubqueryFineTuner(
            finetune_args=finetune_args,
            train_args=self.train_args
        )
        result = tuner._prepare_datasets()

        mock_load_dataset.assert_called_once_with("json", data_files={}, split="train")

    @patch("mx_rag.corag.subquery_finetuner.load_dataset")
    def test_prepare_datasets_failure(self, mock_load_dataset):
        mock_load_dataset.side_effect = FileNotFoundError("Dataset not found")

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        with self.assertRaises(FileNotFoundError):
            tuner._prepare_datasets()

    @patch("mx_rag.corag.subquery_finetuner.Trainer")
    @patch("mx_rag.corag.subquery_finetuner._ChainOfRagCollator")
    def test_initialize_trainer_success(self, mock_collator_class, mock_trainer_class):
        mock_collator = MagicMock()
        mock_collator_class.return_value = mock_collator
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_dataset = MagicMock()

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        tuner.model = mock_model
        tuner.tokenizer = mock_tokenizer
        tuner._initialize_trainer(mock_dataset)

        mock_collator_class.assert_called_once_with(
            tokenizer=mock_tokenizer,
            max_len=self.finetune_args.max_len
        )
        mock_trainer_class.assert_called_once_with(
            model=mock_model,
            args=self.train_args,
            train_dataset=mock_dataset,
            data_collator=mock_collator
        )
        self.assertEqual(tuner.trainer, mock_trainer)

    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._prepare_model_and_tokenizer")
    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._prepare_datasets")
    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._initialize_trainer")
    def test_train_without_do_train(self, mock_init_trainer, mock_prepare_datasets, mock_prepare_model):
        train_args = TrainingArguments(output_dir="./test_output", do_train=False)
        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=train_args
        )
        tuner.train()

        mock_prepare_model.assert_called_once()
        mock_prepare_datasets.assert_called_once()
        mock_init_trainer.assert_called_once()

    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._prepare_model_and_tokenizer")
    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._prepare_datasets")
    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._initialize_trainer")
    def test_train_with_do_train(self, mock_init_trainer, mock_prepare_datasets, mock_prepare_model):
        train_args = TrainingArguments(output_dir="./test_output", do_train=True)
        mock_trainer = MagicMock()
        mock_tokenizer = MagicMock()

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=train_args
        )
        tuner.trainer = mock_trainer
        tuner.tokenizer = mock_tokenizer
        tuner.train()

        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once_with(output_dir=train_args.output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(train_args.output_dir)

    @patch("mx_rag.corag.subquery_finetuner.SubqueryFineTuner._prepare_model_and_tokenizer")
    def test_train_failure(self, mock_prepare_model):
        mock_prepare_model.side_effect = RuntimeError("Training failed")

        tuner = SubqueryFineTuner(
            finetune_args=self.finetune_args,
            train_args=self.train_args
        )
        with self.assertRaises(RuntimeError):
            tuner.train()


class TestChainOfRagCollatorIntegration(unittest.TestCase):
    """测试 _ChainOfRagCollator 集成场景"""

    def test_call_with_real_tokenizer_format(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(side_effect=lambda x, add_special_tokens=False: list(range(len(x))))
        mock_tokenizer.pad = MagicMock(return_value={
            "input_ids": torch.arange(30).unsqueeze(0),
            "attention_mask": torch.tensor([[1] * 30])
        })
        mock_tokenizer.pad_token = "<pad>"

        collator = _ChainOfRagCollator(tokenizer=mock_tokenizer, max_len=2048)

        features = [{
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is RAG?"},
                {"role": "observation", "content": "RAG stands for Retrieval-Augmented Generation."},
                {"role": "assistant", "content": "RAG is a technique that combines retrieval and generation."}
            ]
        }]

        result = collator(features)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)
        self.assertEqual(result["labels"].dtype, torch.long)


class TestFineTuneArgumentsEdgeCases(unittest.TestCase):
    """测试 FineTuneArguments 边界条件"""

    def test_max_len_boundary(self):
        args = FineTuneArguments(max_len=1)
        self.assertEqual(args.max_len, 1)

        args = FineTuneArguments(max_len=8192)
        self.assertEqual(args.max_len, 8192)

    def test_model_path_with_special_chars(self):
        args = FineTuneArguments(model_name_or_path="/path/with spaces/model")
        self.assertEqual(args.model_name_or_path, "/path/with spaces/model")

    def test_train_file_with_unicode(self):
        args = FineTuneArguments(train_file="/路径/训练数据.jsonl")
        self.assertEqual(args.train_file, "/路径/训练数据.jsonl")


if __name__ == '__main__':
    unittest.main()
