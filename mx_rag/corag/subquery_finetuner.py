#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from datasets import Dataset, load_dataset
from loguru import logger


class _ChainOfRagCollator:
    """
    数据整理器类，用于处理Chain-of-RAG的训练数据
    
    私有类，仅用于内部数据处理
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_len: int = 2048,
    ):
        """
        初始化ChainOfRagCollator
        
        Args:
            tokenizer: 分词器
            max_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        logger.info(f"Initialized ChainOfRagCollator with max_len: {max_len}")
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        处理输入特征，返回整理后的数据
        
        Args:
            features: 输入特征列表
            
        Returns:
            整理后的数据字典
        """
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for feature in features:
            messages = feature["messages"]

            input_ids = []
            labels = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                role_str = "system" if role == "observation" else role
                header = f"<|im_start|>{role_str}\n"
                header_ids = self.tokenizer.encode(header, add_special_tokens=False)
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                footer = "<|im_end|>\n"
                footer_ids = self.tokenizer.encode(footer, add_special_tokens=False)

                part_ids = header_ids + content_ids + footer_ids
                input_ids.extend(part_ids)
                
                if role == "assistant":
                    part_labels = [-100] * len(header_ids) + content_ids + footer_ids
                else:
                    part_labels = [-100] * len(part_ids)
                labels.extend(part_labels)

            if len(input_ids) > self.max_len:
                logger.debug(f"Truncating input from {len(input_ids)} to {self.max_len} tokens")
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]

            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        max_batch_len = padded["input_ids"].shape[1]
        padded_labels = []
        for label in labels_list:
            padded_label = label + [-100] * (max_batch_len - len(label))
            padded_labels.append(padded_label)
        padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return padded


@dataclass
class FineTuneArguments:
    """
    模型微调参数类，用于配置模型微调相关参数
    """
    
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_file: Optional[str] = field(
        default="data/aligned_train.jsonl",
        metadata={"help": "The input training data file (a jsonl file)."}
    )
    max_len: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )


class SubqueryFineTuner:
    """
    子查询微调器类，用于微调模型以优化子查询生成，使用前需要调用torch.npu.set_device设置NPU设备
    """
    
    def __init__(
        self,
        finetune_args: FineTuneArguments,
        train_args: TrainingArguments,
    ):
        """
        初始化SubqueryFineTuner
        
        Args:
            finetune_args: 模型微调参数
            train_args: 训练参数
        """
        self.finetune_args = finetune_args
        self.train_args = train_args
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
        self.collator: Optional[_ChainOfRagCollator] = None
        self.trainer: Optional[Trainer] = None
        logger.info(f"Initialized SubqueryFineTuner with model: {finetune_args.model_name_or_path}")
    
    def _prepare_model_and_tokenizer(self):
        """
        准备模型和分词器
        """
        logger.info(f"Loading model from: {self.finetune_args.model_name_or_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.finetune_args.model_name_or_path,
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.finetune_args.model_name_or_path,
                pad_token='</s>',
                local_files_only=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise

    def _prepare_datasets(self) -> Dataset:
        """
        准备数据集
        
        Returns:
            训练数据集
        """
        logger.info(f"Loading dataset from: {self.finetune_args.train_file}")
        data_files = {}
        if self.finetune_args.train_file is not None:
            data_files["train"] = self.finetune_args.train_file
        try:
            dataset = load_dataset("json", data_files=data_files, split="train")
            logger.info(f"Loaded dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _initialize_trainer(self, train_dataset: Dataset):
        """
        初始化训练器
        
        Args:
            train_dataset: 训练数据集
        """
        logger.info("Initializing trainer")
        try:
            self.collator = _ChainOfRagCollator(tokenizer=self.tokenizer, max_len=self.finetune_args.max_len)
            self.trainer = Trainer(
                model=self.model,
                args=self.train_args,
                train_dataset=train_dataset,
                data_collator=self.collator,
            )
            logger.info("Trainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise

    def train(self):
        """
        训练模型
        """
        logger.info("Starting training process")
        try:
            self._prepare_model_and_tokenizer()
            train_dataset = self._prepare_datasets()
            self._initialize_trainer(train_dataset)
            
            if(self.train_args.do_train):
                logger.info("Beginning model training")
                self.trainer.train()
                logger.info("Training completed")
                
                logger.info(f"Saving model to: {self.train_args.output_dir}")
                self.trainer.save_model(output_dir=self.train_args.output_dir)
                self.tokenizer.save_pretrained(self.train_args.output_dir)
                logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        logger.info("Training process finished")
