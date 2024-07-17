# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import shlex
import subprocess

from loguru import logger

from mx_rag.utils.file_check import FileCheck, SecFileCheck

MAX_FILE_SIZE_100M = 100 * 1024 * 1024


def train_reranker(origin_reranker_model_path: str,
                   output_reranker_model_path: str,
                   train_data_path: str):
    FileCheck.dir_check(origin_reranker_model_path)
    FileCheck.dir_check(output_reranker_model_path)
    SecFileCheck(train_data_path, MAX_FILE_SIZE_100M).check()

    command = f"""/usr/local/bin/torchrun --nproc_per_node 1 \
-m FlagEmbedding.reranker.run \
--model_name_or_path {origin_reranker_model_path} \
--output_dir {output_reranker_model_path} \
--train_data {train_data_path} \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10 """

    ret = subprocess.run([c.strip() for c in shlex.split(command)])
    if ret.returncode == 0:
        logger.info("train reranker model success")
        return True
    else:
        logger.error("train reranker model failed")
        return False
