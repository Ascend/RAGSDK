# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import shlex
import subprocess

from loguru import logger

from mx_rag.utils.file_check import FileCheck, SecFileCheck

MAX_FILE_SIZE_100M = 100 * 1024 * 1024


def train_embedding(origin_emb_model_path: str,
                    output_emb_model_path: str,
                    train_data_path: str):
    FileCheck.dir_check(origin_emb_model_path)
    FileCheck.dir_check(output_emb_model_path)
    SecFileCheck(train_data_path, MAX_FILE_SIZE_100M).check()

    command = f"""torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--model_name_or_path {origin_emb_model_path} \
--output_dir {output_emb_model_path} \
--train_data {train_data_path} \
--learning_rate 2e-5 \
--fp16 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" """

    ret = subprocess.run(shlex.split(command))
    if ret.returncode == 0:
        logger.info("train embedding model success")
        return True
    else:
        logger.info("train embedding model failed")
        return False
