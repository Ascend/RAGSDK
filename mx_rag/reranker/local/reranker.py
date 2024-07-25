# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List

from loguru import logger
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, is_torch_npu_available

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.file_check import FileCheck

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except Exception as e:
    logger.warning(f"import torch_npu failed:{e}, LocalReranker will running on cpu")


class LocalReranker(Reranker):
    TEXT_MAX_LEN = 1000 * 1000

    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 k: int = 1,
                 use_fp16: bool = True):
        super(LocalReranker, self).__init__(k)
        self.model_path = model_path
        FileCheck.dir_check(self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

        if use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to(f'npu:{dev_id}')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            raise KeyError("model_path param error. ")

        return LocalReranker(**kwargs)

    def rerank(self,
               query: str,
               texts: List[str],
               batch_size: int = 32,
               max_length: int = 512) -> np.array:
        if len(texts) == 0:
            return np.array([])
        elif len(texts) > LocalReranker.TEXT_MAX_LEN:
            logger.error(f'texts list length must less than {LocalReranker.TEXT_MAX_LEN}')
            return np.array([])

        sentence_pairs = [[query, text] for text in texts]

        result = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            sentence_batch = sentence_pairs[start_index:start_index + batch_size]

            encode_pairs = self.tokenizer(
                sentence_batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(
                self.model.device)

            with torch.no_grad():
                model_output = self.model(**encode_pairs, return_dict=True).logits.view(-1, ).float()

            scores = model_output.cpu().numpy().tolist()
            result = result + scores

        return np.array(result)